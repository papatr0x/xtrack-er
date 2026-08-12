"""Content-addressed cache of separated stems.

Separation costs minutes, so a song is separated once and reloaded thereafter.
The key is derived from file *content*, so renaming or moving a source file --
or re-importing the same recording -- still hits the cache.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf

from src.audio.io import write_audio
from src.audio.types import StemSet
from src.library.paths import cache_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# Bump when the models or pass structure change, to invalidate old entries.
PIPELINE_VERSION = "htdemucs_6s-v1"

STEM_SUBTYPE = "PCM_24"
CHUNK = 1024 * 1024


@dataclass
class StemManifest:
    """Everything needed to load one song's separated stems back from disk."""

    source_name: str
    source_key: str
    samplerate: int
    n_frames: int
    stems: Dict[str, str]
    peak_scale: float = 1.0
    schema_version: int = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    directory: str = ""

    def is_compatible(self) -> bool:
        """False if this entry predates the current schema or pipeline version."""
        return (
            self.schema_version == SCHEMA_VERSION
            and self.pipeline_version == PIPELINE_VERSION
        )


def song_key(path: str) -> str:
    """Hash of file size plus its first and last megabyte.

    Cheap on large files and stable across renames, which is what matters here;
    this is a cache key, not a security check.
    """
    size = os.path.getsize(path)
    digest = hashlib.sha1(str(size).encode())

    with open(path, "rb") as handle:
        digest.update(handle.read(CHUNK))
        if size > CHUNK:
            handle.seek(max(0, size - CHUNK))
            digest.update(handle.read(CHUNK))

    return digest.hexdigest()[:16]


class SongCache:
    """Stores one directory of FLAC stems plus a manifest per song."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        """Directory holding one song's manifest and stem files."""
        return self.root / key

    def get(self, key: str) -> Optional[StemManifest]:
        """Load a cache entry, or None if it's missing, stale or incomplete."""
        manifest_path = self.path_for(key) / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            data = json.loads(manifest_path.read_text())
            manifest = StemManifest(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Ignoring unreadable cache entry %s: %s", key, exc)
            return None

        if not manifest.is_compatible():
            logger.info("Cache entry %s is from an older pipeline; re-separating", key)
            return None

        missing = [
            name for name, filename in manifest.stems.items()
            if not (self.path_for(key) / filename).exists()
        ]
        if missing:
            logger.warning("Cache entry %s is missing stems %s", key, missing)
            return None

        return manifest

    def store(
        self,
        key: str,
        source_name: str,
        buffers: Dict[str, np.ndarray],
        samplerate: int,
    ) -> StemManifest:
        """Write stems as 24-bit FLAC with one shared headroom scale."""
        directory = self.path_for(key)
        directory.mkdir(parents=True, exist_ok=True)

        # Separated stems can exceed +/-1.0 and libsndfile wraps rather than
        # clipping on float-to-int conversion. Scale every stem by the SAME
        # factor so their relative balance survives; per-stem normalisation
        # would ruin the mix when they are played together.
        peak = max(float(np.abs(buf).max()) for buf in buffers.values())
        scale = 1.0 / peak if peak > 1.0 else 1.0

        stems: Dict[str, str] = {}
        n_frames = min(len(buf) for buf in buffers.values())

        for name, buf in buffers.items():
            filename = f"{name}.flac"
            data = np.ascontiguousarray(buf[:n_frames], dtype=np.float32)
            if scale != 1.0:
                data = data * scale
            write_audio(str(directory / filename), data, samplerate, subtype=STEM_SUBTYPE)
            stems[name] = filename

        manifest = StemManifest(
            source_name=source_name,
            source_key=key,
            samplerate=samplerate,
            n_frames=n_frames,
            stems=stems,
            peak_scale=peak if scale != 1.0 else 1.0,
            directory=str(directory),
        )
        payload = asdict(manifest)
        (directory / "manifest.json").write_text(json.dumps(payload, indent=2))
        return manifest

    def load_stem_set(self, manifest: StemManifest) -> StemSet:
        """Read a manifest's stems back, undoing the stored headroom scale."""
        directory = self.path_for(manifest.source_key)
        buffers: Dict[str, np.ndarray] = {}

        for name, filename in manifest.stems.items():
            data, _ = sf.read(str(directory / filename), dtype="float32", always_2d=True)
            if manifest.peak_scale != 1.0:
                data = data * manifest.peak_scale
            buffers[name] = data

        return StemSet.create(buffers, manifest.samplerate)

    def entries(self):
        """Every usable cache entry, newest first."""
        found = []
        for directory in self.root.iterdir():
            if directory.is_dir():
                manifest = self.get(directory.name)
                if manifest is not None:
                    found.append(manifest)
        return sorted(found, key=lambda m: m.created_at, reverse=True)
