"""Source -> separation -> cache. The single road every song takes."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from src.library.cache import SongCache, StemManifest, song_key
from src.library.source import SongSource
from src.separator.engine import SeparationEngine
from src.separator.models import DEMUCS_MODEL

logger = logging.getLogger(__name__)


@dataclass
class Progress:
    """One step of a `SeparationPipeline.run()` call, for progress callbacks.

    `fraction` is only set during the "separate" stage, as the underlying model
    reports chunks completed; every other stage leaves it None.
    """

    stage: str
    message: str
    fraction: Optional[float] = None


ProgressCallback = Optional[Callable[[Progress], None]]


class SeparationPipeline:
    """Separates a source into cached stems, reusing earlier work."""

    def __init__(self, cache: Optional[SongCache] = None, on_progress: ProgressCallback = None):
        self.cache = cache or SongCache()
        self._on_progress = on_progress

    def _report(self, stage: str, message: str, fraction: Optional[float] = None) -> None:
        if self._on_progress is not None:
            self._on_progress(Progress(stage, message, fraction))

    def run(self, source: SongSource, force: bool = False) -> StemManifest:
        """Return cached stems if present, otherwise separate and cache them.

        `force=True` re-separates and overwrites an existing cache entry.
        """
        self._report("hash", "Identifying song...")
        key = song_key(source.path)

        if not force:
            cached = self.cache.get(key)
            if cached is not None:
                self._report("cached", "Loading previously separated stems")
                return cached

        self._report("separate", "Separating into stems...")
        engine = SeparationEngine(DEMUCS_MODEL)
        engine.load_model()
        buffers, samplerate = engine.separate_buffers(
            source.path,
            on_progress=lambda fraction: self._report("separate", "Separating into stems...", fraction),
        )

        self._report("store", f"Caching {len(buffers)} stems...")
        manifest = self.cache.store(key, source.title, buffers, samplerate)

        self._report("done", "Ready")
        return manifest
