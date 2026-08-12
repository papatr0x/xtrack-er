"""Audio file I/O. Writing always goes through soundfile so ffmpeg is never
needed for that. Reading for separation is demucs's job, not this module's --
see the note on EXTRA_CONTAINER_EXTENSIONS below.
"""

import os
from typing import List

import numpy as np
import soundfile as sf

# Compressed containers demucs's own loader may still decode -- first via
# sphn (Symphonia), then via ffmpeg if that's on PATH -- even though
# libsndfile can't touch them directly. Accepting them here doesn't promise
# they'll work: if neither backend can decode the file, separation itself
# raises a clear error explaining why (e.g. "ffmpeg is not installed").
# Rejecting them here on libsndfile's authority alone would be wrong, since
# libsndfile is never actually used to read the separation input.
EXTRA_CONTAINER_EXTENSIONS = {
    ".m4a", ".aac", ".mp4", ".wma", ".opus", ".webm", ".ac3", ".wv", ".mka",
}


def supported_extensions() -> List[str]:
    """Extensions libsndfile can read directly, with no external decoder."""
    return sorted("." + fmt.lower() for fmt in sf.available_formats())


def is_supported(path: str) -> bool:
    """Whether this file is worth attempting at all (see module docstring)."""
    ext = os.path.splitext(path)[1].lower()
    return ext in supported_extensions() or ext in EXTRA_CONTAINER_EXTENSIONS


def write_audio(path: str, data: np.ndarray, samplerate: int, subtype: str = "PCM_24") -> str:
    """Write (n, channels) float32 audio. Caller is responsible for headroom."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, data, samplerate, subtype=subtype)
    return path
