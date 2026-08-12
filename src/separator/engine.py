"""Wraps demucs for stem separation."""

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from demucs.api import Separator as DemucsAPI

from src.separator.models import ModelConfig

logger = logging.getLogger(__name__)

# demucs.api.Separator itself defaults to device="cpu" -- it's demucs's own CLI
# (separate.py) that picks cuda/mps when available, so bypassing that CLI means
# reimplementing its device choice here or silently running CPU-only even on
# machines with a GPU.
# shifts=5, overlap=0.5 mirror demucs's own "--shifts" guidance (the paper used
# 10; the CLI's own default of 1 is a speed compromise, not a quality one) --
# averaging over several random shifts measurably reduces separation artifacts.
# Confirmed empirically: on Apple Silicon (mps) this lands at roughly the same
# wall-clock time the previous shifts=1 defaults took on CPU.
SEPARATION_SHIFTS = 5
SEPARATION_OVERLAP = 0.5


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _progress_callback(on_progress: Callable[[float], None], shifts: int) -> Callable[[dict], None]:
    """Adapt demucs's per-chunk callback into one overall 0..1 fraction.

    demucs reports `shift_idx`/`segment_offset`/`model_idx_in_bag` per chunk but
    never the totals needed for a single percentage. `shifts` is ours to supply
    since we're the ones who set it; `segment_offset / audio_length` approximates
    progress through one shift well enough for a progress bar, without depending
    on demucs's internal segment-length/stride math (private API, and only
    reachable through the model actually wrapped in a BagOfModels).
    """
    def callback(info: dict) -> None:
        if info.get("state") != "end":
            return
        audio_length = info.get("audio_length") or 1
        within_shift = min(1.0, info.get("segment_offset", 0) / audio_length)
        per_shift = (info.get("shift_idx", 0) + within_shift) / max(1, shifts)
        models = info.get("models") or 1
        overall = (info.get("model_idx_in_bag", 0) + per_shift) / models
        on_progress(min(1.0, max(0.0, overall)))

    return callback


class SeparationEngine:
    """Loads htdemucs_6s and separates a file into in-memory stem buffers."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._separator: Optional[DemucsAPI] = None

    def load_model(self) -> None:
        """Load the model, downloading it automatically on first use."""
        device = _best_device()
        logger.info(
            "Loading model: %s on %s (this may take a moment on first use)",
            self.model_config.name, device,
        )
        self._separator = DemucsAPI(
            model=self.model_config.model_filename,
            device=device,
            shifts=SEPARATION_SHIFTS,
            overlap=SEPARATION_OVERLAP,
        )

    def separate_buffers(
        self,
        audio_path: str,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """Separate into in-memory (n, 2) float32 buffers, one per stem.

        Buffers are returned unscaled: callers that persist them are responsible
        for applying one shared headroom factor across all stems. `on_progress`,
        if given, is called with a 0..1 fraction as chunks complete -- demucs's
        callback runs synchronously on this thread (jobs=0), so no locking is
        needed to call it safely.
        """
        logger.info("Processing with %s", self.model_config.name)
        if on_progress is not None:
            self._separator.update_parameter(
                callback=_progress_callback(on_progress, SEPARATION_SHIFTS)
            )
        _origin, separated = self._separator.separate_audio_file(audio_path)

        buffers = {
            stem: np.ascontiguousarray(source.cpu().numpy().T, dtype=np.float32)
            for stem, source in separated.items()
        }
        return buffers, self._separator.samplerate
