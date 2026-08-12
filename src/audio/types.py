"""Core audio value types shared by the mixer, transport and stretcher."""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

# Canonical display order. Stems not listed here are appended alphabetically.
STEM_ORDER = ("vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other")

STEM_LABELS = {
    "vocals": "Vocals",
    "lead_vocals": "Lead vocals",
    "backing_vocals": "Backing vocals",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "other": "Other",
    "Instrumental": "Instrumental",
    "Vocals": "Vocals",
}


def stem_label(stem: str) -> str:
    """Human-readable name for a stem id."""
    return STEM_LABELS.get(stem, stem.replace("_", " ").capitalize())


def sort_stems(stems) -> list:
    """Order stem ids for display, canonical names first."""
    known = [s for s in STEM_ORDER if s in stems]
    rest = sorted(s for s in stems if s not in STEM_ORDER)
    return known + rest


@dataclass(frozen=True)
class StemSet:
    """A set of stems guaranteed to be sample-aligned and ready to mix.

    Every buffer is (n_frames, 2) float32 and C-contiguous. `rate` records the
    time-stretch factor this set was rendered at: 1.0 is the original tempo,
    0.75 plays 25% slower. Frame indices are positions inside *this* set, so
    converting to original-song seconds always goes through `rate`.
    """

    samplerate: int
    n_frames: int
    buffers: Mapping[str, np.ndarray]
    rate: float = 1.0
    stems: tuple = field(default=(), compare=False)

    MAX_DRIFT_SECONDS = 0.01

    @classmethod
    def create(
        cls,
        buffers: Mapping[str, np.ndarray],
        samplerate: int,
        rate: float = 1.0,
    ) -> "StemSet":
        """Normalise, align and validate raw buffers into a StemSet.

        Trims every stem to the shortest length so they stay sample-aligned, but
        refuses to hide a real bug behind that trim: a drift larger than
        MAX_DRIFT_SECONDS means the stems did not come from the same source.
        """
        if not buffers:
            raise ValueError("StemSet needs at least one stem")

        prepared = {name: _as_stereo_float32(buf) for name, buf in buffers.items()}
        lengths = [len(buf) for buf in prepared.values()]
        shortest, longest = min(lengths), max(lengths)

        drift = (longest - shortest) / samplerate
        if drift > cls.MAX_DRIFT_SECONDS:
            raise ValueError(
                f"Stems are misaligned by {drift:.3f}s "
                f"(max {cls.MAX_DRIFT_SECONDS}s); they are not from one source"
            )

        aligned = {
            name: np.ascontiguousarray(buf[:shortest]) for name, buf in prepared.items()
        }
        return cls(
            samplerate=samplerate,
            n_frames=shortest,
            buffers=aligned,
            rate=rate,
            stems=tuple(sort_stems(aligned)),
        )

    @property
    def duration(self) -> float:
        """Length in ORIGINAL song seconds, independent of the stretch rate."""
        return self.n_frames / self.samplerate * self.rate

    def frames_to_seconds(self, frame: int) -> float:
        """Frame index in this set -> original song seconds."""
        return frame / self.samplerate * self.rate

    def seconds_to_frames(self, seconds: float) -> int:
        """Original song seconds -> frame index in this set."""
        return int(seconds * self.samplerate / self.rate)


def _as_stereo_float32(buf: np.ndarray) -> np.ndarray:
    """Coerce any buffer to (n, 2) float32 without averaging channels away."""
    data = np.asarray(buf, dtype=np.float32)

    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]

    return np.ascontiguousarray(data)
