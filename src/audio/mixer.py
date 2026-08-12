"""Stem mixer: sums the selected stems with per-stem gain and mute."""

import threading
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.audio.types import StemSet

DEFAULT_RAMP_MS = 10.0


@dataclass
class StemState:
    """One stem's mix controls: gain, mute and solo."""

    gain: float = 1.0
    muted: bool = False
    solo: bool = False


class StemMixer:
    """Sums a StemSet into one stereo buffer, honouring mute and gain.

    Gain changes are ramped over a few milliseconds. Without that, toggling a
    stem produces an audible click, and toggling stems is the whole point of
    this application.
    """

    def __init__(self, stems: StemSet, ramp_ms: float = DEFAULT_RAMP_MS):
        self._stems = stems
        self._lock = threading.Lock()
        self._state: Dict[str, StemState] = {s: StemState() for s in stems.stems}
        self._gain_now: Dict[str, float] = {s: 1.0 for s in stems.stems}
        self._master_gain = 1.0
        self._ramp_len = max(1, int(stems.samplerate * ramp_ms / 1000.0))
        self._last_peak = 0.0

    # --- introspection -------------------------------------------------

    @property
    def stems(self) -> StemSet:
        with self._lock:
            return self._stems

    @property
    def stem_ids(self) -> tuple:
        return self._stems.stems

    @property
    def last_peak(self) -> float:
        return self._last_peak

    def snapshot(self) -> Dict[str, StemState]:
        """A copy of every stem's current state, safe to read outside the lock."""
        with self._lock:
            return {s: StemState(st.gain, st.muted, st.solo) for s, st in self._state.items()}

    def is_muted(self, stem: str) -> bool:
        with self._lock:
            return self._state[stem].muted

    def is_solo(self, stem: str) -> bool:
        with self._lock:
            return self._state[stem].solo

    @property
    def has_solo(self) -> bool:
        """True if any stem is currently soloed."""
        with self._lock:
            return any(st.solo for st in self._state.values())

    def active_stems(self) -> List[str]:
        """Stems actually audible right now, mute and solo combined."""
        with self._lock:
            any_solo = any(st.solo for st in self._state.values())
            return [
                s for s, st in self._state.items()
                if not st.muted and (not any_solo or st.solo)
            ]

    # --- control surface -----------------------------------------------

    def set_muted(self, stem: str, muted: bool) -> None:
        """Mute or unmute one stem directly, rather than toggling it."""
        with self._lock:
            if stem in self._state:
                self._state[stem].muted = muted

    def toggle(self, stem: str) -> bool:
        """Flip a stem's mute state and return the new 'audible' value."""
        with self._lock:
            if stem not in self._state:
                return False
            self._state[stem].muted = not self._state[stem].muted
            return not self._state[stem].muted

    def set_all(self, muted: bool) -> None:
        """Mute or unmute every stem.

        Unmuting also clears solo: 'a' is meant to mean everything plays, and
        a leftover solo would silently keep restricting playback to whatever
        was soloed before.
        """
        with self._lock:
            for state in self._state.values():
                state.muted = muted
                if not muted:
                    state.solo = False

    def set_solo(self, stem: str, solo: bool) -> None:
        """Solo or unsolo one stem directly, rather than toggling it."""
        with self._lock:
            if stem in self._state:
                self._state[stem].solo = solo

    def toggle_solo(self, stem: str) -> bool:
        """Flip a stem's solo state and return the new value."""
        with self._lock:
            if stem not in self._state:
                return False
            self._state[stem].solo = not self._state[stem].solo
            return self._state[stem].solo

    def clear_solo(self) -> None:
        """Unsolo every stem, restoring normal mute-only playback."""
        with self._lock:
            for state in self._state.values():
                state.solo = False

    def set_gain(self, stem: str, gain: float) -> None:
        """Set one stem's gain, clamped to [0, 2]."""
        with self._lock:
            if stem in self._state:
                self._state[stem].gain = max(0.0, min(2.0, gain))

    def set_master_gain(self, gain: float) -> None:
        """Set the overall output gain, clamped to [0, 2]."""
        with self._lock:
            self._master_gain = max(0.0, min(2.0, gain))

    @property
    def master_gain(self) -> float:
        with self._lock:
            return self._master_gain

    def swap_stems(self, stems: StemSet) -> None:
        """Replace the buffers (same stems, different stretch rate)."""
        with self._lock:
            self._stems = stems
            for stem in stems.stems:
                self._state.setdefault(stem, StemState())
                self._gain_now.setdefault(stem, 1.0)
            self._ramp_len = max(1, int(stems.samplerate * DEFAULT_RAMP_MS / 1000.0))

    # --- audio path ------------------------------------------------------

    def render(self, start: int, n: int) -> np.ndarray:
        """Mix frames [start, start+n) into a fresh (n, 2) float32 buffer.

        Frames past the end of the material are left as silence, so the caller
        can detect the end by comparing against `stems.n_frames`.
        """
        with self._lock:
            stems = self._stems
            any_solo = any(st.solo for st in self._state.values())
            targets = {
                s: (
                    0.0 if st.muted or (any_solo and not st.solo) else st.gain
                )
                for s, st in self._state.items()
            }
            master = self._master_gain
            ramp_len = self._ramp_len

        out = np.zeros((n, 2), dtype=np.float32)
        start = max(0, start)
        avail = max(0, min(n, stems.n_frames - start))

        for stem, target in targets.items():
            buffer = stems.buffers.get(stem)
            if buffer is None:
                continue

            current = self._gain_now.get(stem, target)
            if current == 0.0 and target == 0.0:
                continue

            if avail > 0:
                chunk = buffer[start:start + avail]
                if current == target:
                    if target == 1.0:
                        out[:avail] += chunk
                    else:
                        out[:avail] += chunk * target
                else:
                    out[:avail] += chunk * _ramp(current, target, avail, ramp_len)

            # The ramp always completes inside one block, so the running gain
            # lands exactly on the target rather than drifting toward it.
            self._gain_now[stem] = target

        if master != 1.0:
            out *= master

        peak = float(np.abs(out).max()) if avail > 0 else 0.0
        self._last_peak = peak
        if peak > 1.0:
            # Soft-clip rather than wrap. Only paid for when actually over.
            np.tanh(out, out=out)

        return out


def _ramp(current: float, target: float, n: int, ramp_len: int) -> np.ndarray:
    """Column vector moving from `current` to `target` over `ramp_len` frames."""
    ramp = np.full((n, 1), target, dtype=np.float32)
    k = min(n, ramp_len)
    ramp[:k, 0] = np.linspace(current, target, k, dtype=np.float32)
    return ramp
