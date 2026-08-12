"""Pitch-preserving time stretching.

Speed is changed by resampling the *spectrogram* time axis with a phase vocoder
and rescaling the per-bin phase advance. Bin centre frequencies never move, so
the pitch is preserved -- which is the whole point, and the reason this is not
simply a resample.
"""

import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torchaudio

from src.audio.types import StemSet

logger = logging.getLogger(__name__)

# Speed is exposed as integer steps so the UI can never land between values.
RATE_STEP = 0.01           # 1% per step
MIN_STEP = -40             # -40%
MAX_STEP = 40              # +40%

DEFAULT_N_FFT = 2048

# Hop as a fraction of the window (75% overlap -- hop = n_fft/4). A finer hop
# (e.g. n_fft/8) measurably reduces the "phasiness" torchaudio's phase_vocoder
# picks up on sustained tones, since it has no phase locking -- but it was
# tried and reverted: it roughly doubles render time (confirmed empirically:
# a 4-min song's worst case, slowing down, went from ~4.8s to ~8.1s), and a
# speed change is debounced but still blocks the audible switch until it's
# done, so that latency is directly felt on every speed change. Not worth it
# next to the much bigger, latency-free quality win from separation's own
# shifts=5 (see separator/engine.py).
OVERLAP = 4

# Percussive stems keep their transients better with a shorter window. Output
# length is forced explicitly, so mixing window sizes cannot desynchronise them.
STEM_N_FFT = {
    "drums": 1024,
    "bass": 1024,
}


def rate_from_step(step: int) -> float:
    """Step -> playback rate. 1.40 plays 40% faster, 0.60 plays 40% slower."""
    return 1.0 + step * RATE_STEP


def step_from_rate(rate: float) -> int:
    """Playback rate -> step. Inverse of `rate_from_step`."""
    return int(round((rate - 1.0) / RATE_STEP))


def clamp_step(step: int) -> int:
    """Clamp a step to the supported [-40%, +40%] range."""
    return max(MIN_STEP, min(MAX_STEP, int(step)))


def format_step(step: int) -> str:
    """Human label for a speed step, e.g. '-10.0%' or 'normal'."""
    if step == 0:
        return "100.0%"
    return f"{rate_from_step(step) * 100:.1f}%"


def stretch_buffer(
    data: np.ndarray,
    rate: float,
    length: Optional[int] = None,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: Optional[int] = None,
) -> np.ndarray:
    """Time-stretch (n, 2) float32 audio by `rate`, preserving pitch.

    `length` forces the exact output frame count so every stem of a set comes
    out identical in length regardless of its window size.
    """
    if abs(rate - 1.0) < 1e-9:
        return np.ascontiguousarray(data, dtype=np.float32)

    if hop_length is None:
        hop_length = max(1, n_fft // OVERLAP)

    signal = torch.from_numpy(np.ascontiguousarray(data.T, dtype=np.float32))
    window = torch.hann_window(n_fft)

    spec = torch.stft(
        signal, n_fft, hop_length=hop_length, window=window, return_complex=True
    )
    phase_advance = torch.linspace(0, math.pi * hop_length, spec.shape[-2])[..., None]
    stretched = torchaudio.functional.phase_vocoder(spec, rate, phase_advance)

    if length is None:
        length = int(round(data.shape[0] / rate))

    out = torch.istft(
        stretched, n_fft, hop_length=hop_length, window=window, length=length
    )
    return np.ascontiguousarray(out.numpy().T, dtype=np.float32)


def stretch_stem_set(stems: StemSet, step: int) -> StemSet:
    """Render a whole StemSet at a speed step, keeping every stem aligned."""
    rate = rate_from_step(step)
    if step == 0:
        return stems

    target_len = int(round(stems.n_frames / rate))
    started = time.monotonic()

    buffers: Dict[str, np.ndarray] = {}
    for name, data in stems.buffers.items():
        buffers[name] = stretch_buffer(
            data,
            rate,
            length=target_len,
            n_fft=STEM_N_FFT.get(name, DEFAULT_N_FFT),
        )

    logger.debug(
        "Stretched %d stems to %s in %.2fs",
        len(buffers), format_step(step), time.monotonic() - started,
    )
    return StemSet.create(buffers, stems.samplerate, rate=rate)


class StretchController:
    """Renders speed changes off the audio path, debounced and cached.

    Rendering a whole set takes a few seconds and holds the GIL, so it must
    never happen on the thread feeding the device. Playback continues at the old
    rate until the new one is ready, then swaps in at the same position.
    """

    DEBOUNCE_SECONDS = 0.35

    def __init__(
        self,
        base: StemSet,
        on_ready: Callable[[StemSet], None],
        cache_size: int = 2,
    ):
        self._base = base
        self._on_ready = on_ready
        self._cache_size = cache_size
        self._cache: "OrderedDict[int, StemSet]" = OrderedDict()

        self._lock = threading.Lock()
        self._target_step = 0
        self._applied_step = 0
        self._rendering = False
        self._running = True
        self._wake = threading.Event()

        self._thread = threading.Thread(target=self._worker, name="stretch", daemon=True)
        self._thread.start()

    @property
    def applied_step(self) -> int:
        with self._lock:
            return self._applied_step

    @property
    def target_step(self) -> int:
        with self._lock:
            return self._target_step

    @property
    def is_rendering(self) -> bool:
        with self._lock:
            return self._rendering

    def request_step(self, step: int) -> None:
        """Ask for a new speed step; the worker thread debounces and renders it."""
        with self._lock:
            self._target_step = clamp_step(step)
        self._wake.set()

    def nudge(self, delta: int) -> None:
        """Request the current target step plus `delta`."""
        self.request_step(self.target_step + delta)

    def reset(self) -> None:
        """Request a return to normal (100%) speed."""
        self.request_step(0)

    def close(self) -> None:
        """Stop the worker thread and wait for it to exit."""
        self._running = False
        self._wake.set()
        self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        while self._running:
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            if not self._running:
                return

            # Coalesce a burst of keypresses into a single render.
            time.sleep(self.DEBOUNCE_SECONDS)

            with self._lock:
                target = self._target_step
                if target == self._applied_step:
                    continue
                self._rendering = True

            try:
                stems = self._render(target)
            except Exception:
                logger.exception("Time stretch failed for step %s", target)
                with self._lock:
                    self._rendering = False
                continue

            with self._lock:
                self._rendering = False
                # A newer request arrived mid-render; let the next pass win.
                stale = self._target_step != target
                if not stale:
                    self._applied_step = target
            if not stale:
                self._on_ready(stems)
            else:
                self._wake.set()

    def _render(self, step: int) -> StemSet:
        if step == 0:
            return self._base
        with self._lock:
            cached = self._cache.get(step)
        if cached is not None:
            return cached

        stems = stretch_stem_set(self._base, step)
        with self._lock:
            self._cache[step] = stems
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return stems
