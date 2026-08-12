"""Transport: one shared clock driving the mixer into the audio device.

All stems share this single clock, which is what keeps them in sync. There is
deliberately no per-stem position: independent clocks drift.
"""

import enum
import logging
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from src.audio.mixer import StemMixer
from src.audio.types import StemSet

logger = logging.getLogger(__name__)

DEFAULT_BLOCKSIZE = 1024


class TransportState(enum.Enum):
    """Where a Transport is in its play/pause/finished lifecycle."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


class Transport:
    """Drives a StemMixer into a sounddevice OutputStream from a worker thread.

    Uses a blocking stream rather than a callback: `OutputStream.write` blocks
    until PortAudio has room, so PortAudio's own buffer provides the cushion and
    no lock is ever taken on the audio thread. That matters because rendering a
    new stretch rate holds the GIL for seconds, which would starve a callback.
    """

    def __init__(
        self,
        mixer: StemMixer,
        blocksize: int = DEFAULT_BLOCKSIZE,
        device: Optional[int] = None,
    ):
        self._mixer = mixer
        self._blocksize = blocksize
        self._lock = threading.Lock()
        self._frame = 0
        self._pending_seek: Optional[int] = None
        self._state = TransportState.STOPPED
        self._running = True

        self._stream = sd.OutputStream(
            samplerate=mixer.stems.samplerate,
            channels=2,
            dtype="float32",
            blocksize=blocksize,
            device=device,
            latency="low",
        )
        self._thread = threading.Thread(target=self._run, name="transport", daemon=True)
        self._thread.start()

    # --- state ---------------------------------------------------------

    @property
    def state(self) -> TransportState:
        with self._lock:
            return self._state

    @property
    def is_playing(self) -> bool:
        """Shorthand for `state is TransportState.PLAYING`."""
        return self.state is TransportState.PLAYING

    @property
    def duration(self) -> float:
        return self._mixer.stems.duration

    @property
    def position(self) -> float:
        """Playback position in ORIGINAL song seconds.

        `_frame` counts what has been handed to PortAudio, which runs ahead of
        what the speakers have actually produced, so the device latency is
        subtracted to keep the displayed clock honest.
        """
        with self._lock:
            frame = self._frame
        stems = self._mixer.stems
        latency_frames = int(self._latency() * stems.samplerate)
        audible = max(0, min(frame - latency_frames, stems.n_frames))
        return stems.frames_to_seconds(audible)

    def _latency(self) -> float:
        try:
            return float(self._stream.latency)
        except Exception:  # pragma: no cover - depends on the backend
            return 0.0

    # --- controls ------------------------------------------------------

    def play(self) -> None:
        """Start or resume playback, rewinding first if it had finished."""
        with self._lock:
            if self._state in (TransportState.FINISHED, TransportState.STOPPED):
                if self._frame >= self._mixer.stems.n_frames:
                    self._frame = 0
            self._state = TransportState.PLAYING
        if not self._stream.active:
            self._stream.start()

    def pause(self) -> None:
        """Pause playback in place; `play()` resumes from the same frame."""
        with self._lock:
            if self._state is TransportState.PLAYING:
                self._state = TransportState.PAUSED
        self._safe_stop_stream()

    def toggle(self) -> None:
        """Pause if playing, otherwise play."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        """Stop playback and rewind to the start."""
        with self._lock:
            self._state = TransportState.STOPPED
            self._frame = 0
            self._pending_seek = None
        self._safe_stop_stream()

    def seek(self, seconds: float) -> None:
        """Jump to a position given in original song seconds."""
        stems = self._mixer.stems
        target = max(0.0, min(seconds, stems.duration))
        with self._lock:
            self._pending_seek = stems.seconds_to_frames(target)
            if self._state is TransportState.FINISHED:
                self._state = TransportState.PAUSED

    def seek_relative(self, delta: float) -> None:
        """Seek by `delta` seconds relative to the current position."""
        self.seek(self.position + delta)

    def swap_stems(self, stems: StemSet) -> None:
        """Swap in re-rendered buffers (a new stretch rate) without moving.

        The read-compute-swap-write sequence is one critical section: releasing
        the transport lock between computing `here` and writing the new `_frame`
        would let the worker thread render one block against the new stems at
        the stale `_frame` value (wrong rate's frame space). Nesting the mixer's
        own lock inside ours is safe -- the mixer never calls back into the
        transport, so lock order is always transport-then-mixer.
        """
        with self._lock:
            here = self._mixer.stems.frames_to_seconds(self._frame)
            self._mixer.swap_stems(stems)
            self._frame = stems.seconds_to_frames(here)
            self._pending_seek = None

    def close(self) -> None:
        """Stop the worker thread and release the audio stream."""
        self._running = False
        self._safe_stop_stream()
        self._thread.join(timeout=2.0)
        try:
            self._stream.close()
        except Exception:  # pragma: no cover
            pass

    def _safe_stop_stream(self) -> None:
        try:
            if self._stream.active:
                self._stream.stop()
        except Exception:  # pragma: no cover
            pass

    # --- worker ---------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            with self._lock:
                if self._pending_seek is not None:
                    self._frame = self._pending_seek
                    self._pending_seek = None
                state = self._state
                frame = self._frame

            if state is not TransportState.PLAYING:
                time.sleep(0.01)
                continue

            stems = self._mixer.stems
            if frame >= stems.n_frames:
                with self._lock:
                    self._state = TransportState.FINISHED
                self._safe_stop_stream()
                continue

            block = self._mixer.render(frame, self._blocksize)

            try:
                if not self._stream.active:
                    self._stream.start()
                self._stream.write(block)
            except sd.PortAudioError as exc:  # pragma: no cover - device churn
                logger.debug("Audio write interrupted: %s", exc)
                time.sleep(0.01)
                continue

            with self._lock:
                # A seek that landed while we were blocked in write() wins.
                if self._pending_seek is None and self._state is TransportState.PLAYING:
                    self._frame = frame + self._blocksize


class OfflineTransport:
    """Renders through the mixer with no audio device, for tests."""

    def __init__(self, mixer: StemMixer, blocksize: int = DEFAULT_BLOCKSIZE):
        self._mixer = mixer
        self._blocksize = blocksize

    def render_all(self) -> np.ndarray:
        """Render the whole mixer output to one buffer, block by block."""
        stems = self._mixer.stems
        blocks = []
        for start in range(0, stems.n_frames, self._blocksize):
            blocks.append(self._mixer.render(start, self._blocksize))
        return np.concatenate(blocks)[: stems.n_frames]
