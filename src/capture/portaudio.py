"""PortAudio-based capture, shared by the macOS and Windows device backends."""

import logging
import queue
import threading
from typing import List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.audio.devices import (
    AGGREGATE_HINTS,
    default_output_name,
    input_devices,
    looks_like_loopback,
    native_samplerate,
)
from src.capture.base import (
    BackendUnavailable,
    CaptureBackend,
    CaptureConfig,
    CaptureResult,
    CaptureSource,
    SourceKind,
)

logger = logging.getLogger(__name__)

BLACKHOLE_REMEDY = """To capture system audio on macOS you need a virtual loopback device:

  1. brew install --cask blackhole-2ch
  2. Restart your Mac.
  3. Open 'Audio MIDI Setup' and create a Multi-Output Device containing
     BOTH your speakers AND BlackHole 2ch.
  4. Set that Multi-Output Device as the system output (Sound settings).
     Without this step you would record audio but hear nothing.
  5. Come back here and pick 'BlackHole 2ch' as the source.

Note: Spotify and Apple Music have no per-app output selector, so this
route captures all system audio, not one application."""


ROUTING_REMEDY = """Nothing is being sent to this loopback device, so you would record silence.

Route the system's audio through it:

  1. Open 'Audio MIDI Setup' (in /Applications/Utilities).
  2. Click + at the bottom left -> 'Create Multi-Output Device'.
  3. Tick BOTH your speakers/headphones AND '{device}'.
     Put your speakers first so they act as the clock source.
  4. In System Settings > Sound > Output, select that Multi-Output Device.

You will then hear the audio AND be able to record it. Selecting '{device}'
directly as the system output also works, but you would hear nothing."""

SILENT_INPUT_REMEDY = """The input delivered digital silence — not even a noise floor.

macOS does not raise an error when microphone access is denied; it hands the
app an endless stream of zeros, which looks exactly like this.

  1. Open System Settings > Privacy & Security > Microphone.
  2. Enable access for the terminal app you launched this from
     (Terminal, iTerm, Warp, VS Code...).
  3. Quit that app COMPLETELY and reopen it — the permission is only
     picked up on a fresh launch.

If access is already granted, check that the source is actually playing:
a loopback device is silent when nothing is routed into it."""


def routing_warning(device_name: str) -> Optional[str]:
    """Explain why a loopback source would capture silence, if it would.

    Installing BlackHole is only half the job: unless the system output is
    routed into it, the device is real, selectable, and completely silent.
    """
    if not looks_like_loopback(device_name):
        return None

    output = default_output_name()
    lowered = output.lower()

    if device_name.lower() in lowered or any(h in lowered for h in AGGREGATE_HINTS):
        return None

    return ROUTING_REMEDY.format(device=device_name)


class PortAudioRecorder(CaptureBackend):
    """Records an input device to a file.

    The audio callback only copies into a queue; a writer thread does the file
    I/O. Writing inside the callback causes dropouts.
    """

    name = "portaudio"
    supports_per_application = False

    def __init__(self) -> None:
        self._stream: Optional[sd.InputStream] = None
        self._file: Optional[sf.SoundFile] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._writer: Optional[threading.Thread] = None
        self._running = False
        self._frames = 0
        self._samplerate = 44100
        self._channels = 2
        self._peak_window = 0.0
        self._peak_overall = 0.0
        self._path = ""
        self._overflows = 0
        self._saw_signal = False

    @classmethod
    def is_available(cls) -> bool:
        """True if PortAudio can enumerate devices at all on this machine."""
        try:
            sd.query_devices()
            return True
        except Exception:  # pragma: no cover - no audio backend at all
            return False

    def list_sources(self) -> List[CaptureSource]:
        """Every input device, classified as system/microphone/device."""
        sources: List[CaptureSource] = []

        for device in input_devices():
            if device.is_loopback:
                kind = SourceKind.SYSTEM
                detail = "captures everything playing on this computer"
            elif "microphone" in device.name.lower() or device.channels == 1:
                kind = SourceKind.MICROPHONE
                detail = ""
            else:
                kind = SourceKind.DEVICE
                detail = ""

            sources.append(
                CaptureSource(
                    id=str(device.index),
                    name=device.name,
                    kind=kind,
                    backend=self.name,
                    channels=device.channels,
                    samplerate=device.samplerate,
                    detail=detail,
                )
            )

        return sources

    def has_loopback_source(self) -> bool:
        """True if a loopback device (BlackHole, Stereo Mix, ...) is present."""
        return any(s.kind is SourceKind.SYSTEM for s in self.list_sources())

    def start(self, config: CaptureConfig) -> None:
        """Open the device and start writing to `config.output_path`."""
        if self._running:
            raise RuntimeError("Already recording")

        device_index = int(config.source.id)
        self._samplerate = config.samplerate or native_samplerate(device_index)
        self._channels = config.channels or config.source.channels or 1
        self._path = config.output_path
        self._frames = 0
        self._peak_window = 0.0
        self._peak_overall = 0.0
        self._overflows = 0
        self._saw_signal = False
        self._queue = queue.Queue()

        try:
            self._file = sf.SoundFile(
                config.output_path,
                mode="w",
                samplerate=self._samplerate,
                channels=self._channels,
                subtype=config.subtype,
            )
            self._stream = sd.InputStream(
                device=device_index,
                channels=self._channels,
                samplerate=self._samplerate,
                dtype="float32",
                blocksize=1024,
                callback=self._callback,
            )
        except sd.PortAudioError as exc:
            self._cleanup()
            raise BackendUnavailable(
                f"Could not open '{config.source.name}': {exc}",
                remedy=(
                    "Check that no other app has exclusive use of the device, "
                    "and that microphone access is granted in "
                    "System Settings > Privacy & Security > Microphone."
                ),
            ) from exc

        self._running = True
        self._writer = threading.Thread(target=self._drain, name="capture-writer", daemon=True)
        self._writer.start()
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status and status.input_overflow:
            self._overflows += 1
        self._queue.put(indata.copy())

    def _drain(self) -> None:
        while self._running or not self._queue.empty():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            peak = float(np.abs(block).max())
            self._peak_window = max(self._peak_window, peak)
            self._peak_overall = max(self._peak_overall, peak)
            self._frames += len(block)
            if peak > 0.0:
                self._saw_signal = True

            if self._file is not None:
                self._file.write(block)

    @property
    def elapsed(self) -> float:
        """Seconds recorded so far."""
        return self._frames / self._samplerate if self._samplerate else 0.0

    def level(self) -> float:
        """Peak level since the last call, then reset the window."""
        peak = self._peak_window
        self._peak_window = 0.0
        return peak

    @property
    def saw_signal(self) -> bool:
        """Whether any non-zero sample ever arrived.

        Exactly-zero input is how macOS reports a denied microphone
        permission, so this distinguishes 'silent' from 'blocked'.
        """
        return self._saw_signal

    def stop(self) -> CaptureResult:
        """Stop the stream, flush the writer thread and close the file."""
        if not self._running:
            raise RuntimeError("Not recording")

        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._writer is not None:
            self._writer.join(timeout=3.0)
            self._writer = None

        duration = self.elapsed
        self._cleanup()

        if self._overflows:
            logger.warning("Recording had %d input overflows", self._overflows)

        return CaptureResult(
            path=self._path,
            duration=duration,
            peak=self._peak_overall,
            samplerate=self._samplerate,
            channels=self._channels,
        )

    def _cleanup(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
