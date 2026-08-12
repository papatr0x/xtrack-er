"""Platform-agnostic audio capture interface.

Backends differ wildly in what they can reach -- a loopback device gives system
audio, ScreenCaptureKit gives one application -- but the app only ever sees
CaptureSource values and this ABC, so adding a platform is additive.
"""

import abc
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional


class SourceKind(str, Enum):
    """What a CaptureSource actually records."""

    SYSTEM = "system"            # everything playing on the machine
    APPLICATION = "application"  # one app (needs a helper on macOS)
    MICROPHONE = "microphone"
    DEVICE = "device"            # audio interface or other input


KIND_LABELS = {
    SourceKind.SYSTEM: "System audio",
    SourceKind.APPLICATION: "Application",
    SourceKind.MICROPHONE: "Microphone",
    SourceKind.DEVICE: "Input device",
}


@dataclass(frozen=True)
class CaptureSource:
    """One thing a backend can record from (a device, an app, system audio)."""

    id: str
    name: str
    kind: SourceKind
    backend: str
    channels: int = 2
    samplerate: Optional[int] = None
    detail: str = ""


@dataclass
class CaptureConfig:
    """Parameters for one recording session."""

    source: CaptureSource
    output_path: str
    # None means "record at the device's own rate". Forcing a rate the device
    # does not run at yields samples written under the wrong label, which plays
    # back at the wrong speed and pitch. Downstream code resamples anyway.
    samplerate: Optional[int] = None
    # None means "use whatever the source has". Mono microphones reject a
    # request for two channels outright.
    channels: Optional[int] = None
    subtype: str = "PCM_24"


@dataclass
class CaptureResult:
    """What a finished recording produced."""

    path: str
    duration: float
    peak: float
    samplerate: int
    channels: int


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot run, carrying user-facing instructions.

    Every failure here is 'you must install or configure something in a GUI',
    so the remedy travels with the error and is printed verbatim.
    """

    def __init__(self, message: str, remedy: str = ""):
        super().__init__(message)
        self.remedy = remedy


class CaptureBackend(abc.ABC):
    """One way to capture audio on a platform (PortAudio device, ScreenCaptureKit, ...)."""

    name: ClassVar[str] = "base"
    supports_per_application: ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Whether this backend can run at all on this machine."""

    @abc.abstractmethod
    def list_sources(self) -> List[CaptureSource]:
        """Sources this backend can record from, right now."""

    @abc.abstractmethod
    def start(self, config: CaptureConfig) -> None:
        """Begin recording. Returns immediately; recording runs in background."""

    @abc.abstractmethod
    def stop(self) -> CaptureResult:
        """Stop recording and finalise the output file."""

    @property
    @abc.abstractmethod
    def elapsed(self) -> float:
        """Seconds recorded so far."""

    @abc.abstractmethod
    def level(self) -> float:
        """Peak level (0..1) since the previous call, for a meter."""

    def __enter__(self) -> "CaptureBackend":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.stop()
        except Exception:
            pass
