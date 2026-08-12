"""Audio device enumeration and selection, shared by playback and capture.

Playback and capture pick devices independently. That separation is not a
nicety: on a machine set up for loopback recording, the system default output is
a Multi-Output Device that feeds the loopback driver, so playing through it
would inject the app's own output back into the recording.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import sounddevice as sd

logger = logging.getLogger(__name__)

# Virtual devices that route audio back into the machine rather than to a
# speaker. Recording wants these; playback must avoid them.
LOOPBACK_HINTS = (
    "blackhole",
    "loopback",
    "soundflower",
    "vb-audio",
    "voicemeeter",
    "stereo mix",
)

# Devices that fan output out to several destinations at once, typically
# including a loopback driver.
AGGREGATE_HINTS = ("multi-output", "multi output", "aggregate", "agregado")


@dataclass(frozen=True)
class DeviceInfo:
    """One PortAudio device, as returned by `output_devices()`/`input_devices()`."""

    index: int
    name: str
    channels: int
    samplerate: int
    is_default: bool = False

    @property
    def is_loopback(self) -> bool:
        return looks_like_loopback(self.name)

    @property
    def is_aggregate(self) -> bool:
        lowered = self.name.lower()
        return any(hint in lowered for hint in AGGREGATE_HINTS)

    @property
    def is_virtual(self) -> bool:
        """True when this is not a real speaker or a real input jack."""
        return self.is_loopback or self.is_aggregate

    def label(self) -> str:
        """Display name with a loopback/multi-output/default tag when relevant."""
        tags = []
        if self.is_default:
            tags.append("system default")
        if self.is_loopback:
            tags.append("loopback")
        elif self.is_aggregate:
            tags.append("multi-output")
        return f"{self.name}" + (f"  ({', '.join(tags)})" if tags else "")


def looks_like_loopback(name: str) -> bool:
    """True if a device name matches a known loopback driver (BlackHole, etc.)."""
    lowered = name.lower()
    return any(hint in lowered for hint in LOOPBACK_HINTS)


def _devices(kind: str) -> List[DeviceInfo]:
    channel_key = "max_output_channels" if kind == "output" else "max_input_channels"
    default_index = sd.default.device[1 if kind == "output" else 0]

    found = []
    for index, device in enumerate(sd.query_devices()):
        channels = device.get(channel_key, 0)
        if channels < 1:
            continue
        found.append(
            DeviceInfo(
                index=index,
                name=device["name"],
                channels=min(2, channels),
                samplerate=int(device.get("default_samplerate") or 44100),
                is_default=(index == default_index),
            )
        )
    return found


def output_devices() -> List[DeviceInfo]:
    """Every playback-capable device, system default flagged."""
    return _devices("output")


def input_devices() -> List[DeviceInfo]:
    """Every capture-capable device, system default flagged."""
    return _devices("input")


def find_by_name(name: str, kind: str = "output") -> Optional[DeviceInfo]:
    """Look up a device by its exact name, or None if it's no longer present."""
    devices = output_devices() if kind == "output" else input_devices()
    for device in devices:
        if device.name == name:
            return device
    return None


def default_output_name() -> str:
    """Name of the device the system is currently playing through."""
    for device in output_devices():
        if device.is_default:
            return device.name
    return ""


def native_samplerate(device_index: int, fallback: int = 44100) -> int:
    """The rate a device actually runs at.

    CoreAudio devices commonly run at 48 kHz. PortAudio does not resample for
    us, so using anything else writes samples under the wrong rate and the
    result plays back at the wrong speed and pitch.
    """
    try:
        rate = sd.query_devices()[device_index].get("default_samplerate")
        return int(rate) if rate else fallback
    except Exception:  # pragma: no cover - depends on the backend
        return fallback


def auto_output_device() -> Optional[DeviceInfo]:
    """Pick a real output, preferring the system default when it is physical.

    Falling back to a physical device matters when the user has configured a
    Multi-Output Device for loopback recording: playing through it would feed
    this app's own output straight back into the capture driver.
    """
    devices = output_devices()
    if not devices:
        return None

    physical = [d for d in devices if not d.is_virtual]

    for device in physical:
        if device.is_default:
            return device

    if physical:
        logger.debug("System default output is virtual; using %s", physical[0].name)
        return physical[0]

    # Everything available is virtual; better to play somewhere than nowhere.
    return devices[0]


def resolve_output_device(preferred_name: Optional[str]) -> Optional[DeviceInfo]:
    """Resolve a saved preference, falling back to automatic selection."""
    if preferred_name:
        device = find_by_name(preferred_name, "output")
        if device is not None:
            return device
        logger.info("Saved output device '%s' is gone; picking one", preferred_name)
    return auto_output_device()
