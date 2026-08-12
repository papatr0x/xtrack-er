"""Persisted user preferences, one file per tool.

record and play never read each other's fields (input_device is record-only;
output_device/master_volume/seek_seconds are play-only), so a shared settings
blob would only create a way for one tool to clobber the other's unrelated
change if both happened to save around the same time. Separate files remove
that coupling for free.

Devices are stored by name rather than index: PortAudio indices shift whenever
something is plugged in or a virtual driver appears.
"""

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Type, TypeVar

from src.library.paths import data_root

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="_PersistedSettings")


class _PersistedSettings:
    """Mixin providing load()/save() against a per-class JSON file."""

    _filename: str

    @classmethod
    def load(cls: Type[T]) -> T:
        """Read the settings file, falling back to defaults if it's missing or broken."""
        path = data_root() / cls._filename
        if not path.exists():
            return cls()

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s); using defaults", cls._filename, exc)
            return cls()

        known = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        return cls(**known)

    def save(self) -> None:
        """Write the current settings to their JSON file."""
        path = data_root() / self._filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


@dataclass
class RecordSettings(_PersistedSettings):
    """Persisted preferences for the `record` tool."""

    _filename = "record_settings.json"

    input_device: Optional[str] = None   # None = ask each time


@dataclass
class PlaySettings(_PersistedSettings):
    """Persisted preferences for the `play` tool."""

    _filename = "play_settings.json"

    output_device: Optional[str] = None  # None = choose automatically
    master_volume: float = 0.8
    seek_seconds: float = 5.0
