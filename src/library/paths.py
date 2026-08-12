"""Application directories, resolved at runtime.

Never hard-code a user's home path: these are derived from the platform so the
repository stays free of machine-specific paths.
"""

import os
import sys
from pathlib import Path

APP_NAME = "xtrack-er"


def data_root() -> Path:
    """Base directory for everything this app persists."""
    override = os.environ.get("XTRACK_ER_HOME")
    if override:
        return Path(override).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def cache_dir() -> Path:
    """Where separated stems are stored, creating it if needed."""
    path = data_root() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def recordings_dir() -> Path:
    """Where captured audio is stored, creating it if needed."""
    path = data_root() / "recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path
