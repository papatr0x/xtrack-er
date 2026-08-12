"""Platform dispatch for capture backends.

The UI never names a backend: it asks for sources and gets routed. Adding
ScreenCaptureKit (true per-application capture on macOS) or a Windows backend
means registering it here and nothing else changes.
"""

import logging
import sys
from typing import List, Optional

from src.capture.base import CaptureBackend, CaptureSource, SourceKind
from src.capture.portaudio import (
    BLACKHOLE_REMEDY,
    PortAudioRecorder,
    default_output_name,
    routing_warning,
)

logger = logging.getLogger(__name__)


def available_backends() -> List[CaptureBackend]:
    """Instantiated backends that can run here, most capable first."""
    backends: List[CaptureBackend] = []
    for backend_cls in _candidate_classes():
        try:
            if backend_cls.is_available():
                backends.append(backend_cls())
        except Exception:
            logger.debug("Backend %s unavailable", backend_cls.__name__, exc_info=True)
            continue
    return backends


def _candidate_classes():
    if sys.platform == "darwin":
        # ScreenCaptureKit (per-application) slots in ahead of the device
        # backend once its helper binary exists.
        return [PortAudioRecorder]
    if sys.platform == "win32":
        # WASAPI loopback via the same PortAudio dependency.
        return [PortAudioRecorder]
    return [PortAudioRecorder]


def list_sources() -> List[CaptureSource]:
    """Every source across every available backend."""
    sources: List[CaptureSource] = []
    for backend in available_backends():
        try:
            sources.extend(backend.list_sources())
        except Exception:
            logger.debug("Backend %s failed to list sources", backend.name, exc_info=True)
            continue
    return sources


def backend_for(source: CaptureSource) -> Optional[CaptureBackend]:
    """The available backend instance that produced `source`, if it's still there."""
    for backend in available_backends():
        if backend.name == source.backend:
            return backend
    return None


def source_warning(source: CaptureSource) -> Optional[str]:
    """Warning to show before recording from `source`, if anything is wrong."""
    return routing_warning(source.name)


def current_output_name() -> str:
    """Name of the device the system is currently playing through."""
    return default_output_name()


def system_audio_remedy() -> Optional[str]:
    """Instructions to enable system-audio capture, or None if already possible."""
    if any(s.kind is SourceKind.SYSTEM for s in list_sources()):
        return None
    if sys.platform == "darwin":
        return BLACKHOLE_REMEDY
    return (
        "No loopback input was found. Enable 'Stereo Mix' in your sound "
        "settings, or install a virtual audio cable."
    )
