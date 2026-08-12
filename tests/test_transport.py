"""Transport: the shared clock driving playback.

These open a real sounddevice OutputStream -- there is no injectable device
abstraction (a real gap, noted in the architecture review) -- so the whole
module is skipped when no audio backend is available, e.g. in a headless CI
container. Where this can run, it is the same style of check done by hand
throughout this project's early development, turned into a permanent test.
"""

import time

import numpy as np
import pytest

from src.audio.mixer import StemMixer
from src.audio.types import StemSet

sd = pytest.importorskip("sounddevice")


def _device_available() -> bool:
    try:
        devices = sd.query_devices()
        return any(d.get("max_output_channels", 0) > 0 for d in devices)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _device_available(), reason="no audio output device available"
)

SR = 44100


@pytest.fixture
def transport_module():
    # Imported lazily so a missing sounddevice backend fails the skip check
    # above rather than an import error at collection time.
    from src.audio import transport

    return transport


def make_stems(seconds: float = 5.0) -> StemSet:
    t = np.arange(int(SR * seconds)) / SR
    wave = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return StemSet.create({"tone": np.column_stack([wave, wave])}, samplerate=SR)


@pytest.fixture
def transport(transport_module):
    mixer = StemMixer(make_stems())
    mixer.set_master_gain(0.1)  # keep it quiet during tests
    t = transport_module.Transport(mixer)
    yield t
    t.close()


def test_position_advances_while_playing(transport):
    assert transport.position == pytest.approx(0.0, abs=0.05)
    transport.play()
    time.sleep(1.0)
    assert transport.position > 0.3


def test_position_holds_while_paused(transport):
    transport.play()
    time.sleep(0.6)
    transport.pause()
    held = transport.position
    time.sleep(0.3)
    assert transport.position == pytest.approx(held, abs=0.05)


def test_seek_lands_near_target(transport):
    transport.seek(2.0)
    transport.play()
    time.sleep(0.3)
    assert transport.position == pytest.approx(2.0, abs=0.5)


def test_stop_resets_to_zero(transport):
    transport.play()
    time.sleep(0.3)
    transport.stop()
    assert transport.position == 0.0
    assert transport.state.value == "stopped"


def test_swap_stems_preserves_position(transport):
    transport.play()
    time.sleep(0.5)
    before = transport.position

    stems = make_stems()
    transport.swap_stems(stems)

    assert transport.position == pytest.approx(before, abs=0.1)
