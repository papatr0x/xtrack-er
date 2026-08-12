"""Time-stretch: pitch must be preserved, and every stem must land on the
same length regardless of its own FFT window size, or they drift out of sync.
"""

import numpy as np
import pytest

from src.audio.stretch import (
    MAX_STEP,
    MIN_STEP,
    clamp_step,
    format_step,
    rate_from_step,
    step_from_rate,
    stretch_buffer,
    stretch_stem_set,
)
from src.audio.types import StemSet

SR = 44100


def sine(freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.column_stack([wave, wave])


def dominant_frequency(signal: np.ndarray, sr: int = SR) -> float:
    mono = signal[:, 0]
    windowed = mono * np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    return float(freqs[np.argmax(spectrum)])


@pytest.mark.parametrize("step", [-40, -20, 20, 40])
def test_stretch_preserves_pitch(step):
    rate = rate_from_step(step)
    original = sine(440, 3.0)
    stretched = stretch_buffer(original, rate)
    # Phase-vocoder stretching resamples the spectrogram's time axis and
    # rescales phase advance per bin -- bin centre frequencies never move.
    assert dominant_frequency(stretched) == pytest.approx(440, abs=3)


@pytest.mark.parametrize("step", [-40, -20, 20, 40])
def test_stretch_changes_length_by_inverse_of_rate(step):
    rate = rate_from_step(step)
    original = sine(220, 2.0)
    stretched = stretch_buffer(original, rate)
    expected_len = round(len(original) / rate)
    assert stretched.shape[0] == pytest.approx(expected_len, abs=2)


def test_stretch_step_zero_is_identity():
    original = sine(220, 1.0)
    result = stretch_buffer(original, rate_from_step(0))
    np.testing.assert_array_equal(result, original)


def test_stretch_stem_set_keeps_all_stems_the_same_length():
    """drums/bass use a shorter FFT window than the rest; lengths must still match."""
    stems = StemSet.create(
        {
            "drums": sine(100, 2.0),
            "bass": sine(60, 2.0),
            "vocals": sine(440, 2.0),
        },
        samplerate=SR,
    )
    stretched = stretch_stem_set(stems, step=-30)
    lengths = {len(buf) for buf in stretched.buffers.values()}
    assert len(lengths) == 1, f"stems desynced: {lengths}"


def test_stretch_stem_set_records_the_rate_and_preserves_original_duration():
    stems = StemSet.create({"a": sine(220, 4.0)}, samplerate=SR)
    stretched = stretch_stem_set(stems, step=-40)  # 60% speed (step is 1% each)
    assert stretched.rate == pytest.approx(0.60)
    # Slower playback -> more frames in the buffer, but the same song length
    # once you account for the rate.
    assert stretched.n_frames > stems.n_frames
    assert stretched.duration == pytest.approx(stems.duration, abs=0.05)


def test_stretch_stem_set_step_zero_returns_the_same_object():
    stems = StemSet.create({"a": sine(220, 1.0)}, samplerate=SR)
    assert stretch_stem_set(stems, step=0) is stems


def test_rate_step_conversions_round_trip():
    for step in (-40, -10, 0, 10, 40):
        assert step_from_rate(rate_from_step(step)) == step


def test_clamp_step_enforces_40_percent_bounds():
    assert clamp_step(MAX_STEP + 100) == MAX_STEP
    assert clamp_step(MIN_STEP - 100) == MIN_STEP
    assert clamp_step(0) == 0


def test_format_step_labels():
    assert format_step(0) == "100.0%"
    assert format_step(-40) == "60.0%"
    assert format_step(40) == "140.0%"
