"""StemSet: the one place sample alignment and stereo handling are enforced."""

import numpy as np
import pytest

from src.audio.types import StemSet, sort_stems, stem_label


def sine(freq: float, seconds: float, sr: int = 44100, channels: int = 2) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 1:
        return wave[:, None]
    return np.column_stack([wave, wave])


def test_create_trims_to_shortest_length():
    sr = 44100
    stems = StemSet.create(
        {"a": sine(220, 1.0, sr), "b": sine(440, 1.0, sr)[:-100]},
        samplerate=sr,
    )
    assert stems.n_frames == sr - 100
    assert all(len(buf) == stems.n_frames for buf in stems.buffers.values())


def test_create_rejects_large_drift():
    sr = 44100
    with pytest.raises(ValueError, match="misaligned"):
        StemSet.create(
            {"a": sine(220, 2.0, sr), "b": sine(440, 1.0, sr)},
            samplerate=sr,
        )


def test_create_upmixes_mono_without_averaging():
    sr = 44100
    mono = sine(220, 0.5, sr, channels=1)
    stems = StemSet.create({"solo": mono}, samplerate=sr)
    buf = stems.buffers["solo"]
    assert buf.shape[1] == 2
    # Upmixed by duplication, not by averaging away information.
    np.testing.assert_array_equal(buf[:, 0], buf[:, 1])
    np.testing.assert_allclose(buf[:, 0], mono[:, 0])


def test_create_rejects_empty_input():
    with pytest.raises(ValueError):
        StemSet.create({}, samplerate=44100)


def test_frame_second_roundtrip_at_normal_rate():
    sr = 44100
    stems = StemSet.create({"a": sine(220, 3.0, sr)}, samplerate=sr, rate=1.0)
    frame = stems.seconds_to_frames(1.5)
    assert stems.frames_to_seconds(frame) == pytest.approx(1.5, abs=1e-3)
    assert stems.duration == pytest.approx(3.0, abs=1e-3)


def test_frame_second_roundtrip_at_stretched_rate():
    """Duration must stay in ORIGINAL song seconds even when rate != 1.0."""
    sr = 44100
    # 3s of original audio rendered at 75% speed occupies 4s of buffer.
    stretched_frames = int(3.0 / 0.75 * sr)
    stems = StemSet.create(
        {"a": sine(220, stretched_frames / sr, sr)}, samplerate=sr, rate=0.75
    )
    assert stems.duration == pytest.approx(3.0, abs=1e-2)
    frame = stems.seconds_to_frames(1.5)
    assert stems.frames_to_seconds(frame) == pytest.approx(1.5, abs=1e-2)


def test_stem_label_known_and_fallback():
    assert stem_label("backing_vocals") == "Backing vocals"
    assert stem_label("some_new_stem") == "Some new stem"


def test_sort_stems_canonical_order_first():
    stems = {"other", "vocals", "zzz_custom", "drums"}
    assert sort_stems(stems) == ["vocals", "drums", "other", "zzz_custom"]
