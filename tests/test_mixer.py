"""StemMixer: per-stem gain/mute, and the declick ramp that makes it usable."""

import numpy as np
import pytest

from src.audio.mixer import StemMixer
from src.audio.types import StemSet

SR = 44100


def make_stems(**freqs: float) -> StemSet:
    """A 2s StemSet with one stem per keyword, each a distinct sine tone."""
    t = np.arange(SR * 2) / SR
    buffers = {
        name: (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[:, None] * np.ones((1, 2), np.float32)
        for name, freq in freqs.items()
    }
    return StemSet.create(buffers, samplerate=SR)


def render_past_ramp(mixer: StemMixer, start: int, n: int) -> np.ndarray:
    """Advance past the mixer's declick ramp, then return one clean block."""
    mixer.render(start, mixer._ramp_len + 64)
    return mixer.render(start + mixer._ramp_len + 64, n)


def test_render_sums_unmuted_stems():
    stems = make_stems(drums=100, bass=200)
    mixer = StemMixer(stems)
    mixed = render_past_ramp(mixer, 0, 1024)
    # The very first render() ramps up from silence, so compare against a
    # later, already-settled window rather than frame 0.
    start = mixer._ramp_len + 64
    expected = stems.buffers["drums"][start:start + 1024] + stems.buffers["bass"][start:start + 1024]
    np.testing.assert_allclose(mixed, expected, atol=1e-5)


def test_muted_stem_contributes_exactly_zero_after_ramp_settles():
    stems = make_stems(drums=100, bass=200, vocals=400)
    mixer = StemMixer(stems)
    mixer.set_muted("vocals", True)
    mixer.set_muted("bass", True)

    start = 10_000
    mixed = render_past_ramp(mixer, start, 2048)
    block_start = start + mixer._ramp_len + 64
    expected_drums_only = stems.buffers["drums"][block_start:block_start + 2048]
    np.testing.assert_allclose(mixed, expected_drums_only, atol=1e-6)


def test_toggle_flips_and_reports_new_audible_state():
    stems = make_stems(drums=100)
    mixer = StemMixer(stems)
    assert mixer.toggle("drums") is False  # now muted -> not audible
    assert mixer.is_muted("drums") is True
    assert mixer.toggle("drums") is True  # now unmuted -> audible
    assert mixer.is_muted("drums") is False


def test_toggle_unknown_stem_is_a_no_op():
    stems = make_stems(drums=100)
    mixer = StemMixer(stems)
    assert mixer.toggle("nonexistent") is False


def test_master_gain_scales_output():
    stems = make_stems(drums=100)
    mixer = StemMixer(stems)
    mixer.set_master_gain(0.5)
    mixed = render_past_ramp(mixer, 0, 1024)
    start = mixer._ramp_len + 64
    expected = stems.buffers["drums"][start:start + 1024] * 0.5
    np.testing.assert_allclose(mixed, expected, atol=1e-5)


def test_swap_stems_preserves_existing_mute_state():
    stems = make_stems(drums=100, bass=200)
    mixer = StemMixer(stems)
    mixer.set_muted("drums", True)

    other_stems = make_stems(drums=100, bass=200, vocals=440)
    mixer.swap_stems(other_stems)

    snapshot = mixer.snapshot()
    assert snapshot["drums"].muted is True  # preserved
    assert snapshot["bass"].muted is False
    assert snapshot["vocals"].muted is False  # new stem defaults to audible


def test_toggle_solo_flips_and_returns_new_state():
    stems = make_stems(drums=100)
    mixer = StemMixer(stems)
    assert mixer.toggle_solo("drums") is True
    assert mixer.is_solo("drums") is True
    assert mixer.toggle_solo("drums") is False
    assert mixer.is_solo("drums") is False


def test_solo_silences_every_other_stem():
    stems = make_stems(drums=100, bass=200, vocals=400)
    mixer = StemMixer(stems)
    mixer.set_solo("bass", True)

    start = 10_000
    mixed = render_past_ramp(mixer, start, 2048)
    block_start = start + mixer._ramp_len + 64
    expected_bass_only = stems.buffers["bass"][block_start:block_start + 2048]
    np.testing.assert_allclose(mixed, expected_bass_only, atol=1e-6)


def test_solo_is_additive_across_multiple_stems():
    stems = make_stems(drums=100, bass=200, vocals=400)
    mixer = StemMixer(stems)
    mixer.set_solo("bass", True)
    mixer.set_solo("drums", True)
    assert set(mixer.active_stems()) == {"drums", "bass"}


def test_mute_overrides_solo_on_the_same_stem():
    """Soloing a muted stem does not un-mute it -- both flags stay independent."""
    stems = make_stems(drums=100, bass=200)
    mixer = StemMixer(stems)
    mixer.set_muted("drums", True)
    mixer.set_solo("drums", True)
    assert mixer.active_stems() == []


def test_has_solo_reflects_any_soloed_stem():
    stems = make_stems(drums=100, bass=200)
    mixer = StemMixer(stems)
    assert mixer.has_solo is False
    mixer.set_solo("drums", True)
    assert mixer.has_solo is True


def test_clear_solo_restores_normal_playback():
    stems = make_stems(drums=100, bass=200)
    mixer = StemMixer(stems)
    mixer.set_solo("drums", True)
    mixer.clear_solo()
    assert mixer.has_solo is False
    assert set(mixer.active_stems()) == {"drums", "bass"}


def test_set_all_unmuted_also_clears_solo():
    stems = make_stems(drums=100, bass=200)
    mixer = StemMixer(stems)
    mixer.set_solo("drums", True)
    mixer.set_all(False)  # "a" -- everything should actually play
    assert mixer.has_solo is False
    assert set(mixer.active_stems()) == {"drums", "bass"}


def test_swap_stems_preserves_existing_solo_state():
    stems = make_stems(drums=100, bass=200)
    mixer = StemMixer(stems)
    mixer.set_solo("drums", True)

    other_stems = make_stems(drums=100, bass=200, vocals=440)
    mixer.swap_stems(other_stems)

    snapshot = mixer.snapshot()
    assert snapshot["drums"].solo is True  # preserved
    assert snapshot["vocals"].solo is False  # new stem defaults to not soloed


def test_render_past_end_of_material_is_silent_and_flagged_via_length():
    stems = make_stems(drums=100)
    mixer = StemMixer(stems)
    tail = mixer.render(stems.n_frames - 10, 100)
    assert tail.shape == (100, 2)
    # Only the first 10 frames have material; the rest is silence.
    np.testing.assert_array_equal(tail[10:], np.zeros((90, 2), dtype=np.float32))
