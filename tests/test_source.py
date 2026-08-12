"""SongSource validation.

is_supported() must not gate on libsndfile alone: separation actually reads
files through demucs's own loader (sphn, then ffmpeg), which can decode some
containers libsndfile never touches (m4a in particular). These tests check the
gate's *permissiveness*, not real decoding -- that only happens once a file
reaches the separation pipeline.
"""

import pytest

from src.library.source import from_file


def test_from_file_accepts_a_real_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "song.wav"
    sf.write(path, np.zeros((100, 2), dtype="float32"), 44100)

    source = from_file(str(path))
    assert source.path == str(path)
    assert source.title == "song"


def test_from_file_accepts_m4a_on_extension_alone(tmp_path):
    """libsndfile can't read m4a, but demucs's own loader (sphn/ffmpeg) might.

    Content doesn't matter here -- from_file() only gates on extension; a real
    decode attempt happens later, in the separation pipeline.
    """
    path = tmp_path / "song.m4a"
    path.write_bytes(b"not a real decoder will ever see this in this test")

    source = from_file(str(path))
    assert source.title == "song"


def test_from_file_rejects_an_obviously_wrong_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")

    with pytest.raises(ValueError, match="Unsupported format"):
        from_file(str(path))


def test_from_file_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_file(str(tmp_path / "does_not_exist.wav"))
