"""A song source: an audio file on disk.

A recording produced by 'record' is just a WAV file handed to 'separate' like
any other — there is no in-process channel between the tools, so provenance
past the file itself is not tracked here.
"""

import os
from dataclasses import dataclass

from src.audio.io import EXTRA_CONTAINER_EXTENSIONS, is_supported, supported_extensions


@dataclass(frozen=True)
class SongSource:
    """An audio file ready to feed into separation -- from a file or a recording."""

    path: str
    title: str

    @property
    def exists(self) -> bool:
        """True if the underlying file is still on disk."""
        return os.path.isfile(self.path)


def from_file(path: str) -> SongSource:
    """Build a source from a user-supplied path, validating the format."""
    path = os.path.expanduser(path.strip().strip("'\""))

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    if not is_supported(path):
        maybe = ", ".join(sorted(EXTRA_CONTAINER_EXTENSIONS))
        raise ValueError(
            f"Unsupported format '{os.path.splitext(path)[1]}'. "
            f"Supported: {', '.join(supported_extensions())}. "
            f"({maybe} may also work if ffmpeg is installed.)"
        )

    return SongSource(path=path, title=os.path.splitext(os.path.basename(path))[0])
