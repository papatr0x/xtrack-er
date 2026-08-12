"""KeyReader: raw single-keypress reading, including escape sequences.

Needs a real pty to exercise termios cbreak mode and select()-based reads --
a mocked stdin wouldn't reproduce the actual bug this guards against: mixing
select() on the raw fd with a buffered sys.stdin.read() let one 3-byte arrow
escape sequence come back as three separate keys ('escape', '[', letter)
instead of one. Runs the reader in a real subprocess (not an in-process fork)
so pytest's own output capturing can't interfere with the pty. Skipped where
pty isn't available (Windows).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pty = pytest.importorskip("pty", reason="pty is POSIX-only")

REPO_ROOT = Path(__file__).parent.parent

_CHILD_SCRIPT = """
import json
from src.app.keys import KeyReader

keys = []
with KeyReader() as reader:
    for _ in range({count}):
        keys.append(reader.read(timeout=2.0))
print(json.dumps(keys))
"""


def _read_keys(chunks, count):
    """Run a subprocess that calls KeyReader.read() `count` times over a pty, feeding it `chunks`."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SCRIPT.format(count=count)],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
        text=True,
    )
    os.close(slave)
    try:
        time.sleep(0.3)
        for chunk in chunks:
            os.write(master, chunk)
            time.sleep(0.1)
        out, err = proc.communicate(timeout=5)
    finally:
        os.close(master)

    assert proc.returncode == 0, f"child failed: {err}"
    return json.loads(out)


def test_arrow_keys_are_read_as_one_key_each():
    keys = _read_keys([b"\x1b[C", b"\x1b[D", b"\x1b[A", b"\x1b[B"], count=4)
    assert keys == ["right", "left", "up", "down"]


def test_bare_escape_is_still_escape():
    assert _read_keys([b"\x1b"], count=1) == ["escape"]


def test_plain_characters_pass_through():
    assert _read_keys([b"q"], count=1) == ["q"]


def test_enter_is_normalised_from_carriage_return():
    assert _read_keys([b"\r"], count=1) == ["enter"]
