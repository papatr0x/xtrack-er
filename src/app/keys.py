"""Single-keypress reader for the live views.

A live mixer needs keys to act immediately, without waiting for Enter, so the
terminal is put in cbreak mode for the duration of a view.
"""

import os
import select
import sys
import termios
import tty
from typing import Optional

# Names yielded for keys that are not a single printable character.
UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
ENTER = "enter"
ESCAPE = "escape"

_ARROWS = {"A": UP, "B": DOWN, "C": RIGHT, "D": LEFT}


class KeyReader:
    """Context manager putting stdin in cbreak mode.

    Falls back to a no-op reader when stdin is not a TTY (pipes, CI), so the
    app can still be driven non-interactively.
    """

    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._saved = None
        self.interactive = sys.stdin.isatty()

    def __enter__(self) -> "KeyReader":
        if self.interactive:
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self.interactive and self._fd is not None and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def _read_byte(self, timeout: float) -> Optional[str]:
        """Read exactly one byte straight from the fd, or None on timeout.

        Deliberately bypasses sys.stdin's buffered TextIOWrapper: it can pull
        an entire escape sequence into its own internal buffer on the first
        read(1), leaving nothing for select() to see on the next call and
        splitting one arrow key into three "keys" (escape, '[', letter).
        Reading the raw fd keeps what select() sees and what gets consumed in
        sync, one byte at a time.
        """
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        return data.decode(errors="ignore") if data else None

    def read(self, timeout: float = 0.1) -> Optional[str]:
        """Return one key, or None if nothing arrived within `timeout`."""
        if not self.interactive:
            return None

        char = self._read_byte(timeout)
        if not char:
            return None

        if char == "\x1b":
            # Escape, or an escape sequence such as an arrow key.
            second = self._read_byte(0.05)
            if second != "[":
                return ESCAPE
            return _ARROWS.get(self._read_byte(0.05), ESCAPE)

        if char in ("\r", "\n"):
            return ENTER
        if char == "\x03":
            raise KeyboardInterrupt

        return char
