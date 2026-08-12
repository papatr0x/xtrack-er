"""Shared skeleton for a Rich Live view driven by single keypresses.

record_cli and play_cli each poll a key, re-render, and act on it in a loop;
this is that loop, factored out once so a fix to the polling/refresh behaviour
lands in one place instead of two near-identical copies.
"""

from typing import Callable

from rich.console import Console
from rich.live import Live

from src.app.keys import KeyReader

REFRESH_HZ = 12


def run_live_view(
    console: Console,
    render: Callable[[], object],
    on_key: Callable[[str], bool],
    refresh_hz: int = REFRESH_HZ,
) -> bool:
    """Render and poll keys until `on_key` returns True.

    `render` is called with no arguments and must return a Rich renderable.
    `on_key` receives each keypress and returns True to stop the loop.
    If `on_key` raises, the Live view is torn down (transient, so its content
    is erased) before the exception propagates to the caller.

    Returns False without entering the loop if stdin is not a TTY, so callers
    can tell "not interactive" apart from "user pressed q".
    """
    with KeyReader() as keys:
        if not keys.interactive:
            return False

        with Live(console=console, refresh_per_second=refresh_hz, transient=True) as live:
            while True:
                live.update(render())
                key = keys.read(timeout=1.0 / refresh_hz)
                if key is None:
                    continue
                if on_key(key):
                    break

    return True
