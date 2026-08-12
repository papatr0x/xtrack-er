"""Standalone recording tool.

Captures audio from an input to a WAV file. Nothing else: it does not separate
the recording and does not play anything back. Run 'separate' on the resulting
file when you're ready.
"""

import logging
import sys
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from src.app import prompts, views
from src.app.live import run_live_view
from src.app.settings import RecordSettings
from src.audio.devices import default_output_name, input_devices
from src.capture.base import BackendUnavailable, CaptureConfig, CaptureResult, KIND_LABELS
from src.capture.portaudio import SILENT_INPUT_REMEDY
from src.capture.registry import backend_for, list_sources, source_warning, system_audio_remedy
from src.library.paths import recordings_dir

logger = logging.getLogger(__name__)

console = Console()


class RecordApp:
    """Captures audio from an input to a file. Does not separate or play."""

    def __init__(self) -> None:
        self.settings = RecordSettings.load()

    def run(self) -> None:
        """Loop: record a file or change settings, until the user quits."""
        console.print(Panel.fit(
            "[bold blue]xtrack-er — Record[/bold blue]\n"
            "[dim]Capture audio to a file. Run 'separate' on it afterwards "
            "to split it into tracks.[/dim]",
            border_style="blue",
        ))

        while True:
            choice = prompts.select(
                "What next? (Esc to quit)",
                [
                    ("Start a new recording", "record"),
                    (f"Settings   [in: {self.settings.input_device or 'ask each time'}]",
                     "settings"),
                ],
            )

            if choice is None:
                break
            if choice == "settings":
                self._choose_input_device()
                self.settings.save()
                continue

            path = self._record_one()
            if path:
                console.print(f"[green]Saved:[/green] {path}")
                console.print(f'[dim]Next: uv run python -m src.main separate "{path}"[/dim]')

        console.print("[dim]Bye.[/dim]")

    def _choose_input_device(self) -> None:
        options = [("Ask each time", None)]
        for device in input_devices():
            tag = "  (system audio)" if device.is_loopback else ""
            options.append((device.name + tag, device.name))

        chosen = prompts.select(
            "Default input device", options, default=self.settings.input_device
        )
        self.settings.input_device = chosen
        console.print(f"[green]Input: {chosen or 'ask each time'}[/green]")

    def _record_one(self) -> Optional[str]:
        sources = list_sources()
        if not sources:
            console.print("[red]No audio inputs found.[/red]")
            return None

        capture_source = None
        if self.settings.input_device:
            capture_source = next(
                (s for s in sources if s.name == self.settings.input_device), None
            )

        if capture_source is None:
            remedy = system_audio_remedy()
            if remedy:
                console.print(Panel(remedy, title="To record system audio",
                                    border_style="yellow"))

            options = [
                (f"{s.name}   [{KIND_LABELS[s.kind]}]" + (f" — {s.detail}" if s.detail else ""), s)
                for s in sources
            ]
            capture_source = prompts.select("Record from which input? (Esc to cancel)", options)
            if capture_source is None:
                return None

        backend = backend_for(capture_source)
        if backend is None:
            console.print("[red]That input is no longer available.[/red]")
            return None

        # A loopback device with nothing routed into it records pure silence,
        # which is otherwise only discoverable by recording and finding it empty.
        warning = source_warning(capture_source)
        if warning:
            console.print(Panel(
                warning,
                title=f"System output is currently '{default_output_name()}'",
                border_style="yellow",
            ))
            if not prompts.confirm("Record anyway?", default=False):
                return None

        title = prompts.ask_text(
            "Name for this recording", default=time.strftime("recording-%Y%m%d-%H%M%S")
        )
        if not title:
            return None
        output_path = str(recordings_dir() / f"{title}.wav")

        result = self._run_recorder(backend, capture_source, output_path)
        if result is None:
            return None

        # macOS reports a denied microphone permission as an endless stream of
        # zeros rather than an error, so exact silence is worth calling out.
        if not getattr(backend, "saw_signal", True):
            console.print(Panel(SILENT_INPUT_REMEDY, title="Recorded nothing",
                                border_style="red"))
            return None

        if result.duration < 1.0:
            console.print("[red]Recording too short to keep.[/red]")
            return None

        console.print(
            f"[green]Recorded {result.duration:.1f}s[/green] at {result.samplerate} Hz "
            f"(peak {result.peak:.2f})"
        )
        return output_path

    def _run_recorder(self, backend, capture_source, output_path) -> Optional[CaptureResult]:
        """Live start/stop recording view. Returns a CaptureResult or None."""
        config = CaptureConfig(source=capture_source, output_path=output_path)
        recording = False
        result: Optional[CaptureResult] = None
        peak = 0.0

        def render():
            nonlocal peak
            level = backend.level() if recording else 0.0
            peak = max(peak, level)
            return views.render_recorder(
                capture_source.name, recording,
                backend.elapsed if recording else 0.0,
                level, peak, output_path if recording else None,
            )

        def on_key(key: str) -> bool:
            nonlocal recording, result
            if key == " " and not recording:
                backend.start(config)  # BackendUnavailable propagates to the caller
                recording = True
                return False
            if key in (" ", "s") and recording:
                result = backend.stop()
                return True
            if key == "q" and not recording:
                return True
            return False

        try:
            ran = run_live_view(console, render, on_key)
        except BackendUnavailable as exc:
            console.print(f"[red]{exc}[/red]")
            if exc.remedy:
                console.print(Panel(exc.remedy, border_style="yellow"))
            return None

        if not ran:
            console.print("[red]Recording needs an interactive terminal.[/red]")
            return None

        return result


def main() -> None:
    """Entry point for `python -m src.main record`."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if not sys.stdin.isatty():
        console.print("[red]Recording needs an interactive terminal.[/red]")
        return

    try:
        RecordApp().run()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Interrupted.[/dim]")


if __name__ == "__main__":
    main()
