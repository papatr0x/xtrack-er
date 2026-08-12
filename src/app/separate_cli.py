"""Standalone separation tool.

Splits a file into stems and caches them. Nothing else: it does not record and
does not play anything back. Run 'play' afterwards to listen.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress as RichProgress, SpinnerColumn, TextColumn
from rich.table import Table

from src.app import prompts
from src.audio.types import stem_label
from src.library.cache import SongCache, StemManifest, song_key
from src.library.paths import recordings_dir
from src.library.pipeline import SeparationPipeline
from src.library.source import SongSource, from_file

logger = logging.getLogger(__name__)

console = Console()

RECENT_RECORDINGS_SHOWN = 10


class SeparateApp:
    """Separates a file into stems and caches them. Does not record or play."""

    def __init__(self) -> None:
        self.cache = SongCache()
        self.pipeline = SeparationPipeline(self.cache, on_progress=self._show_progress)
        self._spinner: Optional[RichProgress] = None
        self._spinner_task = None

    def run(self, path: Optional[str] = None) -> None:
        """Separate `path` once and return, or loop the interactive file picker."""
        console.print(Panel.fit(
            "[bold blue]xtrack-er — Separate[/bold blue]\n"
            "[dim]Split a file into tracks and cache them. "
            "Run 'play' afterwards to listen.[/dim]",
            border_style="blue",
        ))

        if path:
            self._separate_path(path)
            return

        while True:
            source = self._choose_file()
            if source is None:
                break
            self._separate_source(source)

        console.print("[dim]Bye.[/dim]")

    def _show_progress(self, progress) -> None:
        if progress.stage == "separate":
            if self._spinner is None:
                self._spinner = RichProgress(
                    SpinnerColumn("dots", style="blue"),
                    TextColumn("[dim]{task.description}[/dim]"),
                    TextColumn("[dim]{task.fields[pct]}[/dim]"),
                    console=console,
                    transient=True,
                )
                self._spinner.start()
                self._spinner_task = self._spinner.add_task(progress.message, pct="")
            pct = f"{progress.fraction * 100:.0f}%" if progress.fraction is not None else ""
            self._spinner.update(self._spinner_task, pct=pct)
            return

        self._stop_spinner()
        console.print(f"[dim]{progress.message}[/dim]")

    def _stop_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None
            self._spinner_task = None

    def _choose_file(self) -> Optional[SongSource]:
        recordings = sorted(
            Path(recordings_dir()).glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:RECENT_RECORDINGS_SHOWN]

        options = [(recording.name, str(recording)) for recording in recordings]
        options.append(("Enter a different path...", "__browse__"))

        choice = (
            prompts.select("Which file? (Esc to quit)", options) if recordings else "__browse__"
        )
        if choice is None:
            return None

        if choice == "__browse__":
            raw = prompts.ask_path("Path to an audio file (Tab completes)")
            if not raw:
                return None
            choice = raw

        try:
            return from_file(choice)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            return None

    def _separate_path(self, path: str) -> None:
        try:
            source = from_file(path)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            return
        self._separate_source(source)

    def _separate_source(self, source: SongSource) -> None:
        force = False
        cached = self.cache.get(song_key(source.path))
        if cached is not None:
            if not prompts.confirm(
                f"'{source.title}' is already separated. Re-separate and overwrite?",
                default=False,
            ):
                console.print("[dim]Keeping the existing tracks.[/dim]")
                self._show_manifest(cached)
                return
            force = True

        try:
            manifest = self.pipeline.run(source, force=force)
        except Exception as exc:
            logger.debug("Separation failed", exc_info=True)
            console.print(f"[red]Error:[/red] {exc}")
            return
        finally:
            self._stop_spinner()

        self._show_manifest(manifest)

    def _show_manifest(self, manifest: StemManifest) -> None:
        table = Table(title=f"{len(manifest.stems)} tracks cached", title_style="green")
        table.add_column("Track")
        for stem in sorted(manifest.stems):
            table.add_row(stem_label(stem))
        console.print(table)
        console.print("[dim]Next: uv run python -m src.main play[/dim]")


def main() -> None:
    """Entry point for `python -m src.main separate [file]`."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else None

    if path is None and not sys.stdin.isatty():
        console.print("[red]Provide a file path, or run this interactively.[/red]")
        return

    try:
        SeparateApp().run(path)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Interrupted.[/dim]")


if __name__ == "__main__":
    main()
