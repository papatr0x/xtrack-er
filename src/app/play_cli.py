"""Standalone playback tool.

Mixes and plays songs that have already been separated. Nothing else: it does
not record and does not run separation — if a song is not cached yet, it tells
you to run 'separate' first.
"""

import logging
import sys
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

from src.app import prompts, views
from src.app.keys import ESCAPE, LEFT, RIGHT
from src.app.live import run_live_view
from src.app.views import SOLO_KEYS
from src.app.settings import PlaySettings
from src.audio.devices import DeviceInfo, default_output_name, output_devices, resolve_output_device
from src.audio.mixer import StemMixer
from src.audio.stretch import StretchController
from src.audio.transport import Transport
from src.audio.types import stem_label
from src.library.cache import SongCache, StemManifest

logger = logging.getLogger(__name__)

console = Console()

RECENT_SONGS_SHOWN = 20


class PlayApp:
    """Mixes and plays cached stems. Does not record or separate."""

    def __init__(self) -> None:
        self.settings = PlaySettings.load()
        self.cache = SongCache()
        self.output_device: Optional[DeviceInfo] = None

    def run(self) -> None:
        """Loop: pick a cached song and play it, or change settings, until quit."""
        console.print(Panel.fit(
            "[bold blue]xtrack-er — Play[/bold blue]\n"
            "[dim]Mix and play separated tracks. "
            "Run 'separate' first if a song isn't listed.[/dim]",
            border_style="blue",
        ))
        self._resolve_output_device(announce=True)

        while True:
            choice = self._choose_song()
            if choice is None:
                break
            if choice == "settings":
                self._settings_menu()
                continue

            try:
                self._play(choice)
            except Exception as exc:
                logger.debug("Playback failed", exc_info=True)
                console.print(f"[red]Error:[/red] {exc}")

        console.print("[dim]Bye.[/dim]")

    # --- devices ---------------------------------------------------------

    def _resolve_output_device(self, announce: bool = False) -> None:
        """Pick where playback goes, avoiding loopback and multi-output devices.

        Playing through the same Multi-Output Device used for loopback recording
        would feed this app's own output back into the capture driver.
        """
        self.output_device = resolve_output_device(self.settings.output_device)

        if not announce or self.output_device is None:
            return

        system_default = default_output_name()
        if self.settings.output_device is None and self.output_device.name != system_default:
            console.print(
                f"[yellow]Playing through [bold]{self.output_device.name}[/bold][/yellow] "
                f"instead of the system default '{system_default}', "
                "which would loop this app's output back into recordings.\n"
                "[dim]Change it under Settings.[/dim]"
            )

    def _settings_menu(self) -> None:
        while True:
            choice = prompts.select(
                "Settings (Esc to go back)",
                [
                    (f"Playback device  →  {self.output_device.name if self.output_device else '—'}",
                     "output"),
                    (f"Master volume    →  {int(self.settings.master_volume * 100)}%", "volume"),
                    (f"Seek step        →  {self.settings.seek_seconds:g}s", "seek"),
                ],
            )

            if choice is None:
                return
            if choice == "output":
                self._choose_output_device()
            elif choice == "volume":
                self._choose_volume()
            elif choice == "seek":
                self._choose_seek_step()

            self.settings.save()

    def _choose_output_device(self) -> None:
        devices = output_devices()
        options = [("Automatic (avoid loopback devices)", None)]
        for device in devices:
            warn = "  ⚠ feeds back into recordings" if device.is_virtual else ""
            options.append((device.label() + warn, device.name))

        chosen = prompts.select(
            "Where should playback go?", options, default=self.settings.output_device
        )
        if chosen is None and not prompts.confirm("Use automatic selection?", default=True):
            return

        self.settings.output_device = chosen
        self._resolve_output_device()
        if self.output_device is not None:
            console.print(f"[green]Playback: {self.output_device.name}[/green]")

    def _choose_volume(self) -> None:
        chosen = prompts.select(
            "Master volume",
            [(f"{level}%", level / 100) for level in (40, 60, 80, 100, 120)],
            default=self.settings.master_volume,
        )
        if chosen is not None:
            self.settings.master_volume = chosen

    def _choose_seek_step(self) -> None:
        chosen = prompts.select(
            "Seek step (←/→ in the mixer)",
            [(f"{seconds}s", float(seconds)) for seconds in (2, 5, 10, 15, 30)],
            default=self.settings.seek_seconds,
        )
        if chosen is not None:
            self.settings.seek_seconds = chosen

    # --- song selection ----------------------------------------------------

    def _choose_song(self):
        entries = self.cache.entries()
        if not entries:
            console.print("[dim]Nothing separated yet. Run: uv run python -m src.main separate[/dim]")

        options = [
            (f"{m.source_name}   [{len(m.stems)} tracks · {m.created_at.replace('T', ' ')}]", m)
            for m in entries[:RECENT_SONGS_SHOWN]
        ]
        volume = int(self.settings.master_volume * 100)
        options.append((
            f"Settings   [out: {self.output_device.name if self.output_device else '—'} "
            f"· vol {volume}%]",
            "settings",
        ))
        return prompts.select("Which song? (Esc to quit)", options)

    # --- playback --------------------------------------------------------

    def _play(self, manifest: StemManifest) -> None:
        stems = self.cache.load_stem_set(manifest)
        console.print(
            f"[green]{len(stems.stems)} tracks ready[/green]: "
            + ", ".join(stem_label(s) for s in stems.stems)
        )

        mixer = StemMixer(stems)
        mixer.set_master_gain(self.settings.master_volume)

        device_index = self.output_device.index if self.output_device else None
        transport = Transport(mixer, device=device_index)
        stretch = StretchController(stems, on_ready=transport.swap_stems)

        try:
            self._mixer_view(manifest.source_name, mixer, transport, stretch)
        finally:
            stretch.close()
            transport.close()

    def _mixer_view(self, title, mixer, transport, stretch) -> None:
        stem_ids: List[str] = list(mixer.stem_ids)
        subtitle = self.output_device.name if self.output_device else ""
        started = False

        def render():
            # Deferred until the first real render, so playback never starts
            # when stdin turns out not to be a TTY (run_live_view bails first).
            nonlocal started
            if not started:
                transport.play()
                started = True

            snapshot = mixer.snapshot()
            return views.render_mixer(
                title=title,
                stems=stem_ids,
                muted=[snapshot[s].muted for s in stem_ids],
                solo=[snapshot[s].solo for s in stem_ids],
                state=transport.state,
                position=transport.position,
                duration=transport.duration,
                applied_step=stretch.applied_step,
                target_step=stretch.target_step,
                rendering=stretch.is_rendering,
                peak=mixer.last_peak,
                device=subtitle,
            )

        def on_key(key: str) -> bool:
            if key == ESCAPE:
                return True
            self._handle_mixer_key(key, stem_ids, mixer, transport, stretch)
            return False

        ran = run_live_view(console, render, on_key)
        if not ran:
            console.print("[red]Playback needs an interactive terminal.[/red]")
            return

        transport.pause()

    def _handle_mixer_key(self, key, stem_ids, mixer, transport, stretch) -> None:
        if key == " ":
            transport.toggle()

        elif key.isdigit() and key != "0":
            index = int(key) - 1
            if 0 <= index < len(stem_ids):
                mixer.toggle(stem_ids[index])

        elif key in SOLO_KEYS:
            index = SOLO_KEYS.index(key)
            if index < len(stem_ids):
                mixer.toggle_solo(stem_ids[index])

        elif key == "a":
            mixer.set_all(False)
        elif key == "n":
            mixer.set_all(True)

        elif key in ("-", "_"):
            stretch.nudge(-1)
        elif key in ("=", "+"):
            stretch.nudge(1)
        elif key in (",", "<"):
            stretch.nudge(-5)
        elif key in (".", ">"):
            stretch.nudge(5)
        elif key == "0":
            stretch.reset()

        elif key == LEFT:
            transport.seek_relative(-self.settings.seek_seconds)
        elif key == RIGHT:
            transport.seek_relative(self.settings.seek_seconds)


def main() -> None:
    """Entry point for `python -m src.main play`."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if not sys.stdin.isatty():
        console.print("[red]Playback needs an interactive terminal.[/red]")
        return

    try:
        PlayApp().run()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Interrupted.[/dim]")


if __name__ == "__main__":
    main()
