"""Rendering for the live views. Presentation only -- no audio logic here."""

from typing import List, Optional

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.audio.stretch import format_step
from src.audio.transport import TransportState
from src.audio.types import stem_label
from src.utils.audio import format_time

BAR_WIDTH = 32

# Positionally matches the digit row used for mute (1-9): q is above 1, w
# above 2, and so on, so the two hands of keys line up on the keyboard.
SOLO_KEYS = "qwertyuio"

STATE_LABELS = {
    TransportState.PLAYING: "[green]playing[/green]",
    TransportState.PAUSED: "[yellow]paused[/yellow]",
    TransportState.STOPPED: "[dim]stopped[/dim]",
    TransportState.FINISHED: "[dim]finished[/dim]",
}

MIXER_HELP = (
    "[bold]space[/bold] play/pause   "
    "[bold]1-9[/bold] mute   "
    "[bold]q w e...[/bold] solo   "
    "[bold]a[/bold]/[bold]n[/bold] all/none\n"
    "[bold]-[/bold]/[bold]+[/bold] speed 1%   "
    "[bold],[/bold]/[bold].[/bold] speed 5%   "
    "[bold]0[/bold] reset speed   "
    "[bold]←[/bold]/[bold]→[/bold] seek 5s   "
    "[bold]esc[/bold] back to menu"
)


def progress_bar(position: float, duration: float, width: int = BAR_WIDTH) -> str:
    """Rich markup for a filled/unfilled playback progress bar."""
    if duration <= 0:
        return "─" * width
    filled = int(width * min(1.0, position / duration))
    return "[green]" + "━" * filled + "[/green][dim]" + "─" * (width - filled) + "[/dim]"


def level_bar(level: float, width: int = 24) -> str:
    """Rich markup for a VU-style input level bar, red once it's clipping."""
    filled = int(width * min(1.0, level))
    colour = "red" if level > 0.98 else "green"
    return f"[{colour}]" + "█" * filled + f"[/{colour}][dim]" + "░" * (width - filled) + "[/dim]"


def render_mixer(
    title: str,
    stems: List[str],
    muted: List[bool],
    solo: List[bool],
    state: TransportState,
    position: float,
    duration: float,
    applied_step: int,
    target_step: int,
    rendering: bool,
    peak: float,
    device: str = "",
) -> Panel:
    """Render the live mixer panel: per-stem mute/solo table, transport and speed."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", width=3)
    table.add_column(width=16)
    table.add_column(width=10)
    table.add_column(width=10)

    for index, stem in enumerate(stems, 1):
        is_muted = muted[index - 1]
        is_solo = solo[index - 1]
        marker = "[dim]○ off[/dim]" if is_muted else "[green]● on[/green]"
        name = f"[dim]{stem_label(stem)}[/dim]" if is_muted else stem_label(stem)
        solo_key = SOLO_KEYS[index - 1] if index - 1 < len(SOLO_KEYS) else ""
        solo_marker = f"[yellow]{solo_key} solo[/yellow]" if is_solo else f"[dim]{solo_key}[/dim]"
        table.add_row(f"[bold]{index}[/bold]", name, marker, solo_marker)

    speed = format_step(applied_step)
    if rendering or target_step != applied_step:
        speed += f" [yellow]→ {format_step(target_step)}...[/yellow]"

    clip = "  [red]CLIP[/red]" if peak > 1.0 else ""

    header = Text.from_markup(
        f"[bold]{title}[/bold]\n"
        f"{STATE_LABELS[state]}   "
        f"{format_time(position)} / {format_time(duration)}   "
        f"speed [bold]{speed}[/bold]{clip}"
    )
    bar = Text.from_markup(progress_bar(position, duration))

    body = Table.grid(padding=(0, 0))
    body.add_row(header)
    body.add_row(bar)
    body.add_row("")
    body.add_row(table)
    body.add_row("")
    body.add_row(Text.from_markup(MIXER_HELP))

    return Panel(
        body,
        border_style="blue",
        title="Mixer",
        title_align="left",
        subtitle=f"[dim]♪ {device}[/dim]" if device else None,
        subtitle_align="right",
    )


def render_recorder(
    source_name: str,
    recording: bool,
    elapsed: float,
    level: float,
    peak: float,
    output_path: Optional[str],
) -> Panel:
    """Render the live recorder panel: status, elapsed time and input level."""
    if recording:
        status = "[red]● REC[/red]"
        hint = "[bold]space[/bold] or [bold]s[/bold] to stop"
    else:
        status = "[dim]ready[/dim]"
        hint = "[bold]space[/bold] to start   [bold]q[/bold] to cancel"

    lines = Table.grid(padding=(0, 0))
    lines.add_row(Text.from_markup(f"Source: [bold]{source_name}[/bold]"))
    lines.add_row(Text.from_markup(f"{status}   {format_time(elapsed)}"))
    lines.add_row("")
    lines.add_row(Text.from_markup(f"level {level_bar(level)}"))
    if peak > 0.99:
        lines.add_row(Text.from_markup("[red]Input is clipping — lower the source volume[/red]"))
    lines.add_row("")
    lines.add_row(Text.from_markup(hint))
    if output_path:
        lines.add_row(Text.from_markup(f"[dim]{output_path}[/dim]"))

    return Panel(Align.left(lines), border_style="red" if recording else "blue",
                 title="Record", title_align="left")
