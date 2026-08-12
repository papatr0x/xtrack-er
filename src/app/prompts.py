"""Interactive prompts built on questionary.

Kept separate from cli.py so the flow reads as flow, and so swapping the prompt
toolkit later touches one file.
"""

from typing import List, Optional, Sequence, Tuple

import questionary
from prompt_toolkit.keys import Keys
from questionary import Choice

# Muted grey for the pointer and highlighted row; everything else is default so
# the prompts inherit the user's terminal colours.
STYLE = questionary.Style([
    ("qmark", "fg:#5f87ff bold"),
    ("question", "bold"),
    ("pointer", "fg:#5f87ff bold"),
    ("highlighted", "fg:#5f87ff bold"),
    ("selected", "fg:#00af5f"),
    ("answer", "fg:#00af5f"),
    ("instruction", "fg:#808080"),
])

CANCEL = "__cancel__"


def select(message: str, options: Sequence[Tuple[str, object]],
           default: Optional[object] = None) -> Optional[object]:
    """Arrow-key menu. Returns the chosen value, or None if cancelled."""
    choices = [Choice(title=title, value=value) for title, value in options]
    default_choice = next((c for c in choices if c.value == default), None)

    question = questionary.select(
        message,
        choices=choices,
        style=STYLE,
        default=default_choice,
        use_shortcuts=False,
        instruction="(↑↓ to move, Enter to choose, Esc to go back)",
    )
    # questionary only binds Ctrl+C/Ctrl+Q to cancel a select prompt; a
    # catch-all binding absorbs Escape otherwise, so it has to be added here.
    question.application.key_bindings.add(Keys.Escape, eager=True)(
        lambda event: event.app.exit(result=None)
    )
    answer = question.ask()

    return None if answer is None or answer == CANCEL else answer


def ask_path(message: str) -> Optional[str]:
    """File path with tab completion."""
    answer = questionary.path(message, style=STYLE).ask()
    if answer is None:
        return None
    return answer.strip().strip("'\"")


def ask_text(message: str, default: str = "") -> Optional[str]:
    """Free-text input. Returns None if cancelled."""
    return questionary.text(message, default=default, style=STYLE).ask()


def confirm(message: str, default: bool = False) -> bool:
    """Yes/no prompt. Cancelling counts as False."""
    answer = questionary.confirm(message, default=default, style=STYLE).ask()
    return bool(answer)


def checkbox(message: str, options: Sequence[Tuple[str, object, bool]]) -> Optional[List[object]]:
    """Multi-select. Options are (title, value, checked)."""
    choices = [Choice(title=title, value=value, checked=checked)
               for title, value, checked in options]
    return questionary.checkbox(message, choices=choices, style=STYLE).ask()
