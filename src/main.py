"""xtrack-er entry point.

Dispatches to one of three independent tools. Each does exactly one job and
talks to the next only through files: a recording is a file, a separation is a
cache entry keyed off a file's content. Nothing here holds state that spans
commands.
"""

import sys

USAGE = """xtrack-er

Usage:
  uv run python -m src.main record            Capture audio to a file
  uv run python -m src.main separate [path]   Split a file into tracks
  uv run python -m src.main play              Mix and play separated tracks
"""

COMMANDS = {"record", "separate", "play"}


def main() -> None:
    """Dispatch to record/separate/play based on `sys.argv[1]`."""
    args = sys.argv[1:]
    command = args[0] if args else None

    if command not in COMMANDS:
        print(USAGE)
        if command not in (None, "-h", "--help"):
            print(f"Unknown command: {command}")
            sys.exit(1)
        return

    # Let the chosen tool read the rest of argv as if it were run directly.
    sys.argv = sys.argv[:1] + args[1:]

    if command == "record":
        from src.app.record_cli import main as run
    elif command == "separate":
        from src.app.separate_cli import main as run
    else:
        from src.app.play_cli import main as run

    run()


if __name__ == "__main__":
    main()
