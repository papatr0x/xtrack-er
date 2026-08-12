"""Audio utility helpers."""


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    minutes, secs = divmod(int(seconds), 60)
    hours, mins = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    else:
        return f"{mins}:{secs:02d}"
