"""Unused at runtime -- the app runs as `python -m src.main`, never imports this.

Exists only because uv_build (see [build-system] in pyproject.toml) requires a
`src/<distribution-name>/__init__.py` module root to build the package at all;
deleting this breaks `uv sync`.
"""
