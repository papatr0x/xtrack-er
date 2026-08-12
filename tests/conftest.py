"""Puts the repo root on sys.path so tests can `from src... import ...`.

Mirrors how `python -m src.main` is run: from the repo root, with no installed
console-script entry point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
