"""Bootstrap ``src`` onto ``sys.path`` for direct script execution.

Installing the project (``pip install -e .``) is the recommended route, but the
scripts must also work from a fresh clone with nothing installed, so each entry
point calls :func:`ensure_src_on_path` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def ensure_src_on_path() -> Path:
    """Add ``<project>/src`` to ``sys.path`` if it is not already importable."""
    path = str(SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
    return SRC
