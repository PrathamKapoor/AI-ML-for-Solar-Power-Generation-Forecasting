"""Filesystem helpers for machine-readable results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars/arrays and pandas containers into plain JSON."""

    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            value = float(o)
            return value if np.isfinite(value) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, pd.Timedelta)):
            return str(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        if o is pd.NA or o is pd.NaT:
            return None
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return super().default(o)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: Any, path: Path) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories as needed."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, cls=_NumpyEncoder,
                  sort_keys=False)
        handle.write("\n")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write a DataFrame as CSV with stable float formatting."""
    ensure_parent(path)
    df.to_csv(path, index=index, float_format="%.6g", encoding="utf-8")
    return path


def save_figure(fig, path: Path, dpi: int = 300, also_pdf: bool = True) -> list[Path]:
    """Save a matplotlib figure at publication resolution.

    PNG is written at ``dpi`` for reports and previews; a vector PDF is written
    alongside it because journal submission requires vector artwork for line
    plots. Returns the paths actually written.
    """
    ensure_parent(path)
    written = [path]
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if also_pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        written.append(pdf_path)
    return written
