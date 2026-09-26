"""Reproducibility helpers: global seeding and captured software versions."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

DEFAULT_SEED = 42

# Packages whose versions materially affect numerical results. Recorded with
# every experiment so a result can be reproduced or explained.
TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "shap",
    "matplotlib", "seaborn", "pvlib", "statsmodels", "openpyxl", "pyyaml",
    "requests", "tqdm", "joblib",
]


def set_seed(seed: int = DEFAULT_SEED, deterministic_torch: bool = True) -> None:
    """Seed every random source the project touches.

    Parameters
    ----------
    seed
        The seed value.
    deterministic_torch
        When True, forces cuDNN into deterministic mode. On CPU this has no
        effect but is harmless, and it keeps behaviour identical if the project
        is later run on a GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the classical-only path
        pass


def seeded_rng(seed: int = DEFAULT_SEED, offset: int = 0) -> np.random.Generator:
    """Return an independent, explicitly seeded generator.

    Used for bootstrap resampling and any other stochastic component, so that
    adding a new random step in one place cannot shift the stream elsewhere.
    """
    return np.random.default_rng(seed + offset)


def package_versions() -> dict[str, str]:
    """Version string for every tracked package that is importable."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - a missing optional dep is not an error
            versions[name] = "not installed"
    return versions


def torch_info() -> dict[str, Any]:
    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                            else "cpu"),
            "num_threads": int(torch.get_num_threads()),
        }
    except Exception:  # noqa: BLE001
        return {"version": "not installed"}


def platform_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def git_revision() -> str:
    """Current git commit, or a clear marker when not inside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
        )
        return out.stdout.strip() if out.returncode == 0 else "not a git repository"
    except Exception:  # noqa: BLE001
        return "unavailable"


def run_metadata(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Full provenance block attached to every experiment record."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "git_revision": git_revision(),
        "platform": platform_info(),
        "torch": torch_info(),
        "packages": package_versions(),
    }
