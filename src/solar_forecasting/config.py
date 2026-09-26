"""Configuration loading and project path resolution.

All experiment behaviour is driven by the YAML files in ``configs/``. Nothing in
this package reads a hard-coded machine-specific path: the project root is
derived from this file's location, and every output directory is created on
demand relative to it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or internally inconsistent."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Config:
    """An immutable view over the merged configuration tree."""

    data: dict[str, Any]
    sources: tuple[Path, ...]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def section(self, *keys: str) -> Any:
        """Fetch a nested section, raising a clear error if the path is wrong."""
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"missing config path: {'.'.join(keys)}")
            node = node[key]
        return node

    def with_overrides(self, overrides: dict[str, Any] | None) -> "Config":
        if not overrides:
            return self
        return Config(_deep_merge(self.data, overrides), self.sources)

    @staticmethod
    def project_root_for(folder: str) -> Path:
        """Resolve a top-level project folder (``data``, ``results``, ...)."""
        if folder not in {"data", "results", "configs", "literature", "docs",
                          "paper", "experiments", "notebooks", "scripts", "tests"}:
            raise ConfigError(f"refusing to resolve unknown project folder: {folder!r}")
        return PROJECT_ROOT / folder

    def fingerprint(self) -> str:
        """Stable hash of the effective configuration.

        Recorded with every experiment so a result can be tied to the exact
        settings that produced it.
        """
        blob = json.dumps(self.data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"configuration file must contain a mapping: {path}")
    return loaded


def load_config(
    data_config: Path | str | None = None,
    experiments_config: Path | str | None = None,
    models_config: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load and merge the project configuration.

    Parameters
    ----------
    data_config, experiments_config, models_config
        Optional explicit paths. Defaults to ``configs/data.yaml``,
        ``configs/experiments.yaml`` and ``configs/models.yaml``.
    overrides
        Nested dict merged last, used by the CLI to apply ``--set`` flags.
    """
    paths = {
        "data": Path(data_config) if data_config else CONFIG_DIR / "data.yaml",
        "experiments": (Path(experiments_config) if experiments_config
                        else CONFIG_DIR / "experiments.yaml"),
        "models": Path(models_config) if models_config else CONFIG_DIR / "models.yaml",
    }
    merged: dict[str, Any] = {}
    used: list[Path] = []
    for name, path in paths.items():
        if path.exists():
            section = _read_yaml(path)
            merged = _deep_merge(merged, section)
            used.append(path)
        elif name == "data":
            raise ConfigError(f"required configuration missing: {path}")

    config = Config(merged, tuple(used))
    return config.with_overrides(overrides)


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_for(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def results_dir(*parts: str) -> Path:
    return ensure_dir(RESULTS_DIR.joinpath(*parts))


def data_dir(*parts: str) -> Path:
    return ensure_dir(DATA_DIR.joinpath(*parts))


def set_thread_env(threads: int = 1) -> None:
    """Pin BLAS/OpenMP thread counts so runtimes are comparable across machines.

    Without this, an 8-thread BLAS can make a small model look faster than a
    large one purely because of parallelisation, which would corrupt the
    computational-cost experiment.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(threads))
