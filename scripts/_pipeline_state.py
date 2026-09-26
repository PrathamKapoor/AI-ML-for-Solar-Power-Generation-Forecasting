#!/usr/bin/env python
"""Shared helpers for the command-line entry points.

Kept in ``scripts`` rather than in the package because it is glue, not library
code: it resolves the project layout and rebuilds the in-memory pipeline state
from the cached artefacts on disk, so that ``train.py`` and ``evaluate.py`` do
not each re-implement it.

``load_pipeline_state`` reconstructs the exact state that
:func:`solar_forecasting.preprocessing.pipeline.run_pipeline` returns, by reading
the cached parquet files and refitting the scalers on the cached training split.
The refit is not an approximation: the cached training frame is the same frame the
pipeline produced, so the scaler statistics, and therefore every downstream
model, are identical to a full pipeline run. The alternative -- pickling fitted
estimators into the repository -- was rejected because binary objects in a
repository are unreviewable and platform-dependent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from solar_forecasting.config import Config
from solar_forecasting.preprocessing import splitting as split_mod


def results_dir(config: Config) -> Path:
    path = config.project_root_for("results") / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_pipeline_state(config: Config) -> dict[str, Any]:
    """Rebuild the pipeline result dictionary from the cached processed artefacts."""
    processed = config.project_root_for("data") / "processed"
    required = {
        "featured": processed / "featured_primary_station.parquet",
        "columns": processed / "feature_columns.json",
        "train": processed / "split_train.parquet",
        "val": processed / "split_val.parquet",
        "test": processed / "split_test.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "processed data is missing: " + ", ".join(missing) +
            "\nRun: python scripts/preprocess.py")

    featured = pd.read_parquet(required["featured"])
    with required["columns"].open(encoding="utf-8") as handle:
        columns = json.load(handle)["feature_columns"]
    splits = {name: pd.read_parquet(required[key])
              for name, key in (("train", "train"), ("val", "val"), ("test", "test"))}
    scalers = split_mod.fit_scalers(splits["train"], columns, config.section("dataset")["target_column"])

    report_path = results_dir(config) / "pipeline_report.json"
    station: dict[str, Any] = {}
    if report_path.exists():
        stages = json.loads(report_path.read_text(encoding="utf-8")).get("stages", {})
        for stage in stages.values():
            if isinstance(stage, dict) and stage.get("rated_w"):
                station = stage
                break

    split = split_mod.SplitResult(
        train=splits["train"], val=splits["val"], test=splits["test"],
        boundaries=split_mod.boundaries_from_config(config.section("split")))
    return {"report": {}, "featured": featured, "split": split,
            "columns": columns, "scalers": scalers, "station": station}
