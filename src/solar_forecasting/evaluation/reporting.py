"""Results tables.

Every table in the paper and in the results directory is produced here, from the
experiment registry and from stored predictions, so that a table can never drift
away from the run that produced it. Each builder returns a DataFrame and a short
provenance string; :mod:`scripts.evaluate` writes both.

The tables are deliberately wide rather than long: a reader comparing models
wants the metrics side by side for the same run, and the registry already carries
the split boundaries, seed and configuration fingerprint that make a row
interpretable.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation.metrics import documentation_table
from ..evaluation.stratified import stratified_table

#: Columns that make up the headline comparison, in reading order.
HEADLINE_COLUMNS = [
    "experiment_id", "model", "model_family", "horizon", "feature_spec", "seed",
    "n_train", "n_val", "n_test", "n_features",
    "mae", "rmse", "nrmse_capacity", "nrmse_mean", "r2", "smape", "bias", "peak_error",
    "persistence_rmse", "smart_persistence_rmse",
    "skill_vs_persistence_rmse", "skill_vs_smart_persistence_rmse",
    "rmse_all_steps", "mae_all_steps", "r2_all_steps",
    "train_seconds", "inference_ms_per_window", "n_parameters",
    "config_fingerprint", "git_revision", "python_version", "torch_version", "notes",
]

MODEL_ORDER = [
    "persistence", "smart_persistence", "linear_regression", "random_forest",
    "gradient_boosting", "xgboost", "lstm", "gru", "cnn_lstm", "attention_lstm",
    "transformer",
]

MODEL_FAMILIES = {
    "persistence": "rule-based reference",
    "smart_persistence": "rule-based reference",
    "linear_regression": "classical machine learning",
    "random_forest": "classical machine learning",
    "gradient_boosting": "classical machine learning",
    "xgboost": "classical machine learning",
    "lstm": "recurrent neural network",
    "gru": "recurrent neural network",
    "cnn_lstm": "hybrid convolutional-recurrent",
    "attention_lstm": "hybrid recurrent-attention",
    "transformer": "attention-only transformer",
}


def _ordered(frame: pd.DataFrame, column: str, order: Sequence[str]) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    ranked = {name: i for i, name in enumerate(order)}
    return frame.sort_values(column, key=lambda s: s.map(lambda v: ranked.get(v, 99)),
                             kind="stable")


def overall_comparison(registry: pd.DataFrame, horizon: str = "1h",
                       feature_spec: str = "full",
                       group: str | None = "B_ml_vs_dl") -> pd.DataFrame:
    """The headline table: every model at one horizon and one input set.

    When several experiment groups contain the same ``(model, horizon,
    feature_spec)`` — the baseline group repeats a subset of the headline group
    under an identical configuration — the requested group is preferred and the
    duplicate is dropped, so the table has one row per model rather than one row
    per run of the same configuration.
    """
    if registry.empty:
        return registry
    frame = registry[(registry["horizon"] == horizon)
                     & (registry["feature_spec"] == feature_spec)].copy()
    if group and group in set(frame["experiment_group"]):
        frame = frame[frame["experiment_group"] == group]
    frame = frame.drop_duplicates(subset=["model"], keep="first")
    frame = _ordered(frame, "model", MODEL_ORDER)
    columns = [c for c in HEADLINE_COLUMNS if c in frame.columns]
    return frame[columns].reset_index(drop=True)


def horizon_comparison(registry: pd.DataFrame,
                       groups: Sequence[str] = ("C_horizons", "B_ml_vs_dl")
                       ) -> pd.DataFrame:
    """Accuracy by horizon, accepting the 1-hour runs from either group.

    The horizon experiment and the headline experiment train under the same
    seed, split and configuration, so where both contain a ``(model, horizon)``
    pair the two runs are the same run. Rows from the horizon group are therefore
    preferred, and for any horizon the horizon group does not cover, the
    headline group's row is used instead. The ``source_group`` column records
    which record each row came from, so the substitution is visible in the output
    rather than buried in the code.
    """
    if registry.empty:
        return registry
    frame = registry[registry["experiment_group"].isin(list(groups))].copy()
    full = frame[frame["feature_spec"] == "full"]
    preferred_group, *others = groups
    preferred = full[full["experiment_group"] == preferred_group]
    covered = set(zip(preferred["model"], preferred["horizon"]))
    fallback = full[full["experiment_group"] != preferred_group]
    fallback = fallback[[(m, h) not in covered
                         for m, h in zip(fallback["model"], fallback["horizon"])]]
    combined = pd.concat([preferred, fallback], ignore_index=True)
    combined = combined.drop_duplicates(subset=["model", "horizon"], keep="first")
    combined["source_group"] = combined["experiment_group"]
    combined = _ordered(combined, "model", MODEL_ORDER)
    columns = ["model", "model_family", "horizon", "horizon_steps", "source_group",
               "experiment_id", "rmse", "mae", "nrmse_capacity", "nrmse_mean", "r2",
               "smape", "persistence_rmse", "smart_persistence_rmse",
               "skill_vs_persistence_rmse", "skill_vs_smart_persistence_rmse",
               "train_seconds", "n_parameters"]
    return combined[[c for c in columns if c in combined.columns]].reset_index(drop=True)


def regime_comparison(predictions: Mapping[str, pd.DataFrame], rated_w: float | None,
                      horizon: str = "1h") -> pd.DataFrame:
    """Regime-stratified metrics for every model with stored predictions."""
    frames = [stratified_table(frame, rated_w, model, horizon, strata=("regime",))
              for model, frame in predictions.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    return _ordered(table, "model", MODEL_ORDER)


def seasonal_comparison(predictions: Mapping[str, pd.DataFrame], rated_w: float | None,
                        horizon: str = "1h") -> pd.DataFrame:
    """Season-stratified metrics for every model with stored predictions."""
    frames = [stratified_table(frame, rated_w, model, horizon, strata=("season",))
              for model, frame in predictions.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return _ordered(pd.concat(frames, ignore_index=True), "model", MODEL_ORDER)


def full_error_analysis(predictions: Mapping[str, pd.DataFrame], rated_w: float | None,
                        horizon: str = "1h") -> pd.DataFrame:
    """Every stratification for every model: regime, season, time of day, level, ramp."""
    frames = [stratified_table(frame, rated_w, model, horizon)
              for model, frame in predictions.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return _ordered(pd.concat(frames, ignore_index=True), "model", MODEL_ORDER)


def feature_ablation(registry: pd.DataFrame) -> pd.DataFrame:
    """Every feature-ablation run, with degradation relative to the full input set."""
    if registry.empty:
        return registry
    frame = registry[registry["experiment_group"] == "E_feature_ablation"].copy()
    if frame.empty:
        return frame
    frame = _ordered(frame, "model", MODEL_ORDER)
    reference = (frame[frame["feature_spec"] == "full"]
                 .set_index("model")[["rmse", "mae", "nrmse_capacity", "r2"]])
    for metric in ("rmse", "mae", "nrmse_capacity"):
        if metric in frame.columns:
            base = frame["model"].map(reference[metric])
            frame[f"{metric}_vs_full_pct"] = 100.0 * (frame[metric] - base) / base
    frame["skill_vs_persistence_rmse"] = frame.get("skill_vs_persistence_rmse")
    columns = ["experiment_id", "model", "feature_spec", "n_features", "rmse", "mae",
               "nrmse_capacity", "r2", "smape", "rmse_vs_full_pct", "mae_vs_full_pct",
               "nrmse_capacity_vs_full_pct", "skill_vs_persistence_rmse", "seed",
               "train_seconds"]
    return frame[[c for c in columns if c in frame.columns]].reset_index(drop=True)


def feature_regime_comparison(registry: pd.DataFrame,
                              group: str = "N_feature_regimes") -> pd.DataFrame:
    """Additive input regimes: what each kind of information is worth alone.

    Complements :func:`feature_ablation`, which is leave-one-out. Here each row
    is a model given only one kind of input, so the comparison answers "can a
    model forecast from irradiance alone?" rather than "what does this group add
    to everything else?".
    """
    if registry.empty:
        return registry
    frame = registry[registry["experiment_group"] == group].copy()
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["model", "feature_spec"], keep="first")
    frame = _ordered(frame, "model", MODEL_ORDER)
    order = ["pv_only", "weather_only", "pv_weather", "pv_weather_solar", "full"]
    ranked = {name: i for i, name in enumerate(order)}
    frame = frame.sort_values("feature_spec", key=lambda s: s.map(lambda v: ranked.get(v, 99)))
    columns = ["experiment_id", "model", "feature_spec", "n_features", "mae", "rmse",
               "nrmse_capacity", "r2", "smape", "skill_vs_persistence_rmse", "seed",
               "train_seconds"]
    return frame[[c for c in columns if c in frame.columns]].reset_index(drop=True)


def model_ablation(registry: pd.DataFrame, config: Config | None = None
                   ) -> pd.DataFrame:
    """Composite architectures against their backbones, from the headline runs.

    Every model here was trained under the same seed, split and input set, so the
    difference in accuracy is attributable to the architecture. No significance is
    claimed at this stage: the paired test on the same rows is in the statistical
    tables.
    """
    if registry.empty:
        return registry
    frame = registry[(registry["experiment_group"] == "B_ml_vs_dl")
                     & (registry["feature_spec"] == "full")].copy()
    if frame.empty or "model" not in frame.columns:
        return frame
    backbones = ["lstm", "gru"]
    composites = ["cnn_lstm", "attention_lstm", "transformer"]
    lookup = frame.set_index("model")
    rows: list[dict[str, Any]] = []
    for composite in composites:
        if composite not in lookup.index:
            continue
        composite_row = lookup.loc[composite]
        best_backbone, best_error = None, np.inf
        for backbone in backbones:
            if backbone in lookup.index:
                value = float(lookup.loc[backbone, "rmse"])
                if value < best_error:
                    best_backbone, best_error = backbone, value
        entry = {
            "composite": composite,
            "backbone": best_backbone,
            "composite_rmse": float(composite_row["rmse"]),
            "backbone_rmse": best_error,
            "rmse_change_w": float(composite_row["rmse"]) - best_error,
            "rmse_change_pct": 100.0 * (float(composite_row["rmse"]) - best_error) / best_error,
            "composite_mae": float(composite_row["mae"]),
            "composite_nrmse_capacity": float(composite_row["nrmse_capacity"]),
            "composite_r2": float(composite_row["r2"]),
            "composite_train_seconds": float(composite_row.get("train_seconds", np.nan)),
            "composite_parameters": float(composite_row.get("n_parameters", np.nan)),
            "horizon": composite_row["horizon"],
            "interpretation": (
                "Negative RMSE change means the composite is better than the best of "
                "its recurrent backbones under an identical protocol. A positive value "
                "means the added structure did not pay for itself on this dataset."
            ),
        }
        rows.append(entry)
    return pd.DataFrame(rows)


def computational_cost(registry: pd.DataFrame) -> pd.DataFrame:
    """Training duration, inference latency, parameter count and the Pareto flag."""
    if registry.empty:
        return registry
    frame = registry.copy()
    if "feature_spec" in frame.columns:
        frame = frame[frame["feature_spec"] == "full"]
    frame = frame.drop_duplicates(subset=["model", "horizon"], keep="first")
    rows = []
    for _, row in _ordered(frame, "model", MODEL_ORDER).iterrows():
        train_seconds = float(row.get("train_seconds", np.nan) or 0.0)
        inference_ms = float(row.get("inference_ms_per_window", np.nan) or 0.0)
        rows.append({
            "model": row["model"],
            "model_family": MODEL_FAMILIES.get(row["model"], row.get("model_family")),
            "horizon": row["horizon"],
            "requires_training": bool(train_seconds > 0),
            "train_seconds": train_seconds,
            "inference_ms_per_window": inference_ms,
            "n_parameters": float(row.get("n_parameters", np.nan) or 0.0),
            "nrmse_capacity": float(row.get("nrmse_capacity", np.nan)),
            "rmse": float(row.get("rmse", np.nan)),
            "thread_note": ("BLAS and OpenMP thread counts are pinned to 1 for every run "
                            "so the durations are comparable across machines; absolute "
                            "values are machine specific and only the ranking is portable"),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    frontier: list[bool] = []
    best = np.inf
    for _, row in table.sort_values(["train_seconds", "nrmse_capacity"]).iterrows():
        if row["nrmse_capacity"] < best:
            best = row["nrmse_capacity"]
            frontier.append(True)
        else:
            frontier.append(False)
    order = table.sort_values(["train_seconds", "nrmse_capacity"]).index
    flags = pd.Series(frontier, index=order)
    table["pareto_optimal"] = [bool(flags.loc[i]) for i in table.index]
    return table.reset_index(drop=True)


def metric_documentation() -> pd.DataFrame:
    """Formula, interpretation and limitation of every metric computed."""
    return documentation_table()


def dataset_statistics(featured: pd.DataFrame, station: Mapping[str, Any],
                       splits: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Descriptive statistics of the case-study record, for the paper's dataset section."""
    frame = featured
    target = "pv_power_w"
    rows: list[dict[str, Any]] = [
        {"quantity": "records", "value": float(len(frame)),
         "unit": "15-minute steps", "note": "after cleaning and daylight labelling"},
        {"quantity": "first timestamp", "value": str(pd.to_datetime(frame["Time"]).min()),
         "unit": "", "note": "start of the usable record"},
        {"quantity": "last timestamp", "value": str(pd.to_datetime(frame["Time"]).max()),
         "unit": "", "note": "end of the usable record"},
        {"quantity": "rated capacity", "value": float(station.get("rated_w", np.nan)),
         "unit": "W", "note": "nameplate AC rating from the dataset metadata file"},
        {"quantity": "maximum observed power", "value": float(frame[target].max()),
         "unit": "W", "note": "independent check on the rated capacity"},
        {"quantity": "mean power (all steps)", "value": float(frame[target].mean()),
         "unit": "W", "note": "includes night-time zeros"},
        {"quantity": "mean power (daylight)", "value": float(
            frame.loc[frame["daylight"].astype(bool), target].mean()),
         "unit": "W", "note": "daylight steps only, the headline evaluation scope"},
        {"quantity": "capacity factor", "value": float(
            frame[target].mean() / station.get("rated_w", np.nan)),
         "unit": "fraction", "note": "mean power over rated capacity, full record"},
        {"quantity": "daylight fraction", "value": float(frame["daylight"].astype(bool).mean()),
         "unit": "fraction", "note": "share of steps with the sun above the horizon"},
        {"quantity": "zero-power fraction (daylight)", "value": float(
            (frame.loc[frame["daylight"].astype(bool), target] <= 0).mean()),
         "unit": "fraction", "note": "why MAPE is not defined for this target"},
    ]
    for column in ("ghi", "temperature", "relative_humidity", "wind_speed"):
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            rows.append({"quantity": f"{column} mean", "value": float(series.mean()),
                         "unit": "native", "note": "post-aggregation, 15-minute resolution"})
            rows.append({"quantity": f"{column} missing", "value": float(series.isna().mean()),
                         "unit": "fraction", "note": "before imputation, for the record"})
    if splits:
        for name, split in splits.items():
            rows.append({"quantity": f"{name} split size", "value": float(len(split)),
                         "unit": "15-minute steps", "note": f"{name} period after warm-up drop"})
    return pd.DataFrame(rows)


def results_summary(registry: pd.DataFrame) -> dict[str, Any]:
    """Compact machine-readable digest of everything that has been run."""
    if registry.empty:
        return {"n_experiments": 0}
    best = registry.dropna(subset=["nrmse_capacity"])
    headline = overall_comparison(registry)
    return {
        "n_experiments": int(len(registry)),
        "groups": sorted(registry["experiment_group"].unique().tolist()),
        "models": sorted(registry["model"].unique().tolist()),
        "horizons": sorted(registry["horizon"].unique().tolist()),
        "feature_specs": sorted(registry["feature_spec"].unique().tolist()),
        "best_nrmse_run": (best.loc[best["nrmse_capacity"].idxmin(),
                                   ["experiment_id", "model", "horizon", "feature_spec",
                                    "nrmse_capacity"]].to_dict()
                           if not best.empty else None),
        "best_rmse_run": (registry.dropna(subset=["rmse"])
                          .loc[lambda d: d["rmse"].idxmin(),
                               ["experiment_id", "model", "horizon", "feature_spec",
                                "rmse"]].to_dict()
                          if registry["rmse"].notna().any() else None),
        "models_beating_persistence_at_1h": sorted(
            headline.loc[headline["skill_vs_persistence_rmse"] > 0, "model"].tolist()
        ) if "skill_vs_persistence_rmse" in headline.columns else [],
        "note": ("Skill is reported against the persistence reference on identical "
                 "timestamps. A negative skill means the model is worse than holding "
                 "the most recent observation forward, which is a meaningful and "
                 "reportable outcome, not a failure of the experiment."),
    }


def format_markdown_table(frame: pd.DataFrame, float_format: str = "{:,.4g}",
                          max_rows: int | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if frame.empty:
        return "_no rows available_"
    data = frame.head(int(max_rows)) if max_rows else frame
    columns = [str(c) for c in data.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in data.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else float_format.format(value))
            elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                cells.append("")
            else:
                cells.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    if max_rows and len(frame) > int(max_rows):
        lines.append(f"| _... {len(frame) - int(max_rows)} further rows in "
                     f"{'the CSV'} |" + " |" * (len(columns) - 2) + " |")
    return "\n".join(lines)
