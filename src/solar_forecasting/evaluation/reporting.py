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
    # Inference latency is part of the horizon question: a model that is accurate
    # at every horizon may still be unusable at a short one, where the forecast
    # must be emitted on every cadence. It is therefore carried in the table
    # rather than only in the cost table.
    if "inference_ms_per_window" in combined.columns:
        pass
    elif "inference_ms_per_window" in registry.columns:
        combined = combined.merge(registry[["model", "inference_ms_per_window"]]
                                 .drop_duplicates("model"),
                                 on="model", how="left")
    columns = ["model", "model_family", "horizon", "horizon_steps", "source_group",
               "experiment_id", "mae", "rmse", "nrmse_capacity", "nrmse_mean", "r2",
               "smape", "persistence_rmse", "smart_persistence_rmse",
               "skill_vs_persistence_rmse", "skill_vs_smart_persistence_rmse",
               "train_seconds", "inference_ms_per_window", "n_parameters"]
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


# A footprint is measured once per model per process. The measurement involves a
# fit, and the value is a property of the architecture rather than of the run.
_FOOTPRINT_CACHE: dict[str, dict[str, float]] = {}


def model_footprint(model_name: str) -> dict[str, float]:
    """Parameter count, serialised size and training peak memory for one model.

    Parameter count alone understates the deployment footprint of a tree
    ensemble, because a fitted forest is mostly a lookup table rather than a
    weight tensor, so all three measures are reported: the number of trainable
    parameters, the size of the pickled artefact, and the peak resident memory
    observed while fitting. A rule-based model is reported as zero, which is its
    exact cost.
    """
    import pickle
    import tracemalloc

    import numpy as np

    from ..models import registry as model_registry

    info = model_registry.model_info(model_name)
    if info["requires_training"] == "no":
        return {"n_parameters": 0.0, "model_size_bytes": 0.0, "peak_train_memory_mb": 0.0,
                "resident_memory_mb": 0.0, "size_measure": "no fitted state"}
    cached = _FOOTPRINT_CACHE.get(model_name)
    if cached is not None:
        return dict(cached)

    settings: dict[str, Any] = {}
    if info["kind"] == "torch":
        from ..config import load_config
        settings = (load_config().get("neural", {}) or {}).get(model_name, {})
    if info["kind"] == "sklearn":
        from ..config import load_config
        settings = (load_config().get("classical", {}) or {}).get(model_name, {})

    try:
        model = model_registry.build_model(model_name, n_features=16,
                                           random_state=42, **settings)
    except Exception:
        return {"n_parameters": float("nan"), "model_size_bytes": float("nan"),
                "peak_train_memory_mb": float("nan"), "size_measure": "unavailable"}

    rng = np.random.default_rng(0)
    # A sequence model consumes a lookback window, so the probe input has to be
    # three-dimensional for it and two-dimensional for a tabular estimator.
    if info["kind"] == "torch":
        lookback = int((load_config().get("neural", {}) or {}).get("lookback", 24))
        X = rng.normal(size=(400, lookback, 16))
    else:
        X = rng.normal(size=(400, 16))
    y = rng.normal(size=400)
    size_bytes = float("nan")
    peak_mb = float("nan")
    resident_mb = float("nan")
    if info["kind"] == "torch":
        # A sequence forecaster is trained by the experiment runner, not by the
        # adapter, so there is no fit to trace here. The deployment cost is
        # measured instead: build the module and run one inference pass. The
        # training peak is left as NaN rather than reported as zero, because
        # "not measured" and "free" are different claims.
        try:
            import torch

            rss_before = _rss_mb()
            size_bytes = parameter_bytes(model)
            with torch.no_grad():
                model(torch.as_tensor(X[:1], dtype=torch.float32))
            resident_mb = max(0.0, _rss_mb() - rss_before)
        except Exception:
            resident_mb = float("nan")
        _FOOTPRINT_CACHE[model_name] = {
            "n_parameters": fitted_size(model), "model_size_bytes": size_bytes,
            "peak_train_memory_mb": float("nan"), "resident_memory_mb": resident_mb,
            "size_measure": "parameter buffer",
            "memory_measure": "resident after one inference pass"}
        return dict(_FOOTPRINT_CACHE[model_name])
    try:
        rss_before = _rss_mb()
        tracemalloc.start()
        model.fit(X, y)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024.0 * 1024.0)
        # Resident memory attributable to the fitted artefact. Unlike tracemalloc,
        # which only sees Python allocations, this covers the native buffers of a
        # fitted estimator.
        resident_mb = max(0.0, _rss_mb() - rss_before)
    except Exception:
        try:
            tracemalloc.stop()
        except Exception:
            pass
    try:
        size_bytes = float(len(pickle.dumps(model)))
    except Exception:
        size_bytes = parameter_bytes(model)
    _FOOTPRINT_CACHE[model_name] = {
        "n_parameters": fitted_size(model), "model_size_bytes": size_bytes,
        "peak_train_memory_mb": peak_mb, "resident_memory_mb": resident_mb,
        "size_measure": "fitted structure",
        "memory_measure": "python peak plus resident after fit"}
    return dict(_FOOTPRINT_CACHE[model_name])


def _rss_mb() -> float:
    """Resident set size of this process in MB, or NaN if it cannot be read."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def parameter_bytes(model: Any) -> float:
    """Bytes occupied by a neural network's parameters, in float32."""
    import torch

    module = model
    if not isinstance(module, torch.nn.Module):
        inner = getattr(model, "inner_model", None) or getattr(model, "module", None)
        module = inner if isinstance(inner, torch.nn.Module) else None
    if module is None:
        return float("nan")
    return float(sum(p.numel() * p.element_size()
                     for p in module.parameters() if p.requires_grad))


def _flatten(items: Any) -> list:
    "Flatten nested sequences one or two levels deep."
    out: list = []
    for item in items:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(item)
        else:
            out.append(item)
    return out


def fitted_size(model: Any) -> float:
    """A structural size for any fitted estimator, in the same units as parameters.

    A fitted tree ensemble has no weight tensor, so reporting a parameter count of
    zero would understate it by orders of magnitude. The measure is the number of
    stored scalars in the fitted structure: tree nodes for a scikit-learn or
    gradient-boosting ensemble, coefficients for a linear model, and the tensor
    count for a neural network.
    """
    import torch

    inner = getattr(model, "estimator", model)

    if isinstance(inner, torch.nn.Module) or isinstance(model, torch.nn.Module):
        return float(sum(p.numel() for p in inner.parameters() if p.requires_grad))

    if hasattr(inner, "get_booster"):
        # XGBoost: the number of stored tree nodes across the boosted rounds,
        # which is the direct analogue of a scikit-learn ensemble's node count.
        try:
            dumps = inner.get_booster().get_dump(dump_format="json")
            total = sum(dump.count('"nodeid"') for dump in dumps)
            return float(total) if total else float("nan")
        except Exception:
            return float("nan")

    total = 0
    for predictor in _flatten(getattr(inner, "_predictors", []) or []):
        nodes = getattr(predictor, "nodes", None)
        if nodes is not None:
            # HistGradientBoosting: one TreePredictor per boosting iteration, each
            # holding a structured node array. The attribute is nested one level
            # deep in recent scikit-learn versions.
            total += len(nodes)
    if total:
        return float(total)

    if hasattr(inner, "coefs_"):
        # A multi-layer perceptron stores a weight matrix per layer.
        return float(sum(np.asarray(w).size for w in inner.coefs_))

    estimators = getattr(inner, "estimators_", None)
    if estimators is not None:
        total = 0
        for node in np.ravel(estimators):
            tree = getattr(node, "tree", None)
            if tree is not None and hasattr(tree, "node_count"):
                total += int(tree.node_count)
            elif hasattr(node, "tree_"):
                total += int(node.tree_.node_count)
            elif hasattr(node, "estimators_"):     # nested ensemble
                for child in np.ravel(node.estimators_):
                    inner_tree = getattr(child, "tree_", None)
                    if inner_tree is not None:
                        total += int(inner_tree.node_count)
        return float(total) if total else float("nan")

    if hasattr(inner, "coef_"):
        return float(np.asarray(inner.coef_).size)
    n_parameters = getattr(model, "n_parameters", None)
    if n_parameters:
        return float(n_parameters)
    return float("nan")


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
        recorded = row.get("n_parameters")
        n_parameters = float(recorded) if recorded and np.isfinite(recorded) else None
        # The footprint is a property of the architecture, not of the horizon, so
        # it is measured once per model. Re-measuring it per row would both cost
        # a fit each time and report the allocator's noise as if it were a
        # property of the model.
        measured = model_footprint(str(row["model"]))
        size_bytes = measured["model_size_bytes"]
        peak_mb = measured["peak_train_memory_mb"]
        resident_mb = measured["resident_memory_mb"]
        size_measure = measured.get("size_measure")
        # The structural size is reported as the parameter count for every family,
        # because it is the only measure that means the same thing across them.
        # What the runner recorded is kept in its own column: for a tree ensemble
        # it is an estimator count, not a parameter count, and labelling it as
        # one would make a 200-tree forest look three orders of magnitude
        # smaller than a 200-weight layer.
        n_parameters = measured["n_parameters"]
        n_parameters_recorded = float(recorded) if recorded is not None else None
        entries = {
            "model": row["model"],
            "model_family": MODEL_FAMILIES.get(row["model"], row.get("model_family")),
            "horizon": row["horizon"],
            "requires_training": bool(train_seconds > 0),
            "train_seconds": train_seconds,
            "inference_ms_per_window": inference_ms,
            "n_parameters": n_parameters,
            "n_parameters_recorded": n_parameters_recorded,
            "nrmse_capacity": float(row.get("nrmse_capacity", np.nan)),
            "rmse": float(row.get("rmse", np.nan)),
        }
        if size_bytes is not None:
            entries["model_size_bytes"] = size_bytes
            entries["model_size_kb"] = size_bytes / 1024.0
        if peak_mb is not None:
            entries["peak_train_memory_mb"] = peak_mb
        if resident_mb is not None:
            entries["resident_memory_mb"] = resident_mb
        if size_measure:
            entries["n_parameters_measure"] = size_measure
        entries["thread_note"] = (
            "BLAS and OpenMP thread counts are pinned to 1 for every run "
            "so the durations are comparable across machines; absolute "
            "values are machine specific and only the ranking is portable")
        rows.append(entries)
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





