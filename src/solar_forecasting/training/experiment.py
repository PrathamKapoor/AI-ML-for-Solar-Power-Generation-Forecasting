"""Experiment runner: fit, predict, score and record, with leakage guards.

One run of :func:`run_experiment` produces a complete, self-describing record.
The record is the unit of reproducibility in this project: it contains the
split boundaries, the seed, the feature list, the hyperparameters, the software
versions and the durations, so that a reader can reconstruct the run without
guessing.

The reference forecasts (persistence and smart persistence) are computed first
and stored, because every model's skill score is measured against them on
exactly the same timestamps.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, results_dir
from ..evaluation import metrics as metrics_mod
from ..evaluation import regimes as regime_mod
from ..features.builder import apply_feature_ablation
from ..models import registry as reg
from ..models.baselines import smart_persistence_forecast
from ..models.classical import flatten_windows
from ..preprocessing.pipeline import build_target
from ..training.sequences import SequenceSet, build_sequences, verify_causality
from ..training.trainer import TrainingConfig, predict_neural, train_model
from ..utils.io import write_json
from ..utils.seeding import run_metadata, set_seed


@dataclass
class ExperimentSpec:
    """Declarative description of one experiment run."""

    experiment_id: str
    model: str
    horizon: str
    horizon_steps: int
    features: str = "full"
    seed: int = 42
    ablation: str | None = None
    concurrent_weather: bool = False
    label: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Everything produced by one run."""

    spec: ExperimentSpec
    metrics_daylight: dict[str, float]
    metrics_all_steps: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    predictions: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None
    training_history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "metrics_daylight": self.metrics_daylight,
            "metrics_all_steps": self.metrics_all_steps,
            "metadata": self.metadata,
            # The epoch history is persisted so the training-curve figure can be
            # regenerated from the record alone, without refitting the model.
            "training_history": self.training_history,
            "feature_importance": (self.feature_importance.to_dict(orient="records")
                                   if self.feature_importance is not None else None),
        }


def horizon_lookup(config: Config) -> dict[str, dict[str, Any]]:
    return {h["name"]: h for h in config.section("horizons")}


def select_features(columns: list[str], features_spec: str,
                    config: Config) -> list[str]:
    """Resolve a feature specification to a concrete column list."""
    if features_spec == "full":
        return list(columns)
    groups = config.section("features").get("ablation_groups", {})
    if features_spec not in groups:
        raise KeyError(
            f"unknown feature specification {features_spec!r}. "
            f"Available: 'full' or {sorted(groups)}")
    return apply_feature_ablation(columns, list(columns), groups[features_spec]["remove"])


def build_horizon_frames(featured: pd.DataFrame, horizon_steps: int,
                         target_column: str = "pv_power_w") -> pd.DataFrame:
    """Attach the future target for one horizon.

    The target at row ``t`` becomes the observed power at ``t + horizon_steps``,
    which is what makes a model at row ``t`` a genuine ``horizon_steps``-ahead
    forecast rather than a one-step-ahead forecast with a confusing label.
    """
    return build_target(featured, target_column, horizon_steps)


def reference_forecasts(frame: pd.DataFrame, horizon_steps: int) -> dict[str, np.ndarray]:
    """Persistence and smart-persistence forecasts for the whole frame.

    The returned arrays are indexed by *frame row*, not by sequence. Use
    :func:`aligned_reference_forecasts` to obtain values aligned to a
    :class:`~solar_forecasting.training.sequences.SequenceSet`.
    """
    from ..models.baselines import persistence_from_lag, smart_persistence_from_frame

    return {
        # lag=0 is the observation at the row itself, i.e. the most recent value
        # available at the forecast origin. See persistence_from_lag.
        "persistence": persistence_from_lag(frame, lag=0),
        "smart_persistence": smart_persistence_from_frame(frame, horizon_steps),
    }


def aligned_reference_forecasts(seq: SequenceSet, frame: pd.DataFrame,
                                horizon_steps: int) -> dict[str, np.ndarray]:
    """Reference forecasts re-indexed onto a sequence set's samples.

    Both references are indexed by **forecast origin**, not by target time:

    * persistence at frame row ``j`` is ``pv_power_w[j]`` -- the value held flat
      from the origin;
    * smart persistence at frame row ``j`` combines ``pv_power_w[j]`` with the
      clear-sky ratio from ``j`` to ``j + horizon_steps``.

    So they must be read at ``origin_row``, the row whose observations the
    forecast starts from. Reading them at ``origin_row + horizon_steps`` would
    read the reference one horizon too late -- and for persistence that lands
    exactly on the target, making the reference identical to the truth, the
    skill score collapse to zero information, and every baseline look perfect.
    ``persistence_rmse`` is the canary: if it ever comes back as 0.0, the
    reference is reading the answer.
    """
    full = reference_forecasts(frame, horizon_steps)
    origin_rows = np.asarray(seq.origin_row, dtype=int)
    if origin_rows.max() >= len(frame):
        raise IndexError(
            f"sequence origin row {origin_rows.max()} exceeds the frame length {len(frame)}")
    return {name: np.asarray(values, dtype=float)[origin_rows]
            for name, values in full.items()}
def _daylight_mask(seq: SequenceSet) -> np.ndarray:
    return np.asarray(seq.daylight, dtype=bool)


def score(seq: SequenceSet, predicted_w: np.ndarray, rated_w: float | None,
          reference: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, float]]:
    """Score a forecast on daylight steps (headline) and on all steps.

    Rows with a missing forecast, which occur at the tail of a frame where the
    persistence reference needs a forward value, are dropped consistently from
    both arrays so the two scopes and the reference are compared on identical
    rows.
    """
    actual = np.asarray(seq.target_raw, dtype=float)
    predicted = np.asarray(predicted_w, dtype=float)
    daylight = _daylight_mask(seq)

    valid_all = np.isfinite(actual) & np.isfinite(predicted)
    valid_day = valid_all & daylight

    def compute(mask: np.ndarray) -> dict[str, float]:
        if mask.sum() == 0:
            return {"n": 0.0}
        return metrics_mod.compute_all(actual[mask], predicted[mask], rated_w=rated_w)

    out = {"daylight": compute(valid_day), "all_steps": compute(valid_all)}

    # Skill against both persistence references, on the same rows.
    for scope, mask in (("daylight", valid_day), ("all_steps", valid_all)):
        if mask.sum() == 0:
            continue
        y, f = actual[mask], predicted[mask]
        entry = out[scope]
        for ref_name, ref_values in reference.items():
            ref_masked = np.asarray(ref_values, dtype=float)[mask]
            if not np.isfinite(ref_masked).all():
                continue
            ref_rmse = float(np.sqrt(np.mean((y - ref_masked) ** 2)))
            ref_mae = float(np.mean(np.abs(y - ref_masked)))
            if ref_rmse > metrics_mod.EPS:
                entry[f"skill_vs_{ref_name}_rmse"] = metrics_mod.skill(ref_rmse, entry["rmse"])
            if ref_mae > metrics_mod.EPS:
                entry[f"skill_vs_{ref_name}_mae"] = metrics_mod.skill(ref_mae, entry["mae"])
            entry[f"{ref_name}_rmse"] = ref_rmse
            entry[f"{ref_name}_mae"] = ref_mae
    return out["daylight"], out["all_steps"]


def _build(model_name: str, config: Config, n_features: int):
    """Construct a model with the hyperparameters declared in ``models.yaml``."""
    info = reg.model_info(model_name)
    if info["kind"] == "rule":
        # Rules ignore the input width, but the constructor still requires it so
        # that every model is built through one uniform signature.
        return reg.build_model(model_name, n_features=n_features)
    if info["kind"] == "sklearn":
        return reg.build_model(
            model_name,
            n_features=n_features,
            random_state=42,
            **((config.get("classical", {}) or {}).get(model_name, {})))
    return reg.build_model(
        model_name, n_features=n_features,
        **((config.get("neural", {}) or {}).get(model_name, {})))


def run_experiment(config: Config, spec: ExperimentSpec, featured: pd.DataFrame,
                   split_train: pd.DataFrame, split_val: pd.DataFrame,
                   split_test: pd.DataFrame, columns: list[str],
                   target_scaler: Any, rated_w: float | None,
                   persist_predictions: bool = True,
                   progress: bool = True) -> ExperimentResult:
    """Run one model/horizon/feature-set combination end to end.

    The scalers passed in were fitted on the training split by the pipeline and
    are reused unchanged here, which is what keeps the leakage guard intact
    across every experiment.
    """
    set_seed(spec.seed)

    horizon = spec.horizon
    steps = spec.horizon_steps
    selected = select_features(columns, spec.features, config)

    frames = {
        "train": build_horizon_frames(split_train, steps),
        "val": build_horizon_frames(split_val, steps),
        "test": build_horizon_frames(split_test, steps),
    }
    # Sequences are built per split. Rolling the lookback back across a split
    # boundary would import observations from the neighbouring split, so each
    # split builds its own windows and relies on the warm-up drop performed by
    # the pipeline to cover the lookback.
    lookback = int(config.section("features")["lookback_steps"])
    sequences = {
        name: build_sequences(frame, selected, "target", lookback, steps,
                              target_scaler=target_scaler)
        for name, frame in frames.items()
    }

    reference = {name: aligned_reference_forecasts(sequences[name], frames[name], steps)
                 for name in frames}

    info = reg.model_info(spec.model)
    model = _build(spec.model, config, len(selected))

    metadata: dict[str, Any] = {
        "model_family": info["family"],
        "model_kind": info["kind"],
        "model_description": info["description"],
        "feature_spec": spec.features,
        "n_features": len(selected),
        "feature_columns": list(selected),
        "lookback_steps": lookback,
        "horizon": horizon,
        "horizon_steps": steps,
        "seed": spec.seed,
        "split_sizes": {name: len(seq) for name, seq in sequences.items()},
        "split_date_ranges": {
            name: [str(seq.timestamps.iloc[0]), str(seq.timestamps.iloc[-1])]
            if len(seq) else None
            for name, seq in sequences.items()
        },
        "target_split_dates": {
            name: [str(seq.target_timestamps.iloc[0]), str(seq.target_timestamps.iloc[-1])]
            if len(seq) else None
            for name, seq in sequences.items()
        },
        "concurrent_weather": spec.concurrent_weather,
        "information_set": (
            "weather observed at or before the forecast origin (strict historical setting)"
            if not spec.concurrent_weather else
            "weather observed at the target timestamp (concurrent measurement setting)"),
    }

    history: list[dict[str, float]] = []
    importance: pd.DataFrame | None = None
    infer_ms = float("nan")

    if info["requires_training"] == "no":
        # Rule-based baselines read the lag and clear-sky columns straight from
        # the frame, indexed by the sequence set's own origin and target rows so
        # that the returned array has one entry per sample.
        test_seq = sequences["test"]
        origin_rows = np.asarray(test_seq.origin_row, dtype=int)
        target_rows = origin_rows + steps
        test_frame = frames["test"]
        # The persistence reference is the observation *at* the forecast origin,
        # which is the most recent value available when the forecast is made.
        last = test_frame["pv_power_w"].to_numpy(dtype=float)[origin_rows]
        if spec.model == "persistence":
            predicted_w = last
        elif spec.model == "smart_persistence":
            now = test_frame["ghi_clear"].to_numpy(dtype=float)[origin_rows]
            future = test_frame["ghi_clear"].to_numpy(dtype=float)[target_rows]
            predicted_w = smart_persistence_forecast(last, now, future, rated_w=rated_w)
        else:
            raise KeyError(f"no rule implemented for baseline {spec.model!r}")
        predicted_scaled = predicted_w
        metadata["training"] = {"requires_training": False}
        metadata["cost"] = {"train_seconds": 0.0, "inference_ms_per_window": 0.0,
                            "n_parameters": 0}
    else:
        if info["kind"] == "torch":
            train_cfg = config.get("training", {}) or {}
            tcfg = TrainingConfig(
                epochs=int(train_cfg.get("epochs", 60)),
                batch_size=int(train_cfg.get("batch_size", 256)),
                learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                patience=int(train_cfg.get("patience", 12)),
                device=str(train_cfg.get("device", "cpu")),
                verbose=bool(train_cfg.get("verbose", False)),
            )
            model = _build(spec.model, config, len(selected))
            result = train_model(
                model, sequences["train"], sequences["val"], tcfg,
                target_scale=float(target_scaler.scale_[0]),
                verify=lambda: _verify(sequences, frames, lookback, steps, selected))
            model = result.model
            history = result.history
            scaled_pred, per_window_ms = predict_neural(
                result.model, sequences["test"],
                batch_size=int(train_cfg.get("eval_batch_size", 512)),
                device=str(train_cfg.get("device", "cpu")))
            predicted_scaled = target_scaler.inverse_transform(
                scaled_pred.reshape(-1, 1)).ravel()
            infer_ms = per_window_ms
            metadata["training"] = result.summary()
            metadata["cost"] = {
                "train_seconds": result.train_seconds,
                "inference_ms_per_window": per_window_ms,
                "n_parameters": result.n_parameters,
            }
        else:
            start = time.perf_counter()
            model.fit(sequences["train"].X, sequences["train"].y)
            train_seconds = time.perf_counter() - start
            tuned = {"tuned": False, "reason": "no grid configured for this model"}
            grid = ((config.get("tuning", {}) or {}).get(spec.model) or {}).get("grid")
            if grid:
                tuned = model.tune(sequences["train"].X, sequences["train"].y,
                                   sequences["val"].X, sequences["val"].y)
                train_seconds = time.perf_counter() - start
            start = time.perf_counter()
            predicted_scaled = np.asarray(model.predict(sequences["test"].X), dtype=float)
            infer_seconds = time.perf_counter() - start
            predicted_scaled = target_scaler.inverse_transform(
                predicted_scaled.reshape(-1, 1)).ravel()
            infer_ms = (infer_seconds / max(len(sequences["test"]), 1)) * 1000.0
            metadata["training"] = {"requires_training": True, "train_seconds": train_seconds,
                                    "tuning": tuned, "describe": model.describe()}
            metadata["cost"] = {
                "train_seconds": train_seconds,
                "inference_ms_per_window": infer_ms,
                "n_parameters": model.n_parameters,
            }
            importances = model.feature_importances_
            if importances is not None and len(importances) == len(flatten_windows(
                    sequences["test"].X[:1])):
                importance = collapse_importance(importances, selected, lookback)

    test_seq = sequences["test"]
    daylight, all_steps = score(test_seq, predicted_scaled, rated_w, reference["test"])

    predictions = None
    if persist_predictions:
        predictions = pd.DataFrame({
            "Time": test_seq.target_timestamps.to_numpy(),
            "forecast_origin": test_seq.timestamps.to_numpy(),
            "actual": test_seq.target_raw,
            "predicted": predicted_scaled,
            "daylight": test_seq.daylight,
            "regime": test_seq.regime.astype(str),
            "season": test_seq.season.astype(str),
        })
        for ref_name, ref_values in reference["test"].items():
            predictions[f"reference_{ref_name}"] = ref_values
        predictions["error"] = predictions["predicted"] - predictions["actual"]
        predictions["abs_error"] = predictions["error"].abs()

    metadata["environment"] = run_metadata(spec.seed)
    metadata["dataset"] = {
        "name": config.section("dataset")["name"],
        "primary_station": config.section("dataset")["primary_station"],
        "rated_w": rated_w,
    }
    metadata["experiment_label"] = spec.label
    metadata["notes"] = spec.notes
    metadata["ablation"] = spec.ablation
    metadata["regime_distribution_test"] = regime_mod.regime_distribution(
        test_seq.regime, daylight_only=True)

    result = ExperimentResult(
        spec=spec,
        metrics_daylight=daylight,
        metrics_all_steps=all_steps,
        metadata=metadata,
        predictions=predictions,
        feature_importance=importance,
        training_history=history,
    )

    if progress:
        print(f"    {spec.model:<18} {horizon:<6} "
              f"MAE {daylight.get('mae', float('nan')):>9.1f} W  "
              f"RMSE {daylight.get('rmse', float('nan')):>9.1f} W  "
              f"nRMSE {daylight.get('nrmse_capacity', float('nan')):>6.3f}  "
              f"R2 {daylight.get('r2', float('nan')):>6.3f}  "
              f"skill {daylight.get('skill_vs_persistence_rmse', float('nan')):>6.3f}")
    return result


def _verify(sequences: dict[str, SequenceSet], frames: dict[str, pd.DataFrame],
            lookback: int, steps: int, columns: list[str]) -> None:
    """Assert window causality for every split before training."""
    for name, seq in sequences.items():
        verify_causality(seq, frames[name], lookback, steps, columns, n_checks=25)


def collapse_importance(importances: np.ndarray, feature_columns: list[str],
                        lookback: int) -> pd.DataFrame:
    """Sum per-timestep importances into one value per original feature.

    A tree model sees the flattened window, so each original feature appears
    ``lookback`` times. Reporting the sum keeps the table interpretable: a reader
    wants to know whether irradiance matters, not which of 96 lagged irradiance
    values matters most.
    """
    values = np.asarray(importances, dtype=float).ravel()
    if lookback <= 1 or len(values) == len(feature_columns):
        collapsed = values
    elif len(values) == len(feature_columns) * lookback:
        collapsed = values.reshape(lookback, len(feature_columns)).sum(axis=0)
    else:
        collapsed = values
    total = float(collapsed.sum()) or 1.0
    frame = pd.DataFrame({
        "feature": list(feature_columns)[:len(collapsed)],
        "importance": collapsed,
    })
    frame["relative_importance"] = frame["importance"] / total
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def save_experiment(result: ExperimentResult, directory: Path | None = None) -> Path:
    """Write one experiment record to ``results/experiments``."""
    target = directory or results_dir("experiments")
    path = target / f"{result.spec.experiment_id}.json"
    write_json(result.to_dict(), path)
    return path
