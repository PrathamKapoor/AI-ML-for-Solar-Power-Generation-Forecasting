"""Explainability analysis for the headline models (Experiment H).

This module refits one model under the declared configuration and then measures
how much each input variable contributes to its forecasts. Refitting rather than
loading a saved estimator is deliberate: the repository stores no binary weights,
and under the recorded seed the refit reproduces the same model, so the analysis
remains reproducible.

Two methods are used, one per model family, and the choice is recorded with the
result:

* **Tree ensemble (XGBoost): SHAP.** Exact additive attributions from
  ``TreeExplainer``, collapsed over the lookback window, plus the estimator's own
  impurity-based importances for comparison.
* **Recurrent and attention models: grouped permutation importance.** A variable
  is scrambled across the whole window and the increase in daylight MAE is
  recorded. Attention weights are not used as explanations; see
  :mod:`solar_forecasting.explainability.importance` for why.

Both are *predictive* attributions. They describe what the fitted model relies on,
not what physically causes the output, and no causal claim is made anywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..evaluation import metrics as metrics_mod
from ..explainability.importance import (collapse_over_lookback, dependence_frame,
                                         permutation_importance_by_regime,
                                         permutation_importance_sequence,
                                         shap_importance)
from ..models import registry as model_registry
from ..models.classical import flatten_windows
from ..training.experiment import _build, build_horizon_frames, select_features
from ..training.sequences import build_sequences
from ..training.trainer import TrainingConfig, predict_neural, train_model


def _training_config(config: Config) -> TrainingConfig:
    train_cfg = config.get("training", {}) or {}
    return TrainingConfig(
        epochs=int(train_cfg.get("epochs", 70)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        patience=int(train_cfg.get("patience", 14)),
        device=str(train_cfg.get("device", "cpu")),
        verbose=bool(train_cfg.get("verbose", False)),
    )


def _prepare_sequences(config: Config, state: dict[str, Any], features: str = "full"
                       ) -> dict[str, Any]:
    """Rebuild the windowed test set exactly as the experiment runner does."""
    columns = state["columns"]
    selected = select_features(columns, features, config)
    steps = state["horizon_steps"]
    lookback = int(config.section("features")["lookback_steps"])
    frame = build_horizon_frames(state["split"].test, steps)
    sequence = build_sequences(frame, selected, "target", lookback, steps,
                               target_scaler=state["scalers"].target_scaler)
    return {"frame": frame, "sequence": sequence, "selected": selected,
            "lookback": lookback, "steps": steps}


def _fit(config: Config, model_name: str, state: dict[str, Any], prepared: dict[str, Any]):
    """Fit one model and return ``(estimator_or_module, kind)``."""
    info = model_registry.model_info(model_name)
    selected = prepared["selected"]
    n_features = len(selected)
    steps = prepared["steps"]
    lookback = prepared["lookback"]

    if info["kind"] == "sklearn":
        settings = (config.get("classical", {}) or {}).get(model_name, {})
        estimator = model_registry.build_model(model_name, n_features=n_features,
                                               random_state=42, **settings)
        train_frame = build_horizon_frames(state["split"].train, steps)
        train_seq = build_sequences(train_frame, selected, "target", lookback, steps,
                                    target_scaler=state["scalers"].target_scaler)
        estimator.fit(flatten_windows(train_seq.X), train_seq.y)
        return estimator, "sklearn"

    if info["kind"] == "torch":
        train_frame = build_horizon_frames(state["split"].train, steps)
        val_frame = build_horizon_frames(state["split"].val, steps)
        train_seq = build_sequences(train_frame, selected, "target", lookback, steps,
                                    target_scaler=state["scalers"].target_scaler)
        val_seq = build_sequences(val_frame, selected, "target", lookback, steps,
                                  target_scaler=state["scalers"].target_scaler)
        module = _build(model_name, config, n_features)
        result = train_model(module, train_seq, val_seq, _training_config(config),
                             target_scale=float(state["scalers"].target_scaler.scale_[0]))
        return result.model, "torch"

    raise KeyError(f"explainability is not defined for model kind {info['kind']!r}")


def _shap_values(estimator, X: np.ndarray, seed: int = 42, n_samples: int = 600
                 ) -> pd.DataFrame | None:
    """Raw SHAP values at the origin timestep, for beeswarm and dependence plots.

    The mean-|SHAP| table collapses over the whole window, which is the right
    summary for "which variable matters" but the wrong input for a beeswarm
    plot, where the *distribution* is the point. The origin step is used, because
    that is the value present at decision time.
    """
    try:
        import shap
    except Exception:  # pragma: no cover - optional dependency
        return None
    rng = np.random.default_rng(seed)
    n = min(int(n_samples), len(X))
    index = rng.choice(len(X), size=n, replace=False) if n < len(X) else np.arange(len(X))
    tensor = X[index]
    origin = tensor[:, -1, :]
    flat = flatten_windows(tensor)
    try:
        explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
        values = explainer.shap_values(flat)
    except Exception:  # pragma: no cover - estimator-specific failure
        return None
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    columns = list(estimator.feature_names_in_) if hasattr(estimator, "feature_names_in_") \
        else None
    width = values.shape[1]
    step = width // origin.shape[1]
    origin_values = values[:, -origin.shape[1] * step:]
    # Keep the final `n_features` columns, which correspond to the origin step.
    n_features = origin.shape[1]
    origin_values = origin_values[:, -n_features:]
    names = (columns[-n_features:] if columns and len(columns) >= n_features
             else [f"f{i}" for i in range(n_features)])
    return pd.DataFrame(origin_values, columns=names)


def _shap_dependence(X: np.ndarray, shap_values: pd.DataFrame | None,
                     selected: list[str], top_features: Sequence[str]
                     ) -> pd.DataFrame | None:
    """SHAP attribution against the value of the top features at the origin."""
    if shap_values is None or not top_features:
        return None
    origin = X[:, -1, :]
    frame = pd.DataFrame(origin, columns=selected)
    for feature in top_features:
        if feature in shap_values.columns and feature in frame.columns:
            frame[f"shap_{feature}"] = shap_values[feature].to_numpy()
    return frame


def explain_model(config: Config, model_name: str, state: dict[str, Any],
                  n_permutation_repeats: int = 10, n_shap_samples: int = 2000,
                  n_evaluation_samples: int = 4000, seed: int = 42) -> dict[str, Any]:
    """Refit one model and return its importances plus a plain-language summary."""
    prepared = _prepare_sequences(config, state)
    sequence = prepared["sequence"]
    selected = prepared["selected"]
    lookback = prepared["lookback"]
    target_scaler = state["scalers"].target_scaler

    fitted, kind = _fit(config, model_name, state, prepared)

    daylight = np.asarray(sequence.daylight, dtype=bool)
    evaluation_index = np.flatnonzero(daylight)[: int(n_evaluation_samples)]
    if evaluation_index.size < 50:
        evaluation_index = np.flatnonzero(daylight)[: int(n_evaluation_samples)]
    if evaluation_index.size == 0:
        raise ValueError("no daylight windows available for the explainability analysis")
    X = sequence.X[evaluation_index]
    y_true = sequence.target_raw[evaluation_index]

    out: dict[str, Any] = {
        "model": model_name,
        "model_kind": kind,
        "horizon": state.get("horizon", "1h"),
        "horizon_steps": prepared["steps"],
        "lookback_steps": lookback,
        "n_features": len(selected),
        "n_evaluation_samples": int(evaluation_index.size),
        "scope": "daylight test windows",
    }

    if kind == "sklearn":
        predicted = np.asarray(fitted.predict(flatten_windows(X)), dtype=float)
        out["baseline_metrics"] = metrics_mod.compute_all(y_true, predicted)
        shap = shap_importance(fitted, X, selected, lookback, n_samples=n_shap_samples,
                               seed=seed)
        out["shap"] = {k: v for k, v in shap.items() if k != "importance"}
        out["importance"] = shap["importance"].rename("importance").to_frame()
        out["method"] = "SHAP TreeExplainer (mean |shap|), collapsed over the lookback"
        built_in = getattr(fitted, "feature_importances_", None)
        if built_in is not None:
            out["built_in_importance"] = collapse_over_lookback(
                built_in, selected, lookback).rename("importance").to_frame()
        out["shap_values"] = _shap_values(fitted, X, seed=seed, n_samples=600)
        out["dependence"] = _shap_dependence(X, out["shap_values"], selected,
                                             top_features=out["top_features"][:3])
        built = out.get("shap_values")
        if built is not None:
            out["_shap_frame"] = pd.concat(
                [built[[c for c in selected if c in built.columns]].reset_index(drop=True),
                 pd.DataFrame({"predicted": predicted, "actual": y_true})], axis=1)
    else:
        device = str((config.get("training", {}) or {}).get("device", "cpu"))
        scaled, _ = predict_neural(fitted, sequence.subset(evaluation_index),
                                   batch_size=int((config.get("training", {}) or {})
                                                  .get("eval_batch_size", 512)),
                                   device=device)
        predicted = target_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()
        out["baseline_metrics"] = metrics_mod.compute_all(y_true, predicted)

        def predict_fn(tensor: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                raw = fitted(torch.as_tensor(np.asarray(tensor, dtype=np.float32)))
            values = np.asarray(raw.detach().cpu().numpy(), dtype=float).reshape(-1, 1)
            return target_scaler.inverse_transform(values).ravel()

        permutation = permutation_importance_sequence(
            predict_fn, X, y_true, selected, lookback, n_repeats=int(n_permutation_repeats),
            seed=seed)
        out["permutation"] = {"n_repeats": int(n_permutation_repeats),
                             "baseline_mae_w": float(np.abs(predicted - y_true).mean())}
        out["importance"] = permutation
        out["method"] = ("grouped permutation importance over the lookback window "
                         "(increase in daylight MAE); attention weights are not "
                         "treated as explanations")
        out["regime_importance"] = permutation_importance_by_regime(
            predict_fn, X, y_true, selected, lookback, np.asarray(sequence.regime)[
                evaluation_index], n_repeats=max(3, int(n_permutation_repeats) // 2),
            seed=seed)
        out["dependence"] = dependence_frame(
            predict_fn, X, y_true, selected, lookback,
            out["top_features"][:3], n_samples=min(3000, int(evaluation_index.size)),
            seed=seed)

    importance = out["importance"].copy()
    total = float(importance["relative_importance"].sum()) or 1.0
    importance["relative_importance"] = importance["relative_importance"] / total
    out["importance"] = importance.reset_index(drop=True)
    out["top_features"] = list(importance["feature"].head(10))
    return out


def importance_frame(results: dict[str, Any]) -> pd.DataFrame:
    """All per-model importances in one long table, for the results directory."""
    rows: list[pd.DataFrame] = []
    for model, entry in results.items():
        frame = entry.get("importance")
        if frame is None or frame.empty:
            continue
        value_column = ("importance" if "importance" in frame.columns
                        else "mean_mae_increase_w")
        part = frame[["feature", value_column]].copy()
        part.columns = ["feature", "value"]
        part["model"] = model
        part["method"] = entry.get("method")
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
