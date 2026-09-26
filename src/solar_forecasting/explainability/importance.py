"""Model explainability: SHAP for tree ensembles, permutation importance for
sequence models.

Two decisions are worth stating up front, because they are where this kind of
analysis is usually overstated.

**Permutation importance, not attention weights, for the recurrent and attention
models.** Attention maps are internal to the architecture and are routinely read
as explanations. They are not: a head may attend to a position for reasons that
have nothing to do with the value of that position in the output, and an
attribution that has not been validated against a perturbation is an
unvalidated claim. Permutation importance answers a question that can be
checked directly: how much does the forecast degrade if this variable is
scrambled? It is a *predictive* importance, not a causal one, and a feature can
be important for reasons that have nothing to do with a physical mechanism
(for example two collinear sensors, where permuting either one destroys the
information the other still supplies). No causal reading is offered anywhere in
this project.

**Tree features are collapsed over the lookback window.** The tree models see the
flattened window, so each original variable appears once per lag. Importance is
summed over the lags of a variable, because "does irradiance matter" is the
question, and "which of the 96 lagged irradiance values matters most" is an
artefact of the window length.

Both methods are computed on the test split using models refitted with the
declared configuration, and both are reported next to each other so that a
reader can see where they agree and where they do not.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..models.classical import flatten_windows


def collapse_over_lookback(importance: np.ndarray, feature_columns: Sequence[str],
                           lookback: int) -> pd.Series:
    """Sum per-lag importances into one value per original variable."""
    values = np.asarray(importance, dtype=float).ravel()
    n_features = len(feature_columns)
    if lookback > 1 and values.size == n_features * lookback:
        collapsed = values.reshape(lookback, n_features).sum(axis=0)
    else:
        collapsed = values
    series = pd.Series(collapsed, index=list(feature_columns)[:len(collapsed)])
    total = float(series.abs().sum()) or 1.0
    return series.abs() / total


def shap_importance(predictor, X: np.ndarray, feature_columns: Sequence[str],
                    lookback: int, n_samples: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Mean absolute SHAP value per original variable.

    ``X`` is the raw 3-D window tensor ``(n_samples, lookback, n_features)``; the
    tree models consume the flattened version, so the tensor is flattened here
    to match the estimator that was fitted.
    """
    import shap

    rng = np.random.default_rng(seed)
    n = min(int(n_samples), len(X))
    idx = rng.choice(len(X), size=n, replace=False) if n < len(X) else np.arange(len(X))
    subset = X[idx]
    flat = flatten_windows(subset) if subset.ndim == 3 else subset
    background = flat[: min(256, len(flat))]

    explainer = shap.TreeExplainer(predictor, data=background, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(flat)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    absolute = np.abs(values).mean(axis=0)
    return {
        "method": "SHAP TreeExplainer, mean |shap|",
        "n_samples": int(n),
        "n_background": int(len(background)),
        "importance": collapse_over_lookback(absolute, feature_columns, lookback),
    }


def permutation_importance_sequence(predict_fn: Callable[[np.ndarray], np.ndarray],
                                    X: np.ndarray, y_true: np.ndarray,
                                    feature_columns: Sequence[str],
                                    lookback: int, n_repeats: int = 10,
                                    seed: int = 42) -> dict[str, Any]:
    """Grouped permutation importance for a sequence model.

    The permutation is applied to a whole variable across the *entire* window of
    each selected sample, not to individual cells. Permuting cells independently
    would destroy the temporal structure of the input and measure the model's
    sensitivity to noise rather than to the removal of a variable.

    A deterministic generator is used and every repeat uses a different
    permutation, so the reported spread is the spread over permutations and not
    over seeds; the model's own initialisation is fixed upstream by the seed in
    the experiment record.
    """
    baseline = np.abs(np.asarray(predict_fn(X), dtype=float)
                      - np.asarray(y_true, dtype=float)).mean()
    n_features = len(feature_columns)
    rows: list[dict[str, Any]] = []
    for feature in range(n_features):
        deltas = []
        for repeat in range(int(n_repeats)):
            rng = np.random.default_rng(seed + 1000 * repeat + feature)
            permuted = X.copy()
            column = X[:, :, feature]
            order = rng.permutation(len(X))
            permuted[:, :, feature] = column[order]
            score = np.abs(np.asarray(predict_fn(permuted), dtype=float)
                           - np.asarray(y_true, dtype=float)).mean()
            deltas.append(float(score - baseline))
        deltas_arr = np.asarray(deltas)
        rows.append({
            "feature": feature_columns[feature],
            "baseline_mae_w": float(baseline),
            "mean_mae_increase_w": float(deltas_arr.mean()),
            "std_mae_increase_w": float(deltas_arr.std(ddof=1)) if deltas_arr.size > 1 else 0.0,
            "relative_increase": float(deltas_arr.mean() / baseline) if baseline > 0 else float("nan"),
            "n_repeats": int(n_repeats),
            "method": "grouped permutation over the full lookback window",
        })
    frame = pd.DataFrame(rows)
    total = float(frame["mean_mae_increase_w"].clip(lower=0).sum()) or 1.0
    frame["relative_importance"] = frame["mean_mae_increase_w"].clip(lower=0) / total
    return frame.sort_values("mean_mae_increase_w", ascending=False).reset_index(drop=True)


def import_feature_family(name: str) -> str:
    """Map a feature name to the input group used by the ablation study.

    The mapping reuses the groups declared in ``configs/data.yaml`` so that the
    explainability ranking and the ablation ranking are reported in the same
    vocabulary and can be read against each other.
    """
    if name.startswith("ghi") or name in {"temperature", "relative_humidity", "wind_speed",
                                         "clear_sky_index", "rainfall", "visibility",
                                         "sea_level_pressure"}:
        return "weather"
    if name.startswith("pv_") or "rolling" in name or "lag_" in name:
        return "pv_history"
    if name.startswith(("solar_", "sin_solar", "azimuth")):
        return "solar_geometry"
    if name.startswith(("hour_", "doy_", "day_of", "month_")):
        return "calendar"
    return "other"


def group_importance(frame: pd.DataFrame, value_column: str = "relative_importance"
                     ) -> pd.DataFrame:
    """Aggregate per-variable importance to the ablation's input groups."""
    if frame.empty:
        return frame
    work = frame.copy()
    work["feature_group"] = [import_feature_family(f) for f in work["feature"]]
    grouped = work.groupby("feature_group", observed=True).agg(
        total_importance=(value_column, "sum"),
        n_variables=("feature", "count"),
    ).reset_index()
    return grouped.sort_values("total_importance", ascending=False).reset_index(drop=True)
