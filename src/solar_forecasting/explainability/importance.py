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
    # The index is named so that ``reset_index()`` produces a column called
    # "feature", which is the column every importance table and figure expects.
    series.index.name = "feature"
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

    # No background dataset is passed. Supplying ``data=`` switches SHAP to the
    # interventional masker, which does not support gradient-boosted trees
    # containing categorical splits; ``tree_path_dependent`` is the supported mode
    # for this estimator and needs no background.
    explainer = shap.TreeExplainer(predictor, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(flat)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    absolute = np.abs(values).mean(axis=0)
    return {
        "method": "SHAP TreeExplainer, mean |shap|, tree_path_dependent",
        "n_samples": int(n),
        "n_background": 0,
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


def permutation_importance_by_regime(predict_fn: Callable[[np.ndarray], np.ndarray],
                                     X: np.ndarray, y_true: np.ndarray,
                                     feature_columns: Sequence[str], lookback: int,
                                     regimes: np.ndarray, n_repeats: int = 5,
                                     seed: int = 42, min_samples: int = 500
                                     ) -> pd.DataFrame:
    """Grouped permutation importance computed separately inside each regime.

    A variable can matter for clear-sky tracking and be irrelevant under broken
    cloud, and a single global ranking hides that. Each regime is perturbed and
    scored on its own rows only, so a variable whose apparent importance comes
    from one regime cannot be credited with another's performance.

    Regimes with fewer than ``min_samples`` windows are skipped rather than
    summarised, because a permutation measured on a handful of windows is noise.
    """
    regimes = np.asarray(regimes, dtype=object)
    rows: list[dict[str, Any]] = []
    unique_regimes = [r for r in ("Clear", "Partly cloudy", "Cloudy", "High-variability")
                      if r in set(regimes)]
    for regime in unique_regimes:
        index = np.flatnonzero(regimes == regime)
        if index.size < int(min_samples):
            continue
        subset_X = X[index]
        subset_y = np.asarray(y_true, dtype=float)[index]
        baseline = float(np.abs(np.asarray(predict_fn(subset_X), dtype=float)
                                - subset_y).mean())
        for feature in range(len(feature_columns)):
            deltas = []
            for repeat in range(int(n_repeats)):
                rng = np.random.default_rng(seed + 1000 * repeat + 97 * feature
                                            + int(index[0]))
                permuted = subset_X.copy()
                permuted[:, :, feature] = subset_X[rng.permutation(index.size), :, feature]
                score = float(np.abs(np.asarray(predict_fn(permuted), dtype=float)
                                     - subset_y).mean())
                deltas.append(score - baseline)
            rows.append({
                "regime": str(regime),
                "feature": feature_columns[feature],
                "baseline_mae_w": baseline,
                "mean_mae_increase_w": float(np.mean(deltas)),
                "std_mae_increase_w": float(np.std(deltas, ddof=1))
                if len(deltas) > 1 else 0.0,
                "n_windows": int(index.size),
                "n_repeats": int(n_repeats),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["regime", "mean_mae_increase_w"],
                             ascending=[True, False]).reset_index(drop=True)


def dependence_frame(predict_fn: Callable[[np.ndarray], np.ndarray],
                      X: np.ndarray, y_true: np.ndarray, feature_columns: Sequence[str],
                      lookback: int, features: Sequence[str],
                      n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Per-sample attribution rows for a chosen set of input variables.

    Returns one row per sample with the value of each requested variable at the
    forecast origin and the model's signed error, which is what the dependence
    plots are drawn from. Only the origin step is used, because that is the value
    a forecaster actually holds at decision time.
    """
    rng = np.random.default_rng(seed)
    n = min(int(n_samples), len(X))
    index = rng.choice(len(X), size=n, replace=False) if n < len(X) else np.arange(len(X))
    predicted = np.asarray(predict_fn(X[index]), dtype=float)
    actual = np.asarray(y_true, dtype=float)[index]
    origin = X[index][:, -1, :]
    frame = pd.DataFrame({"actual": actual, "predicted": predicted,
                          "error": predicted - actual,
                          "abs_error": np.abs(predicted - actual)})
    for feature in features:
        if feature not in feature_columns:
            continue
        frame[feature] = origin[:, list(feature_columns).index(feature)]
    return frame


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
