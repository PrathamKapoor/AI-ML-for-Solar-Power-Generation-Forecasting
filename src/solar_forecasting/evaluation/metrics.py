"""Forecast evaluation metrics.

Every metric is implemented with its formula, its interpretation and its
limitations recorded alongside it, because a number without those three things
cannot be compared across studies. The metadata in :data:`METRIC_DOCUMENTATION`
is written into the results tables and the methodology document.

Design decisions specific to photovoltaic power:

* **MAPE is not computed.** The target is exactly zero for roughly half of all
  timestamps (every night-time step), so the percentage error is undefined at
  those points and unbounded near them. :func:`mape` therefore raises if called,
  pointing the caller at :func:`smape`.
* **sMAPE is the percentage metric of record.** It is bounded by 2 in absolute
  value, so a night-time step where both actual and forecast are zero
  contributes exactly zero rather than an arbitrary large value.
* **nRMSE is normalised by rated capacity** by default, which makes errors
  comparable across stations of different size in the cross-site experiment.
* **Daylight and night-time are always scored separately.** Aggregating them
  lets 50% exact zeros dominate the statistics and makes every model look
  better than it is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9

METRIC_DOCUMENTATION: dict[str, dict[str, str]] = {
    "mae": {
        "formula": "MAE = (1/n) * sum_i |y_i - f_i|",
        "units": "same as the target (W)",
        "interpretation": "Average absolute deviation between forecast and observation. "
                          "Robust to outliers relative to RMSE because errors are not squared.",
        "limitations": "Weights every timestamp equally, so night-time steps with zero "
                       "target and zero forecast pull the value down without carrying any "
                       "energy. Always report alongside a daylight-only figure.",
        "primary": "yes",
    },
    "rmse": {
        "formula": "RMSE = sqrt( (1/n) * sum_i (y_i - f_i)^2 )",
        "units": "same as the target (W)",
        "interpretation": "Root mean squared deviation. Penalises large errors more "
                          "heavily than MAE, which matters operationally because a large "
                          "miss during peak generation has a higher cost than several small "
                          "misses overnight.",
        "limitations": "Sensitive to outliers. Does not distinguish systematic bias from "
                       "variance, so it should be read together with the residual mean and a "
                       "residual distribution.",
        "primary": "yes",
    },
    "nrmse": {
        "formula": "nRMSE = RMSE / normaliser",
        "units": "dimensionless",
        "interpretation": "RMSE expressed as a fraction of a reference scale, so that sites "
                          "of different rated capacity can be compared directly.",
        "limitations": "Depends entirely on the choice of normaliser. Dividing by rated "
                       "capacity measures performance against nameplate and can look generous "
                       "for a site that never reaches nameplate; dividing by the mean observed "
                       "target measures performance against the realised energy. Both are "
                       "reported so the dependence is visible.",
        "primary": "yes",
    },
    "r2": {
        "formula": "R^2 = 1 - sum_i (y_i - f_i)^2 / sum_i (y_i - mean(y))^2",
        "units": "dimensionless",
        "interpretation": "Fraction of the observed variance explained by the forecast. "
                          "1 is a perfect forecast, 0 is no better than predicting the mean, "
                          "negative is worse than predicting the mean.",
        "limitations": "Inflated by any structure the target itself has. A PV power series "
                       "is strongly diurnal, so a model that learns only the time of day can "
                       "achieve a high R^2 while being unable to predict a single cloud "
                       "transition. R^2 must not be read as evidence of operational skill, "
                       "and it is not comparable across different evaluation windows.",
        "primary": "no",
    },
    "smape": {
        "formula": "sMAPE = (100/n) * sum_i 2|y_i - f_i| / (|y_i| + |f_i|), with the "
                   "0/0 term defined as 0",
        "units": "percent",
        "interpretation": "Symmetric percentage error, bounded by 200%. Scale-free, so it "
                          "can be compared across stations and it does not blow up where the "
                          "target approaches zero.",
        "limitations": "Still compresses to zero whenever both actual and forecast are zero, "
                       "so a night-time step contributes no information about whether the "
                       "model tracked the transition correctly. Reported for comparability "
                       "with the wider literature, not as the headline metric.",
        "primary": "yes",
    },
    "skill_vs_persistence": {
        "formula": "skill = 1 - RMSE_model / RMSE_persistence",
        "units": "fraction (0 to 1 for improvement)",
        "interpretation": "Fractional improvement in RMSE over the persistence forecast. A "
                          "skill of 0.2 means the model reduces RMSE by 20% relative to "
                          "carrying the most recent observation forward.",
        "limitations": "Only meaningful when computed on exactly the same set of "
                       "timestamps as the persistence forecast, and it inherits the "
                       "pathology of whichever base metric is used. Reported here against "
                       "RMSE and MAE.",
        "primary": "yes",
    },
    "mape": {
        "formula": "MAPE = (100/n) * sum_i |y_i - f_i| / |y_i|",
        "units": "percent",
        "interpretation": "Asymmetric percentage error.",
        "limitations": "UNDEFINED for this target. PV power is exactly zero at night, so "
                       "MAPE is undefined on roughly half the dataset and unbounded near "
                       "dawn and dusk, where it is dominated by timesteps carrying almost no "
                       "energy. It is therefore deliberately not computed; sMAPE is used "
                       "instead and the choice is reported.",
        "primary": "no",
    },
    "bias": {
        "formula": "bias = (1/n) * sum_i (f_i - y_i)",
        "units": "same as the target (W)",
        "interpretation": "Mean signed error. A non-zero value indicates systematic "
                          "over- or under-forecasting, which a purely absolute metric such "
                          "as MAE or RMSE cannot reveal.",
        "limitations": "A small mean bias can hide large cancelling errors, so it is a "
                       "diagnostic rather than a performance measure.",
        "primary": "no",
    },
    "peak_error": {
        "formula": "peak_error = max_i |f_i - y_i|",
        "units": "same as the target (W)",
        "interpretation": "Worst single-timestep absolute error, a direct proxy for the "
                          "reserve margin an operator must hold.",
        "limitations": "A single order statistic with high sampling variance; it should be "
                       "read as a bound rather than a central tendency.",
        "primary": "no",
    },
}


def _as_arrays(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
               ) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: y_true {a.shape} vs y_pred {b.shape}")
    if a.size == 0:
        raise ValueError("cannot compute a metric on an empty array")
    return a, b


def mae(y_true, y_pred) -> float:
    """Mean absolute error. See :data:`METRIC_DOCUMENTATION`['mae']."""
    a, b = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error. See :data:`METRIC_DOCUMENTATION`['rmse']."""
    a, b = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mse(y_true, y_pred) -> float:
    """Mean squared error, the quantity RMSE is the square root of."""
    a, b = _as_arrays(y_true, y_pred)
    return float(np.mean((a - b) ** 2))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error. Deliberately refuses to run.

    PV power is exactly zero outside daylight, which makes MAPE undefined. This
    function raises rather than silently substituting an epsilon, because a
    silently regularised MAPE produces a number that looks comparable with the
    literature while being driven by near-zero night-time timesteps.
    """
    raise NotImplementedError(
        "MAPE is undefined for a target that contains exact zeros, which is the case for "
        "PV power outside daylight hours. Use smape(), or report absolute errors "
        "(mae/rmse). See METRIC_DOCUMENTATION['mape'] for the full rationale.")


def smape(y_true, y_pred) -> float:
    """Symmetric mean absolute percentage error, in percent, bounded by 200.

    The 0/0 case (both actual and forecast exactly zero, i.e. night) is defined
    as zero, which is the conventional and the only numerically stable choice.
    """
    a, b = _as_arrays(y_true, y_pred)
    denominator = np.abs(a) + np.abs(b)
    numerator = 2.0 * np.abs(a - b)
    terms = np.where(denominator > EPS, numerator / np.where(denominator > EPS, denominator, 1.0),
                     0.0)
    return float(100.0 * np.mean(terms))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination. See :data:`METRIC_DOCUMENTATION`['r2']."""
    a, b = _as_arrays(y_true, y_pred)
    total = float(np.sum((a - a.mean()) ** 2))
    if total <= EPS:
        # A constant target has no variance to explain; R^2 is undefined, so the
        # neutral value of 0 is returned and the caller is expected to have
        # filtered constant windows out beforehand.
        return 0.0
    return float(1.0 - np.sum((a - b) ** 2) / total)


def nrmse(y_true, y_pred, normaliser: float | None = None,
          capacity_w: float | None = None) -> float:
    """RMSE normalised by a reference scale.

    ``normaliser`` takes precedence when given. Otherwise, when
    ``capacity_w`` is supplied the normaliser is the station's rated capacity;
    with neither, the mean of the observed target is used.
    """
    a, b = _as_arrays(y_true, y_pred)
    if normaliser is None:
        normaliser = capacity_w if capacity_w else float(np.mean(np.abs(a)))
    if normaliser <= EPS:
        raise ValueError(f"nRMSE normaliser must be positive, got {normaliser}")
    return float(rmse(a, b) / normaliser)


def bias(y_true, y_pred) -> float:
    """Mean signed error (forecast minus observed)."""
    a, b = _as_arrays(y_true, y_pred)
    return float(np.mean(b - a))


def peak_error(y_true, y_pred) -> float:
    """Largest single-timestep absolute error."""
    a, b = _as_arrays(y_true, y_pred)
    return float(np.max(np.abs(a - b)))


def skill(reference_metric_value: float, model_metric_value: float) -> float:
    """Fractional improvement of a model over a reference forecast.

    ``skill = 1 - model / reference``. A positive value is an improvement.
    """
    if reference_metric_value <= EPS:
        raise ValueError(
            f"reference metric must be positive to compute skill, got {reference_metric_value}")
    return float(1.0 - model_metric_value / reference_metric_value)


def compute_all(y_true, y_pred, rated_w: float | None = None,
                persistence_rmse: float | None = None,
                persistence_mae: float | None = None) -> dict[str, float]:
    """Compute the full metric set for one forecast.

    Parameters
    ----------
    persistence_rmse, persistence_mae
        Reference values from the persistence forecast on exactly the same
        timestamps. When supplied, skill scores are included.
    """
    a, b = _as_arrays(y_true, y_pred)
    out: dict[str, float] = {
        "mae": mae(a, b),
        "rmse": rmse(a, b),
        "mse": mse(a, b),
        "smape": smape(a, b),
        "r2": r2(a, b),
        "bias": bias(a, b),
        "peak_error": peak_error(a, b),
        "n": float(len(a)),
    }
    out["nrmse_capacity"] = nrmse(a, b, capacity_w=rated_w) if rated_w else float("nan")
    out["nrmse_mean"] = nrmse(a, b)

    if persistence_rmse is not None and persistence_rmse > EPS:
        out["skill_vs_persistence_rmse"] = skill(persistence_rmse, out["rmse"])
    if persistence_mae is not None and persistence_mae > EPS:
        out["skill_vs_persistence_mae"] = skill(persistence_mae, out["mae"])
    return out


def error_frame(y_true, y_pred, timestamps: pd.Series | None = None,
                index: pd.Index | None = None) -> pd.DataFrame:
    """Per-timestep error table used by the error analysis and figures."""
    a, b = _as_arrays(y_true, y_pred)
    frame = pd.DataFrame({"actual": a, "predicted": b,
                          "error": b - a, "abs_error": np.abs(b - a)})
    if timestamps is not None:
        frame.insert(0, "Time", pd.to_datetime(pd.Series(timestamps).to_numpy()))
    elif index is not None:
        frame.insert(0, "index", index)
    return frame.reset_index(drop=True)


def documentation_table() -> pd.DataFrame:
    """The metric documentation as a table, for the results and methodology."""
    rows = []
    for name, doc in METRIC_DOCUMENTATION.items():
        rows.append({"metric": name, **{k: v for k, v in doc.items()}})
    return pd.DataFrame(rows)
