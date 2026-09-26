"""Distribution-free prediction intervals by split conformal prediction.

A deterministic model answers "what will the power be". An operator also needs
to know how much to trust that number, and a well-calibrated interval is worth
more than a small improvement in point accuracy. Split conformal prediction is
used here because it needs no distributional assumption about photovoltaic
power, returns finite-sample marginal coverage, and costs one extra pass over a
calibration set.

**The calibration/evaluation split, stated plainly.** A conformal interval is
valid under exchangeability between calibration and evaluation residuals. The
honest options are (a) calibrate on the validation split, which requires refitting
each model and re-predicting the validation period, or (b) calibrate on an
initial block of the test period and evaluate on the remainder. This project
implements (b) so that the intervals can be derived from the stored test
predictions without refitting, and the cost of that choice is recorded: the
coverage guarantee is then *approximate*, because the calibration block is
earlier in the year than the evaluation block, so the two are not exchangeable
under seasonal drift. The coverage figures reported here are therefore empirical
and should be read as a description of interval behaviour, not as a proof of
validity. The limitation is carried in the output record and in the paper.

Intervals are symmetric and unadapted (``y_hat +/- q``), which is the standard
split-conformal form. Asymmetric or locally-weighted conformal intervals are
listed as future work rather than implemented and left unverified.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CALIBRATION_DAYS = 31


def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """The finite-sample split-conformal quantile.

    Uses the ``ceil((n + 1) * (1 - alpha)) / n`` empirical quantile required for
    the coverage guarantee, with the result clipped at the observed maximum when
    the required rank exceeds ``n``.
    """
    residuals = np.abs(np.asarray(abs_residuals, dtype=float).ravel())
    residuals = residuals[np.isfinite(residuals)]
    n = residuals.size
    if n == 0:
        raise ValueError("no finite residuals to calibrate on")
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    if rank > n:
        return float(residuals.max())
    return float(np.sort(residuals)[rank - 1])


def split_calibrate(predictions: pd.DataFrame, alpha: float = 0.1,
                    calibration_days: int = DEFAULT_CALIBRATION_DAYS) -> dict[str, Any]:
    """Split a stored prediction frame into a calibration block and an evaluation block."""
    frame = predictions.copy()
    frame["Time"] = pd.to_datetime(frame["Time"])
    frame = frame.sort_values("Time")
    start = frame["Time"].min()
    cutoff = start + pd.Timedelta(days=int(calibration_days))
    calibration = frame[frame["Time"] < cutoff]
    evaluation = frame[frame["Time"] >= cutoff]
    if len(calibration) < 100 or len(evaluation) < 100:
        raise ValueError(
            f"conformal split is too small: {len(calibration)} calibration and "
            f"{len(evaluation)} evaluation steps after a {calibration_days}-day "
            f"calibration block; increase calibration_days or use fewer models")
    return {"calibration": calibration, "evaluation": evaluation,
            "calibration_end": cutoff, "calibration_days": int(calibration_days)}


def coverage_table(predictions: pd.DataFrame, model: str, horizon: str,
                   alphas: tuple[float, ...] = (0.1, 0.2),
                   calibration_days: int = DEFAULT_CALIBRATION_DAYS) -> list[dict[str, Any]]:
    """Empirical coverage and interval width for one model at several levels."""
    parts = split_calibrate(predictions, calibration_days=calibration_days)
    calibration, evaluation = parts["calibration"], parts["evaluation"]
    cal_errors = np.abs(calibration["predicted"].to_numpy(dtype=float)
                        - calibration["actual"].to_numpy(dtype=float))
    actual = evaluation["actual"].to_numpy(dtype=float)
    predicted = evaluation["predicted"].to_numpy(dtype=float)
    daylight = evaluation["daylight"].to_numpy(dtype=bool)

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        q = conformal_quantile(cal_errors, alpha)
        lower, upper = predicted - q, predicted + q
        inside = (actual >= lower) & (actual <= upper)
        for scope, mask in (("evaluation_all_steps", np.ones_like(inside, dtype=bool)),
                            ("evaluation_daylight", daylight)):
            rows.append({
                "model": model,
                "horizon": horizon,
                "alpha": float(alpha),
                "nominal_coverage": float(1.0 - alpha),
                "scope": scope,
                "interval_half_width_w": float(q),
                "empirical_coverage": float(inside[mask].mean()),
                "coverage_error": float(inside[mask].mean() - (1.0 - alpha)),
                "n": int(mask.sum()),
                "n_calibration": int(len(calibration)),
                "n_calibration_daylight": int(calibration["daylight"].astype(bool).sum()),
                "calibration_window_days": int(calibration_days),
                "method": "split conformal, symmetric unadapted interval",
                "guarantee": "approximate only: the calibration block precedes the "
                             "evaluation block, so the two are not exchangeable "
                             "under seasonal drift",
            })
    return rows


def interval_frame(predictions: pd.DataFrame, alpha: float = 0.1,
                   calibration_days: int = DEFAULT_CALIBRATION_DAYS) -> pd.DataFrame:
    """Per-step lower/upper bounds, for plotting."""
    parts = split_calibrate(predictions, calibration_days=calibration_days)
    calibration, evaluation = parts["calibration"], parts["evaluation"]
    cal_errors = np.abs(calibration["predicted"].to_numpy(dtype=float)
                        - calibration["actual"].to_numpy(dtype=float))
    q = conformal_quantile(cal_errors, alpha)
    out = evaluation[["Time", "actual", "predicted"]].copy()
    out["lower"] = out["predicted"] - q
    out["upper"] = out["predicted"] + q
    out["inside"] = (out["actual"] >= out["lower"]) & (out["actual"] <= out["upper"])
    out["interval_half_width_w"] = q
    return out.reset_index(drop=True)
