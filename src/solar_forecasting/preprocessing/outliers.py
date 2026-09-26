"""Outlier analysis and physically-motivated correction.

Two distinct problems are handled and reported separately:

1. **Instrument faults** in the irradiance series. The source pyranometer
   occasionally reports physically impossible values (observed maximum
   3608 W/m2 against a solar constant of 1361 W/m2). A rolling-median detector
   flags these, and they are replaced by the local rolling median.

2. **Night-time sensor offset.** The same sensor reports a non-zero irradiance
   at night (observed night-time median 5.3 W/m2, 95th percentile 31.2 W/m2).
   Values below a small threshold are set to zero *only when the solar elevation
   is at or below zero*, using the independently computed solar position, so a
   genuinely dim dawn reading is not destroyed.

Nothing here uses the PV target to decide what is an outlier, which would leak
label information into preprocessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def robust_bounds(series: pd.Series, window: int = 15, n_sigma: float = 6.0
                  ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling median and a robust dispersion band around it.

    A rolling median is used rather than a rolling mean so that a single spike
    does not inflate its own detection threshold.
    """
    med = series.rolling(window=window, center=True, min_periods=max(3, window // 3)).median()
    deviation = (series - med).abs()
    mad = deviation.rolling(window=window, center=True,
                            min_periods=max(3, window // 3)).median()
    upper = med + n_sigma * (1.4826 * mad)
    lower = med - n_sigma * (1.4826 * mad)
    return med, upper, lower


def analyse_irradiance(ghi: pd.Series, solar_elevation: pd.Series,
                       config_values: dict[str, Any],
                       report: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    """Clean the irradiance series and record what was changed.

    Returns the cleaned series and a summary of the corrections applied.
    """
    ghi = pd.to_numeric(ghi, errors="coerce").astype(float)
    original = ghi.copy()
    working = ghi.copy()

    plausible_max = float(config_values.get("ghi_plausible_max", 1200.0))
    plausible_min = float(config_values.get("ghi_plausible_min", 0.0))
    window = int(config_values.get("ghi_spike_window", 15))
    ratio = float(config_values.get("ghi_spike_ratio", 1.8))
    night_threshold = float(config_values.get("ghi_night_offset_threshold", 5.0))
    night_elevation = float(config_values.get("night_solar_elevation_max", 0.0))
    zero_below_horizon = bool(config_values.get("zero_irradiance_below_horizon", True))

    # --- Step 1: hard physical envelope -------------------------------------- #
    out_of_range = working.notna() & ((working < plausible_min) | (working > plausible_max))
    median, upper, lower = robust_bounds(working, window=window)
    n_envelope = int(out_of_range.sum())
    working = working.mask(out_of_range, median)

    # --- Step 2: local spike detection --------------------------------------- #
    spike = working.notna() & (upper.notna()) & (working > upper) & (working > ratio * median)
    n_spike = int(spike.sum())
    working = working.mask(spike, median)

    # --- Step 3: final hard envelope ----------------------------------------- #
    # Substitution by a rolling median can in principle raise a value above the
    # envelope (the local median may itself be high), so the physical limits are
    # re-applied once more as a final gate.
    residual_high = working.notna() & (working > plausible_max)
    n_residual_high = int(residual_high.sum())
    if n_residual_high:
        working = working.mask(residual_high, plausible_max)
    residual_low = working.notna() & (working < plausible_min)
    n_residual_low = int(residual_low.sum())
    if n_residual_low:
        working = working.mask(residual_low, plausible_min)

    # --- Step 3: night-time sensor offset ------------------------------------ #
    # The elevation argument may carry a different index from ``working`` (the
    # meteorological frame is date-indexed while the cleaned series is not), so
    # it is reduced to a positional array before the boolean masks are combined.
    elevation = np.asarray(getattr(solar_elevation, "to_numpy", lambda: solar_elevation)(),
                           dtype=float)
    if elevation.shape[0] != working.shape[0]:
        raise ValueError(
            f"solar elevation length {elevation.shape[0]} does not match irradiance "
            f"length {working.shape[0]}")
    below_horizon = pd.Series(elevation <= night_elevation, index=working.index)

    if zero_below_horizon:
        # Horizontal irradiance is zero by definition when the sun is below the
        # horizon, so any positive reading there is a zero-offset of the
        # instrument. The source pyranometer shows a night-time median of about
        # 5 W/m2 and a 95th percentile near 31 W/m2, which would otherwise be
        # interpreted as real dim light and would propagate into the clearness
        # index and the regime labels.
        night_offset = below_horizon & working.notna() & (working > 0.0)
    else:
        night_offset = (below_horizon & working.notna()
                        & (working > 0.0) & (working <= night_threshold))
    n_night = int(night_offset.sum())
    working = working.mask(night_offset, 0.0)

    summary = {
        "n_input": int(len(working)),
        "n_input_missing": int(original.isna().sum()),
        "n_outside_physical_envelope": n_envelope,
        "fraction_outside_physical_envelope": float(n_envelope / max(len(working), 1)),
        "physical_envelope": [plausible_min, plausible_max],
        "n_local_spikes": n_spike,
        "n_clipped_by_final_envelope": n_residual_high,
        "n_raised_by_final_envelope": n_residual_low,
        "n_night_offset_zeroed": n_night,
        "spike_window_steps": window,
        "spike_ratio_threshold": ratio,
        "night_offset_threshold_wm2": night_threshold,
        "zero_irradiance_below_horizon": zero_below_horizon,
        "night_solar_elevation_max_deg": night_elevation,
        "original_max": float(original.max()) if original.notna().any() else None,
        "cleaned_max": float(working.max()) if working.notna().any() else None,
        "original_mean": float(original.mean()) if original.notna().any() else None,
        "cleaned_mean": float(working.mean()) if working.notna().any() else None,
        "night_bias_before_wm2": (
            float(original[below_horizon].mean()) if below_horizon.any() else None),
        "night_bias_after_wm2": (
            float(working[below_horizon].mean()) if below_horizon.any() else None),
    }
    report["irradiance_cleaning"] = summary
    return working, summary


def analyse_target(power: pd.Series, rated_w: float | None,
                   report: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    """Sanity-check the PV target against its rated capacity.

    No replacement is performed: a high reading is far more likely to be real
    (the archive's maximum matches the rated capacity to within 0.5%) than an
    error, so the series is only *reported* on. Small negative values, which can
    arise from inverter measurement offsets, are clipped to zero because
    negative generation is not physically meaningful.
    """
    power = pd.to_numeric(power, errors="coerce").astype(float)
    negative = power < 0
    n_negative = int(negative.sum())
    cleaned = power.clip(lower=0.0)

    summary = {
        "n_negative_clipped": n_negative,
        "fraction_negative": float(n_negative / max(len(power), 1)),
        "observed_max_w": float(cleaned.max()) if cleaned.notna().any() else None,
        "rated_w": rated_w,
        "max_to_rated_ratio": (
            float(cleaned.max() / rated_w) if rated_w and cleaned.notna().any() else None),
        "policy": "negative values clipped to zero; no other target modification",
    }
    report["target_sanity"] = summary
    return cleaned, summary


def flag_flatlines(series: pd.Series, min_length: int = 8,
                   tolerance: float = 1e-9) -> pd.Series:
    """Flag runs of identical consecutive values.

    A stuck sensor or a frozen logger produces exactly repeated values. PV power
    is genuinely constant for long stretches at night, so this is reported as an
    ``info`` diagnostic and interpreted jointly with the daylight mask rather
    than being treated as a fault on its own.
    """
    values = series.to_numpy()
    same = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        same[1:] = np.abs(np.diff(values)) <= tolerance
    # A run is flagged only when it is interior, i.e. bounded by a change.
    flags = np.zeros(len(values), dtype=bool)
    start = None
    for i, flag in enumerate(list(same) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_length:
                flags[start:i] = True
            start = None
    return pd.Series(flags, index=series.index)


def describe_outliers(series: pd.Series, label: str) -> dict[str, Any]:
    """Distribution summary used by the dataset-statistics table."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"variable": label, "n": 0}
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {
        "variable": label,
        "n": int(len(values)),
        "min": float(values.min()),
        "q1": float(q1),
        "median": float(values.median()),
        "q3": float(q3),
        "max": float(values.max()),
        "iqr": float(iqr),
        "tukey_upper_fence": float(q3 + 1.5 * iqr),
        "tukey_lower_fence": float(q1 - 1.5 * iqr),
        "n_above_upper_fence": int((values > q3 + 1.5 * iqr).sum()),
        "n_below_lower_fence": int((values < q1 - 1.5 * iqr).sum()),
        "skewness": float(values.skew()),
        "kurtosis": float(values.kurtosis()),
    }
