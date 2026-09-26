"""Transparent weather-regime classification.

The taxonomy is defined in ``configs/data.yaml`` under ``regimes`` and
implemented here. Two quantities drive it, both of which exist at forecast time:

``kt``       calibrated clearness index, ``clip(GHI / (alpha * GHI_clear), 0, 1.2)``
``sigma_cv`` coefficient of variation of GHI over a trailing window

The classification is a pure function of the *forecast-time* irradiance record.
It is applied to the realised data only in order to group errors after
prediction, and is never passed to a model. Using the realised target-interval
irradiance (rather than a forecast of it) is a deliberate choice: it labels the
conditions the model actually had to predict, which is what an operator cares
about, and it cannot leak into the forecast because the labels are assigned
afterwards.

The rules, in order of precedence:

1. solar elevation <= 0                      -> Night
2. sigma_cv > variability_min                -> High-variability
3. kt >= clear.kt_min                        -> Clear
4. kt >= partly_cloudy.kt_min                -> Partly cloudy
5. otherwise                                 -> Cloudy

Rule 2 takes precedence over the clearness bands because short-term variability
is the property that determines operational difficulty, and a day can be
bright on average yet unusable for forecasting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REGIME_ORDER = ["Clear", "Partly cloudy", "Cloudy", "High-variability", "Night"]


def irradiance_variability(ghi: pd.Series, window: int = 16,
                           min_ghi: float = 20.0,
                           daylight: pd.Series | None = None) -> pd.Series:
    """Trailing coefficient of variation of daylight GHI.

    ``sigma_cv(t) = std / mean`` of the GHI values in the trailing window that
    ends at ``t`` and for which the sun is above the horizon, subject to a
    minimum irradiance.

    Two design choices matter:

    * **Daylight-only.** A 24-hour window necessarily contains about twelve hours
      of zeros, so a coefficient of variation computed over it is close to 1 for
      every step of every day and carries no information about cloudiness. Night
      samples are therefore excluded from both the mean and the standard
      deviation.
    * **Short window by default.** Variability that matters for short-term
      forecasting is the change over minutes to a couple of hours, which is the
      timescale at which passing cloud causes ramping. The window is configurable.

    Steps with fewer than ``max(4, window // 4)`` usable daylight samples, or a
    trailing mean irradiance at or below ``min_ghi``, are assigned NaN: the ratio
    is not meaningful there.
    """
    ghi = pd.to_numeric(ghi, errors="coerce").astype(float)
    min_periods = max(4, window // 4)

    if daylight is None:
        usable = ghi.where(ghi > float(min_ghi))
    else:
        usable = ghi.where((ghi > float(min_ghi)) & daylight.astype(bool))

    rolled_mean = usable.rolling(window=window, min_periods=min_periods).mean()
    rolled_std = usable.rolling(window=window, min_periods=min_periods).std(ddof=0)
    n_usable = usable.rolling(window=window, min_periods=1).count()

    cv = rolled_std / rolled_mean.where(rolled_mean > float(min_ghi))
    cv = cv.where(n_usable >= min_periods)
    return cv.replace([np.inf, -np.inf], np.nan)


def classify_regimes(kt: pd.Series, sigma_cv: pd.Series, solar_elevation: pd.Series,
                     regime_config: dict[str, Any]) -> pd.Series:
    """Assign a weather-regime label to each step.

    Returns a categorical Series using :data:`REGIME_ORDER`.
    """
    night_max = float(regime_config["night"]["solar_elevation_max"])
    variable_min = float(regime_config["variable"]["variability_min"])
    clear_min = float(regime_config["clear"]["kt_min"])
    partly_min = float(regime_config["partly_cloudy"]["kt_min"])

    is_night = solar_elevation <= night_max
    is_variable = (~is_night) & (sigma_cv > variable_min)
    is_clear = (~is_night) & (~is_variable) & (kt >= clear_min)
    is_partly = (~is_night) & (~is_variable) & (kt >= partly_min) & (kt < clear_min)
    # A daylight step with no usable kt (near-horizon or missing reference) and
    # no usable variability is treated as partly cloudy rather than dropped, so
    # that every daylight step receives a label.
    is_cloudy = (~is_night) & (~is_variable) & (~is_clear) & (~is_partly)

    labels = pd.Series("Cloudy", index=kt.index, dtype=object)
    labels[is_partly.to_numpy()] = "Partly cloudy"
    labels[is_clear.to_numpy()] = "Clear"
    labels[is_variable.to_numpy()] = "High-variability"
    labels[is_night.to_numpy()] = "Night"
    return pd.Categorical(labels, categories=REGIME_ORDER, ordered=True)


def attach_regimes(frame: pd.DataFrame, regime_config: dict[str, Any],
                   ghi_column: str = "ghi", kt_column: str = "clear_sky_index",
                   elevation_column: str = "solar_elevation") -> pd.DataFrame:
    """Add ``sigma_cv`` and ``regime`` columns to a feature frame."""
    out = frame.copy()
    window = int(regime_config.get("variability_window_steps", 16))
    min_ghi = float(regime_config.get("variability_min_ghi_wm2", 20.0))
    out["sigma_cv"] = irradiance_variability(
        out[ghi_column], window=window, min_ghi=min_ghi,
        daylight=out[elevation_column] > 0.0)
    out["regime"] = classify_regimes(out[kt_column], out["sigma_cv"],
                                     out[elevation_column], regime_config)
    return out


def regime_distribution(regimes: pd.Series, daylight_only: bool = True) -> dict[str, Any]:
    """Regime shares, used for the regime-comparison table and its documentation."""
    values = pd.Series(regimes).astype(object)
    if daylight_only:
        values = values[values != "Night"]
    total = len(values)
    counts = values.value_counts()
    return {
        "scope": "daylight" if daylight_only else "all_steps",
        "n": int(total),
        "counts": {str(k): int(v) for k, v in counts.items() if v > 0},
        "shares": {str(k): (round(float(v) / total, 6) if total else None)
                   for k, v in counts.items() if v > 0},
    }


def describe_regime_rules(regime_config: dict[str, Any], scale: float,
                          linke_turbidity: float) -> dict[str, Any]:
    """Machine-readable statement of the classification rule, for the report."""
    return {
        "definition": {
            "kt": "clip(GHI_obs / (alpha * GHI_clear(turbidity=%.2f)), 0, %.2f)"
                  % (linke_turbidity, regime_config.get("max_index", 1.2)),
            "alpha_clear_sky_scale": round(float(scale), 6),
            "alpha_fit_on": "training split only",
            "sigma_cv": ("coefficient of variation of GHI over a trailing window of "
                         f"{regime_config.get('variability_window_steps')} steps, "
                         f"undefined when the trailing mean is below "
                         f"{regime_config.get('variability_min_ghi_wm2')} W/m2"),
        },
        "precedence": [
            "solar_elevation <= %.1f deg -> Night" % regime_config["night"]["solar_elevation_max"],
            "sigma_cv > %.2f -> High-variability (overrides the kt bands)"
            % regime_config["variable"]["variability_min"],
            "kt >= %.2f and sigma_cv <= %.2f -> Clear"
            % (regime_config["clear"]["kt_min"], regime_config["clear"]["variability_max"]),
            "%.2f <= kt < %.2f and sigma_cv <= %.2f -> Partly cloudy"
            % (regime_config["partly_cloudy"]["kt_min"], regime_config["clear"]["kt_min"],
               regime_config["partly_cloudy"]["variability_max"]),
            "kt < %.2f -> Cloudy" % regime_config["partly_cloudy"]["kt_min"],
        ],
        "note": "Regime labels are assigned from the realised forecast-time irradiance "
                "record after prediction, for error stratification only. They are never "
                "supplied to any model.",
    }
