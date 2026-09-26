"""Feature engineering for PV power forecasting.

Every feature falls into exactly one of three groups, and the group determines
what information it is allowed to use. This is the core leakage-control
mechanism of the project.

``historical``   Derived only from the target's own past values, or from
                 weather/solar variables observed strictly **before** the
                 forecast origin. Always admissible.

``concurrent``   Observed at the *target* timestamp (for example the air
                 temperature at the time being predicted). This is the
                 "nowcasting with local measurements" setting. It is legitimate
                 for horizons up to and including the measurement latency of the
                 sensor, and every experiment states which setting it uses.

``calendar``     Deterministic functions of the clock. Always admissible.

The default feature set is the **historical** setting, which is the strict one:
no model ever sees a weather value from the future relative to its forecast
origin. The concurrent setting is enabled explicitly by configuration so the
distinction is visible in the config and in the experiment records.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Columns that are bookkeeping rather than model inputs.
NON_FEATURE_COLUMNS = {
    "Time", "pv_power_w", "pv_power_w_target", "target", "target_lead_steps",
    "target_lead_hours", "target_time",
    "daylight", "regime", "season", "weather_regime", "clear_sky_index",
    "clear_sky_index_raw", "ghi_clear", "sigma_cv", "is_daylight",
    "year", "month", "day", "hour", "minute", "dayofyear", "dayofweek",
    "is_weekend", "hours_since_midnight", "wind_direction",
    "wind_direction_sin", "wind_direction_cos", "imputed_flag",
    # Collinear with solar elevation at this single site; see
    # SOLAR_COLUMNS_EXCLUDED_FROM_INPUT.
    "solar_azimuth", "cos_solar_elevation", "solar_zenith",
}

WEATHER_COLUMNS = [
    "ghi", "temperature", "relative_humidity", "sea_level_pressure",
    "wind_speed", "wind_direction", "visibility", "rainfall",
]

SOLAR_COLUMNS = [
    "solar_elevation", "solar_azimuth", "sin_solar_elevation",
    "cos_solar_elevation", "solar_zenith",
]

#: Solar-geometry columns that are collinear with solar elevation and are therefore
#: kept in the frame for inspection but excluded from the model inputs. Azimuth,
#: the cosine of elevation and the zenith angle are all deterministic functions
#: of elevation and time at this single site, so including them widens the
#: flattened representation without adding information.
SOLAR_COLUMNS_EXCLUDED_FROM_INPUT = [
    "solar_azimuth", "cos_solar_elevation", "solar_zenith",
]

CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

LAG_PREFIX = "pv_power_w_lag_"
ROLLING_PREFIX = "pv_rolling_"


def add_temporal_features(frame: pd.DataFrame, time_column: str = "Time") -> pd.DataFrame:
    """Add calendar and cyclical time features.

    Hours and day-of-year are encoded as sin/cos pairs so that the cyclical
    boundary (23:00 to 00:00, 31 December to 1 January) is not represented as a
    large artificial jump, which matters for day-ahead forecasting.
    """
    out = frame.copy()
    times = pd.DatetimeIndex(out[time_column])
    out["hour"] = times.hour
    out["minute"] = times.minute
    out["dayofyear"] = times.dayofyear
    out["year"] = times.year
    out["month"] = times.month
    out["day"] = times.day
    out["dayofweek"] = times.dayofweek
    out["is_weekend"] = (times.dayofweek >= 5).astype(int)

    # Hours since local midnight, fractional, for a smooth diurnal coordinate.
    hours_float = times.hour + times.minute / 60.0
    out["hours_since_midnight"] = hours_float

    out["hour_sin"] = np.sin(2 * np.pi * hours_float / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours_float / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * times.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * times.dayofyear / 365.25)
    return out


def add_solar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-computed solar-geometry columns to the feature frame."""
    out = frame.copy()
    for column in SOLAR_COLUMNS:
        if column in frame.columns:
            out[column] = frame[column].astype(float)
    if "solar_elevation" in out.columns:
        out["is_daylight"] = (out["solar_elevation"] > 0.0).astype(int)
    return out


def add_weather_features(frame: pd.DataFrame, concurrent: bool = False,
                         lag_steps: int = 1,
                         allowed: list[str] | None = None) -> pd.DataFrame:
    """Attach weather variables, optionally lagged to remove future information.

    With ``concurrent=False`` (the default, and the setting used for all
    headline results) each weather variable is shifted forward by ``lag_steps``
    so that the value attached to a forecast origin is the last one actually
    observed before that origin.

    ``allowed`` restricts which variables are used. The configuration lists the
    variables admitted by the experiment design; anything absent from that list
    is dropped rather than silently included. Rainfall, for example, is excluded
    by default because the source archive has no rainfall record for 2021, so
    including it would make the training period unrepresentative.
    """
    out = frame.copy()
    permitted = set(WEATHER_COLUMNS) if allowed is None else set(allowed)
    present = [c for c in WEATHER_COLUMNS if c in frame.columns and c in permitted]
    for column in present:
        if concurrent or lag_steps <= 1:
            out[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        else:
            out[column] = pd.to_numeric(frame[column], errors="coerce").shift(lag_steps).astype(float)
    for column in WEATHER_COLUMNS:
        if column not in permitted and column in out.columns:
            out = out.drop(columns=[column])
    if "wind_direction" in out.columns:
        radians = np.radians(out["wind_direction"].astype(float))
        out["wind_direction_sin"] = np.sin(radians)
        out["wind_direction_cos"] = np.cos(radians)
        out = out.drop(columns=["wind_direction"])
    return out


def add_target_lags(frame: pd.DataFrame, target_column: str,
                    lags: list[int]) -> pd.DataFrame:
    """Add trailing lags of the target.

    A lag of ``k`` means the value observed ``k`` steps before the current row.
    Because row ``t`` predicts row ``t + h``, the value at row ``t`` is known at
    forecast time, so every lag here is admissible under the strict historical
    setting.

    ``lag = 0`` is included explicitly as ``pv_power_w_current``. It is the value
    at the forecast origin itself, which is the single most informative input for
    short-horizon PV forecasting and is exactly the quantity the persistence
    baseline repeats. Omitting it and relying on ``lag_1`` alone would make every
    learned model start from a systematically stale state.
    """
    out = frame.copy()
    out[f"{LAG_PREFIX}0"] = out[target_column].astype(float)
    for lag in lags:
        if lag == 0:
            continue
        out[f"{LAG_PREFIX}{lag}"] = out[target_column].shift(lag)
    return out


def add_rolling_features(frame: pd.DataFrame, target_column: str,
                         windows: list[int], stats: list[str]) -> pd.DataFrame:
    """Add trailing rolling statistics of the target.

    ``min_periods`` equals the window, so early rows produce NaN rather than
    statistics computed from a partial window. Those rows are dropped later,
    which avoids the subtle bias of training on short-window estimates.
    """
    out = frame.copy()
    for window in windows:
        rolled = out[target_column].rolling(window=window, min_periods=window)
        for stat in stats:
            column = f"{ROLLING_PREFIX}{stat}_{window}"
            if stat == "mean":
                out[column] = rolled.mean()
            elif stat == "std":
                out[column] = rolled.std(ddof=0)
            elif stat == "max":
                out[column] = rolled.max()
            elif stat == "min":
                out[column] = rolled.min()
            else:
                raise ValueError(f"unsupported rolling statistic: {stat!r}")
    return out


def add_weather_rolling_features(frame: pd.DataFrame, columns: list[str],
                                 windows: list[int]) -> pd.DataFrame:
    """Add trailing rolling means of key weather variables.

    Cloudiness is a slow-moving process, so a short trailing mean of irradiance
    is a compact proxy for local cloud state at the forecast origin.
    """
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        for window in windows:
            out[f"{column}_rollmean_{window}"] = (
                out[column].rolling(window=window, min_periods=window).mean())
    return out


def add_irradiance_derivatives(frame: pd.DataFrame, column: str = "ghi") -> pd.DataFrame:
    """Add trailing rate-of-change and ramp features for irradiance.

    Ramps are the operationally important quantity: a fast irradiance change is
    what makes short-term PV forecasting hard, and a trailing ramp at the
    forecast origin is a strong indicator that it will continue. Both the
    difference and the per-step slope are included at two lags, so a model can
    distinguish a fresh ramp from one that began several steps ago.
    """
    out = frame.copy()
    if column not in out.columns:
        return out
    for lag in (1, 4):
        shifted = out[column].shift(lag)
        out[f"{column}_delta_{lag}"] = out[column] - shifted
        out[f"{column}_slope_{lag}"] = out[column].diff(lag) / lag
    return out


def add_daylight_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add daylight-mask-derived features.

    ``is_daylight`` is a deterministic function of the clock and site, so it is
    available at forecast time and legitimately usable. Predicting only during
    daylight and reporting daylight-only metrics keeps the night-time zeros from
    dominating every error statistic.
    """
    out = frame.copy()
    if "solar_elevation" in out.columns:
        out["daylight"] = (out["solar_elevation"] > 0.0).astype(int)
    return out


def add_clear_sky_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the clearness index when it is available.

    A low clearness index at the forecast origin means the site is currently
    under cloud, which is informative about the next few steps. Night-time values
    are zeroed rather than left as NaN so the feature is dense.
    """
    out = frame.copy()
    if "clear_sky_index" in frame.columns:
        kt = pd.to_numeric(frame["clear_sky_index"], errors="coerce")
        out["clear_sky_index"] = kt.fillna(0.0)
    return out


def build_features(frame: pd.DataFrame, config_features: dict[str, Any],
                   target_column: str = "pv_power_w",
                   concurrent_weather: bool = False) -> pd.DataFrame:
    """Assemble the full feature matrix in the documented order.

    Feature order is deterministic: weather, weather derivatives and rolling
    means, target lags, target rolling statistics, solar geometry, calendar.
    Deterministic ordering matters because SHAP and permutation importance
    results are reported by column position.
    """
    out = add_temporal_features(frame)
    out = add_solar_features(out)
    out = add_weather_features(out, concurrent=concurrent_weather,
                               allowed=config_features.get("weather_inputs"))
    out = add_clear_sky_features(out)
    out = add_daylight_features(out)

    out = add_target_lags(out, target_column, list(config_features.get("pv_lags", [])))
    out = add_rolling_features(
        out, target_column,
        list(config_features.get("pv_rolling_windows", [])),
        list(config_features.get("pv_rolling_stats", [])))

    allowed = set(config_features.get("weather_inputs") or [])
    if any(c in out.columns for c in allowed):
        out = add_weather_rolling_features(
            out, [c for c in ("ghi", "temperature") if c in allowed],
            list(config_features.get("weather_rolling_windows", [4, 24])))
    out = add_irradiance_derivatives(out, "ghi")
    return out


def feature_columns(frame: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Return the model-input columns of a feature frame in deterministic order."""
    excluded = NON_FEATURE_COLUMNS | set(exclude or set())
    columns = [c for c in frame.columns if c not in excluded]
    if not columns:
        raise ValueError("no feature columns remain after exclusion")
    return columns


def apply_feature_ablation(frame: pd.DataFrame, columns: list[str],
                           remove: list[str]) -> list[str]:
    """Remove a named group of features for the ablation study."""
    to_drop = [c for c in remove if c in columns]
    return [c for c in columns if c not in to_drop]


def describe_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Group the selected columns by provenance, for the feature-group table."""
    groups: dict[str, list[str]] = {
        "weather": [c for c in columns if c in WEATHER_COLUMNS],
        "weather_derived": [c for c in columns
                            if any(c.startswith(p) for p in
                                   ("ghi_", "temperature_")) and c not in WEATHER_COLUMNS],
        "target_lags": [c for c in columns if c.startswith(LAG_PREFIX)],
        "target_rolling": [c for c in columns if c.startswith(ROLLING_PREFIX)],
        "solar_geometry": [c for c in columns if c in SOLAR_COLUMNS],
        "calendar": [c for c in columns if c in CALENDAR_COLUMNS],
        "other": [c for c in columns
                  if c not in WEATHER_COLUMNS
                  and not c.startswith(LAG_PREFIX)
                  and not c.startswith(ROLLING_PREFIX)
                  and not any(c.startswith(p) for p in ("ghi_", "temperature_"))],
    }
    return {k: v for k, v in groups.items() if v}
