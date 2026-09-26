"""Stratified error analysis: weather regime, season, time of day, power level.

Aggregate metrics hide the behaviour that decides whether a forecast is usable.
A model can hold a good average nRMSE and still be unusable at the morning ramp
or under broken cloud, so every stratified view here is reported next to the
aggregate it decomposes, together with the number of observations that support
it. A stratum with too few daylight steps is not summarised at all, because a
metric computed on a handful of points is noise presented as a result.

The stratifications are *post hoc*: they are computed from forecasts and
observations that both exist after the prediction was made. They are never
supplied to a model, and the regime labels themselves are documented in
:mod:`solar_forecasting.evaluation.regimes`.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_all

#: Below this many daylight observations a stratum is reported as under-sampled.
MIN_STRATUM_SIZE = 200

#: Time-of-day bands in solar time, used for the intraday error analysis.
TIME_OF_DAY_BANDS: list[tuple[str, float, float]] = [
    ("Early morning (05-08)", 5.0, 8.0),
    ("Morning ramp (08-10)", 8.0, 10.0),
    ("Mid-morning (10-12)", 10.0, 12.0),
    ("Solar noon (12-14)", 12.0, 14.0),
    ("Afternoon (14-16)", 14.0, 16.0),
    ("Evening ramp (16-18)", 16.0, 18.0),
    ("Late evening (18-20)", 18.0, 20.0),
]

#: Generation-level bands as a fraction of rated capacity.
POWER_LEVEL_BANDS: list[tuple[str, float, float]] = [
    ("0-10% of capacity", 0.0, 0.10),
    ("10-25% of capacity", 0.10, 0.25),
    ("25-50% of capacity", 0.25, 0.50),
    ("50-75% of capacity", 0.50, 0.75),
    ("75-100% of capacity", 0.75, 1.0001),
]

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def band_of(values: pd.Series, bands: Sequence[tuple[str, float, float]]) -> pd.Series:
    """Assign each value to the first band whose lower bound it reaches."""
    labels = pd.Series(pd.NA, index=values.index, dtype=object)
    for name, low, high in bands:
        mask = (values >= low) & (values < high)
        labels = labels.mask(labels.isna() & mask, name)
    return labels


def time_of_day_band(timestamps: Iterable[pd.Timestamp]) -> pd.Series:
    hours = pd.DatetimeIndex(pd.to_datetime(list(timestamps))).hour
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    labels = pd.Series(pd.NA, index=index, dtype=object)
    for name, low, high in TIME_OF_DAY_BANDS:
        mask = (index.hour >= low) & (index.hour < high)
        labels = labels.mask(labels.isna() & mask, name)
    return labels


def season_of(timestamps: Iterable[pd.Timestamp]) -> pd.Series:
    """Meteorological seasons of the Northern Hemisphere.

    Hong Kong is in the northern subtropics, where the conventional
    astronomical grouping is misleading: December to February is the cool and
    comparatively dry season, and the solar resource is at its annual minimum.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    month = index.month
    labels = pd.Series(pd.NA, index=index, dtype=object)
    for name, months in (("Winter", (12, 1, 2)), ("Spring", (3, 4, 5)),
                         ("Summer", (6, 7, 8)), ("Autumn", (9, 10, 11))):
        labels = labels.mask(labels.isna() & index.month.isin(months), name)
    return pd.Categorical(labels, categories=SEASON_ORDER, ordered=True)


def power_level_band(actual: np.ndarray, rated_w: float | None) -> pd.Series:
    values = pd.Series(actual, dtype=float)
    if not rated_w:
        return pd.Series("unscaled", index=values.index, dtype=object)
    return band_of(values / float(rated_w), POWER_LEVEL_BANDS)


def stratified_metrics(frame: pd.DataFrame, rated_w: float | None,
                       stratum: str, min_size: int = MIN_STRATUM_SIZE,
                       reference_column: str = "reference_persistence"
                       ) -> pd.DataFrame:
    """Metrics for every stratum, with the reference forecast scored on the same rows.

    Scoring the reference inside the stratum is what makes the skill column
    interpretable: a regime in which persistence is nearly perfect cannot hide
    behind a high absolute error.
    """
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby(stratum, observed=True):
        daylight = group[group["daylight"].astype(bool)]
        if len(daylight) < int(min_size):
            rows.append({
                "stratum": str(label), "scope": "daylight", "n": int(len(daylight)),
                "mae": np.nan, "rmse": np.nan, "nrmse_capacity": np.nan, "r2": np.nan,
                "smape": np.nan, "bias": np.nan,
                "persistence_rmse": np.nan, "skill_vs_persistence_rmse": np.nan,
                "under_sampled": True,
            })
            continue
        actual = daylight["actual"].to_numpy(dtype=float)
        predicted = daylight["predicted"].to_numpy(dtype=float)
        metrics = compute_all(actual, predicted, rated_w=rated_w)
        entry = {
            "stratum": str(label), "scope": "daylight", "n": int(len(daylight)),
            "mae": metrics["mae"], "rmse": metrics["rmse"],
            "nrmse_capacity": metrics["nrmse_capacity"], "r2": metrics["r2"],
            "smape": metrics["smape"], "bias": metrics["bias"],
            "under_sampled": False,
        }
        if reference_column in daylight.columns:
            reference = daylight[reference_column].to_numpy(dtype=float)
            if np.isfinite(reference).all():
                ref_rmse = float(np.sqrt(np.mean((actual - reference) ** 2)))
                entry["persistence_rmse"] = ref_rmse
                if ref_rmse > 0:
                    entry["skill_vs_persistence_rmse"] = float(1.0 - metrics["rmse"] / ref_rmse)
        rows.append(entry)
    result = pd.DataFrame(rows)
    if not result.empty and stratum == "regime":
        order = {name: i for i, name in enumerate(
            ["Clear", "Partly cloudy", "Cloudy", "High-variability", "Night"])}
        result = result.sort_values("stratum",
                                    key=lambda s: s.map(lambda v: order.get(v, 99)))
    return result.reset_index(drop=True)


def build_strata(predictions: pd.DataFrame, rated_w: float | None) -> pd.DataFrame:
    """Add the derived stratifications to a stored prediction frame.

    ``Time`` is the *target* timestamp, which is the time the forecast is about,
    so a morning-ramp stratum really does contain the ramp the model had to
    predict.
    """
    frame = predictions.copy()
    frame["Time"] = pd.to_datetime(frame["Time"])
    if "season" not in frame.columns or frame["season"].isna().all():
        frame["season"] = season_of(frame["Time"]).to_numpy()
    else:
        # The pipeline's own season labels are lower-case; they are normalised to
        # the ordering used here so that the table, the figure and the exported
        # categories agree.
        lookup = {name.lower(): name for name in SEASON_ORDER}
        frame["season"] = pd.Categorical(
            [lookup.get(str(value).strip().lower(), value) for value in frame["season"]],
            categories=SEASON_ORDER, ordered=True)
    if "time_of_day" not in frame.columns:
        frame["time_of_day"] = time_of_day_band(frame["Time"]).to_numpy()
    if "power_level" not in frame.columns:
        frame["power_level"] = power_level_band(
            frame["actual"].to_numpy(dtype=float), rated_w).to_numpy()
    frame["ramping"] = _ramp_flag(frame).to_numpy()
    return frame


def _ramp_flag(frame: pd.DataFrame, threshold_fraction: float = 0.10) -> pd.Series:
    """Flag steps in which the target moves by more than a fraction of capacity.

    Rapid change is the regime that matters operationally and it is not
    identifiable from clearness alone, so it is derived from the realised
    target: a step is "ramping" when the absolute change over the horizon
    exceeds the threshold. It is a labelling device used only for error
    stratification after the forecast has been made.
    """
    capacity = float(frame["actual"].abs().max()) or 1.0
    change = frame["actual"].diff().abs()
    return (change > threshold_fraction * capacity).fillna(False)


def stratified_table(predictions: pd.DataFrame, rated_w: float | None,
                     model: str, horizon: str, strata: Sequence[str] = ("regime", "season",
                                                                        "time_of_day",
                                                                        "power_level", "ramping")
                     ) -> pd.DataFrame:
    """Every requested stratification for one model, stacked into one table."""
    frame = build_strata(predictions, rated_w)
    frames = []
    for stratum in strata:
        if stratum not in frame.columns:
            continue
        table = stratified_metrics(frame, rated_w, stratum)
        table.insert(0, "stratum_type", stratum)
        table.insert(0, "model", model)
        table.insert(0, "horizon", horizon)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
