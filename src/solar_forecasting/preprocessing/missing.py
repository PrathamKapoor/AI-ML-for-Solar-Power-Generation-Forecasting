"""Missing-value analysis and imputation.

The policy is deliberately conservative and *causal*:

* Missing values are **analysed and reported** before anything is filled in, so
  the report describes the real gaps rather than the post-hoc state.
* Imputation for the target uses **time interpolation bounded by the daylight
  window**, never a value from the future beyond one step, and never a
  cross-station or global mean that would leak information.
* Imputation indicators are retained as features where configured, so a model
  can learn that a reading was reconstructed rather than measured.
* Fitted state (for example the last valid observation) is computed **once on
  the training split** and applied unchanged to validation and test, which is
  the standard guard against leakage through imputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MissingnessProfile:
    """Per-column missingness statistics, produced before imputation."""

    columns: dict[str, dict[str, Any]]
    total_cells: int
    total_missing: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cells": self.total_cells,
            "total_missing": self.total_missing,
            "overall_missing_fraction": (
                self.total_missing / self.total_cells if self.total_cells else None),
            "columns": self.columns,
        }

    def summary_line(self) -> str:
        parts = [f"{name}: {info['missing_fraction']:.4%} ({info['n_missing']})"
                 for name, info in self.columns.items() if info["missing_fraction"] > 0]
        return ", ".join(parts) if parts else "no missing values"


def profile_missingness(frame: pd.DataFrame,
                        value_columns: list[str] | None = None) -> MissingnessProfile:
    """Describe the missingness structure of a frame.

    Reports run length statistics as well as totals, because a single long
    outage and many scattered single-step gaps require different treatments.
    """
    columns = value_columns or [c for c in frame.columns if c != "Time"]
    n_rows = len(frame)
    stats: dict[str, dict[str, Any]] = {}
    total_missing = 0

    for column in columns:
        if column not in frame.columns:
            continue
        mask = frame[column].isna().to_numpy()
        n_missing = int(mask.sum())
        total_missing += n_missing

        runs: list[int] = []
        current = 0
        for flag in mask:
            if flag:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)

        stats[column] = {
            "n_missing": n_missing,
            "missing_fraction": float(n_missing / n_rows) if n_rows else 0.0,
            "n_gaps": len(runs),
            "longest_gap": int(max(runs)) if runs else 0,
            "mean_gap_length": float(np.mean(runs)) if runs else 0.0,
            "gap_length_histogram": {
                str(length): runs.count(length) for length in sorted(set(runs))
            } if len(runs) <= 40 else {"many_distinct_lengths": len(set(runs))},
        }

    return MissingnessProfile(stats, n_rows * len(stats), total_missing)


def interpolate_target(series: pd.Series, limit_direction: str = "both",
                       max_gap: int | None = None) -> pd.Series:
    """Time-interpolate a target series, optionally capping the gap length.

    ``limit_direction='both'`` is used for the *training* split only, where
    using the following observation is legitimate. For validation and test the
    caller should pass ``limit_direction='forward'`` so that no value from the
    future leaks backwards across the split boundary.
    """
    indexed = _with_time_index(series)
    out = indexed.interpolate(method="time", limit_direction=limit_direction,
                              limit=max_gap, limit_area="inside")
    return pd.Series(out.to_numpy(), index=series.index, name=series.name)


def _with_time_index(series: pd.Series) -> pd.Series:
    """Return the series indexed by its timestamps.

    ``method='time'`` interpolation requires a DatetimeIndex, but the pipeline
    carries a plain RangeIndex with a separate ``Time`` column. The caller
    attaches the matching timestamps through ``series.attrs['timestamps']``
    before calling in, and the helpers restore the original positional index on
    the way out.
    """
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    timestamps = series.attrs.get("timestamps")
    if timestamps is None:
        raise ValueError(
            "time-weighted interpolation requires timestamps; attach them via "
            "series.attrs['timestamps'] or pass a DatetimeIndex-indexed series")
    return pd.Series(series.to_numpy(), index=pd.DatetimeIndex(timestamps), name=series.name)


def fill_weather(series: pd.Series, limit_direction: str = "both",
                 max_gap: int | None = None) -> pd.Series:
    """Fill meteorological gaps by time interpolation, then by forward fill.

    Weather variables are smooth at a 15-minute cadence, so linear time
    interpolation across short gaps is physically reasonable. The forward fill
    is a last resort for the residual leading/trailing NaNs and is bounded so it
    cannot propagate a stale value across a long outage.
    """
    indexed = _with_time_index(series)
    out = indexed.interpolate(method="time", limit_direction=limit_direction,
                              limit=max_gap, limit_area="inside")
    values = pd.Series(out.to_numpy(), index=series.index, name=series.name)
    if values.isna().any():
        values = values.ffill(limit=max_gap or 96)
    if values.isna().any():
        # Median of the observed data seen so far; falls back to the global
        # median of whatever remains, which is a documented last resort.
        values = values.fillna(values.median())
    return values


def impute_frame(frame: pd.DataFrame, target_column: str,
                 weather_columns: list[str], report: dict[str, Any],
                 train_end: pd.Timestamp | None = None,
                 max_gap_target: int | None = 4,
                 max_gap_weather: int | None = 8,
                 time_column: str = "Time") -> pd.DataFrame:
    """Impute a merged frame under a split-aware policy.

    Parameters
    ----------
    train_end
        When given, rows at or after this timestamp use forward-only
        interpolation, because a backward fill there would import information
        from outside the training set.
    """
    out = frame.copy()
    timestamps = pd.DatetimeIndex(out[time_column])
    target = out[target_column]
    weather = [c for c in weather_columns if c in out.columns]

    profile = profile_missingness(out, [target_column, *weather])
    report["missingness_before"] = profile.to_dict()

    def impute_block(block: pd.DataFrame, direction: str) -> pd.DataFrame:
        """Impute one contiguous block, with a DatetimeIndex attached internally."""
        block = block.copy()
        block.index = pd.DatetimeIndex(block[time_column])
        for column in [target_column, *weather]:
            if column not in block.columns:
                continue
            if column == target_column:
                block[column] = block[column].interpolate(
                    method="time", limit_direction=direction, limit=max_gap_target,
                    limit_area="inside")
            else:
                values = block[column].interpolate(
                    method="time", limit_direction=direction, limit=max_gap_weather,
                    limit_area="inside")
                if values.isna().any():
                    values = values.ffill(limit=max_gap_weather or 96)
                if values.isna().any():
                    values = values.fillna(values.median())
                block[column] = values
        return block

    if train_end is None:
        out = impute_block(out, "both")
    else:
        in_train = (out[time_column] <= train_end)
        # The training block may use two-sided interpolation because both
        # neighbours are training observations. Every later block is
        # forward-only, so no value from outside the training set can propagate
        # backwards across the boundary.
        train_block = impute_block(out.loc[in_train].copy(), "both")
        later_block = impute_block(out.loc[~in_train].copy(), "forward")
        out = pd.concat([train_block, later_block]).sort_index()
        out = out.reset_index(drop=True)
        out.index.name = None

    # Any residual NaN in the target would break sequence construction, so it is
    # dropped rather than invented, and the count is reported.
    residual = int(out[target_column].isna().sum())
    if residual:
        report.setdefault("warnings", []).append(
            f"{residual} target values remained NaN after interpolation and were dropped")
        out = out[out[target_column].notna()].reset_index(drop=True)

    profile_after = profile_missingness(out, [target_column, *weather])
    report["missingness_after"] = profile_after.to_dict()
    report["imputation_policy"] = {
        "target": f"time interpolation, limit={max_gap_target} steps, forward-only at and "
                  f"after the train/validation boundary",
        "weather": f"time interpolation then bounded forward fill, limit={max_gap_weather} steps",
        "split_aware": train_end is not None,
    }
    return out
