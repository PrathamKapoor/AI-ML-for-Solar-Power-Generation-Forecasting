"""Schema validation for the raw PV and meteorological files.

Validation runs *before* any transformation so that a malformed input is
reported as a schema failure rather than silently producing NaNs downstream. The
report is written to ``results/metrics/data_quality_report.json`` and is
regenerated on every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_PV_TIME = "Time"
REQUIRED_MET_TIME = "Time"
MIN_PV_COLUMNS = 2


@dataclass
class ValidationIssue:
    severity: str          # "error" | "warning" | "info"
    stage: str
    check: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "stage": self.stage,
            "check": self.check,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, stage: str, check: str, message: str,
            **detail: Any) -> None:
        self.issues.append(ValidationIssue(severity, stage, check, message, detail))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def raise_on_errors(self) -> None:
        if self.errors:
            lines = [f"  [{i.stage}/{i.check}] {i.message}" for i in self.errors]
            raise ValueError("raw-data schema validation failed:\n" + "\n".join(lines))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "n_info": len([i for i in self.issues if i.severity == "info"]),
            "issues": [i.to_dict() for i in self.issues],
        }


def validate_pv_frame(df: pd.DataFrame, target_column: str, station: str,
                      report: ValidationReport) -> pd.DataFrame:
    """Validate a raw PV power frame: columns, dtypes, timestamps and values."""
    stage = f"schema/pv/{station}"

    if REQUIRED_PV_TIME not in df.columns:
        report.add("error", stage, "required_column",
                   f"missing required column {REQUIRED_PV_TIME!r}",
                   columns=list(df.columns))
        return df
    if target_column not in df.columns:
        report.add("error", stage, "required_column",
                   f"missing target column {target_column!r}", columns=list(df.columns))
        return df
    if len(df.columns) < MIN_PV_COLUMNS:
        report.add("error", stage, "min_columns",
                   f"expected at least {MIN_PV_COLUMNS} columns, found {len(df.columns)}")

    n_rows = len(df)
    if n_rows == 0:
        report.add("error", stage, "empty_frame", "PV frame contains no rows")
        return df

    unparsed = int(df[REQUIRED_PV_TIME].isna().sum())
    if unparsed:
        report.add("error", stage, "timestamp_parse",
                   f"{unparsed} timestamps could not be parsed",
                   fraction=unparsed / n_rows)

    power = pd.to_numeric(df[target_column], errors="coerce")
    non_numeric = int(power.isna().sum() - df[target_column].isna().sum())
    if non_numeric > 0:
        report.add("warning", stage, "non_numeric_values",
                   f"{non_numeric} values in {target_column!r} were non-numeric and became NaN",
                   fraction=non_numeric / n_rows)

    negative = int((power < 0).sum())
    if negative:
        report.add("warning", stage, "negative_power",
                   f"{negative} negative power readings", fraction=negative / n_rows)

    nan_fraction = float(power.isna().mean())
    report.add("info", stage, "missing_target",
               f"{nan_fraction:.4%} of target values are NaN",
               n_missing=int(power.isna().sum()), n_rows=n_rows)

    exact_zeros = float((power == 0).sum() / n_rows)
    report.add("info", stage, "zero_fraction",
               f"{exact_zeros:.2%} of readings are exactly zero (expected: night-time)",
               fraction=exact_zeros)

    return df


def validate_met_frame(df: pd.DataFrame, value_columns: dict[str, int],
                       report: ValidationReport, label: str = "meteorological") -> pd.DataFrame:
    """Validate a raw 1-minute meteorological frame."""
    stage = f"schema/{label}"

    if REQUIRED_MET_TIME not in df.columns:
        report.add("error", stage, "required_column",
                   f"missing required column {REQUIRED_MET_TIME!r}", columns=list(df.columns))
        return df
    if not value_columns:
        report.add("error", stage, "no_value_columns", "no meteorological value columns requested")
        return df

    n_rows = len(df)
    if n_rows == 0:
        report.add("error", stage, "empty_frame", "meteorological frame contains no rows")
        return df

    for name, position in value_columns.items():
        if position >= len(df.columns):
            report.add("error", stage, "column_position",
                       f"column position {position} for {name!r} is out of range",
                       n_columns=len(df.columns))
            continue
        series = pd.to_numeric(df.iloc[:, position], errors="coerce")
        nan_fraction = float(series.isna().mean())
        if nan_fraction > 0:
            report.add("info", stage, "missing_values",
                       f"{nan_fraction:.4%} of {name!r} values are NaN",
                       variable=name, fraction=nan_fraction)
        finite = series[np.isfinite(series)]
        if len(finite) == 0:
            report.add("error", stage, "all_invalid",
                       f"{name!r} contains no finite values", variable=name)
            continue
        report.add("info", stage, "range",
                   f"{name}: min={finite.min():.3f} median={finite.median():.3f} "
                   f"max={finite.max():.3f}",
                   variable=name, minimum=float(finite.min()),
                   median=float(finite.median()), maximum=float(finite.max()))
    return df


def check_expected_frequency(index: pd.DatetimeIndex, expected_seconds: int,
                             report: ValidationReport, stage: str,
                             label: str) -> None:
    """Compare observed timestamp spacing against the declared native resolution."""
    if len(index) < 3:
        return
    deltas = index.to_series().diff().dt.total_seconds().dropna()
    mode = float(deltas.mode().iloc[0])
    expected = float(expected_seconds)
    if abs(mode - expected) > 1e-6:
        report.add("warning", stage, "resolution_mismatch",
                   f"{label}: modal spacing {mode:.0f}s differs from declared {expected:.0f}s",
                   observed=mode, expected=expected)
    off_grid = int((deltas != expected).sum())
    if off_grid:
        report.add("warning", stage, "irregular_spacing",
                   f"{label}: {off_grid} of {len(deltas)} intervals are not exactly "
                   f"{expected:.0f}s", n_irregular=off_grid, n_total=len(deltas))


def check_physical_plausibility(power: pd.Series, rated_w: float | None,
                                report: ValidationReport, stage: str,
                                label: str) -> None:
    """Check that measured power is consistent with the declared rated capacity."""
    finite = power[np.isfinite(power)]
    if len(finite) == 0:
        report.add("error", stage, "all_invalid", f"{label}: no finite power values")
        return

    observed_max = float(finite.max())
    report.add("info", stage, "observed_max",
               f"{label}: observed maximum {observed_max / 1000:.2f} kW",
               observed_max_w=observed_max)

    if rated_w and rated_w > 0:
        ratio = observed_max / rated_w
        report.add("info", stage, "capacity_check",
                   f"{label}: observed maximum is {ratio:.3f} x rated capacity "
                   f"({rated_w / 1000:.2f} kW)", ratio=ratio)
        if ratio > 1.25:
            report.add("warning", stage, "capacity_exceeded",
                       f"{label}: observed maximum exceeds rated capacity by more than 25% "
                       f"(ratio {ratio:.3f})", ratio=ratio)
        elif ratio < 0.35:
            report.add("warning", stage, "capacity_undershoot",
                       f"{label}: observed maximum is below 35% of rated capacity "
                       f"(ratio {ratio:.3f}); the station may be under-reported in the archive",
                       ratio=ratio)
    else:
        report.add("info", stage, "capacity_check",
                   f"{label}: no rated capacity available, skipping capacity consistency check")


def summarise_missing_timestamps(index: pd.DatetimeIndex, expected_freq: str,
                                stage: str, report: ValidationReport,
                                label: str) -> dict[str, Any]:
    """Quantify gaps in an irregular index against a regular expected grid.

    Returns a dict that is also embedded in the data-quality report, so the
    missing-value analysis is a recorded result rather than a console message.
    """
    if len(index) == 0:
        return {"n_observed": 0, "n_expected": 0, "missing_fraction": None}

    full = pd.date_range(index.min(), index.max(), freq=expected_freq)
    missing = full.difference(index)
    info = {
        "n_observed": int(len(index)),
        "n_expected": int(len(full)),
        "n_missing": int(len(missing)),
        "missing_fraction": float(len(missing) / len(full)),
        "first_timestamp": str(index.min()),
        "last_timestamp": str(index.max()),
    }
    if len(missing):
        info["largest_gap_steps"] = int(
            max((b - a) // pd.Timedelta(expected_freq) for a, b in
                zip(missing[:-1], missing[1:])) if len(missing) > 1 else 1
        )
    report.add("info", stage, "missing_timestamps",
               f"{label}: {info['n_missing']} absent timestamps "
               f"({info['missing_fraction']:.4%} of the expected grid)",
               **info)
    return info
