"""Load raw archive files into tidy, time-indexed frames.

Two products are produced:

``load_pv_station``  one station's AC power series at its native resolution
``load_meteorology`` the co-located 1-minute meteorological record

Both apply timestamp normalisation (Stage 4 of the pipeline) and duplicate
detection (Stage 5) but leave imputation, outlier handling and feature
construction to the preprocessing and feature modules, so each stage stays
independently testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from . import schema as schema_mod
from .metadata import lookup_station, parse_station_metadata

TIME_COLUMN = "Time"
# Timestamp formats present in the archive.
TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
                "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S.%f"]

SITE_LEVEL_OPTIMIZER = "PV stations with panel level optimizer/Site level dataset"
INVERTER_LEVEL = "PV stations with panel level optimizer/Inverter level dataset"
SITE_LEVEL_PLAIN = "PV stations without panel level optimizer/Site level dataset"

STATION_GROUPS = {
    SITE_LEVEL_OPTIMIZER: "site_level_optimizer",
    INVERTER_LEVEL: "inverter_level",
    SITE_LEVEL_PLAIN: "site_level_plain",
}


def parse_timestamps(series: pd.Series, report: schema_mod.ValidationReport | None = None,
                     label: str = "series") -> pd.DatetimeIndex:
    """Normalise a timestamp column to tz-naive ``datetime64[ns]``.

    The archive mixes ``YYYY/M/D H:MM`` (meteorology, 1-minute) and
    ``YYYY-MM-DD HH:MM:SS`` (PV, 15-minute and 5-minute) conventions, and
    meteorological files carry no seconds field. All formats are attempted in
    order, and anything still unparsed is reported rather than dropped silently.
    """
    raw = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = pd.Series(True, index=series.index)

    for fmt in TIME_FORMATS:
        if not remaining.any():
            break
        attempt = pd.to_datetime(raw.where(remaining), format=fmt, errors="coerce")
        filled = attempt.notna()
        parsed[filled] = attempt[filled]
        remaining = remaining & ~filled

    if remaining.any() and report is not None:
        report.add("error", "timestamps/parse",
                   f"{label}: {int(remaining.sum())} timestamps matched no known format",
                   n_unparsed=int(remaining.sum()),
                   examples=raw[remaining].head(5).tolist())

    index = pd.DatetimeIndex(parsed)
    if index.tz is not None:
        index = index.tz_convert(None) if index.tz else index.tz_localize(None)
    return index


def detect_and_report_duplicates(index: pd.DatetimeIndex,
                                 report: schema_mod.ValidationReport,
                                 label: str) -> int:
    """Count duplicate timestamps and record how they will be resolved."""
    n_duplicates = int(index.duplicated().sum())
    if n_duplicates:
        report.add("warning", "timestamps/duplicates",
                   f"{label}: {n_duplicates} duplicate timestamps; the first occurrence "
                   f"is kept and later duplicates averaged by the preprocessing stage",
                   n_duplicates=n_duplicates)
    return n_duplicates


def load_pv_station(config: Config, station: str,
                    report: schema_mod.ValidationReport,
                    group: str = SITE_LEVEL_OPTIMIZER) -> pd.DataFrame:
    """Load one station's PV power series.

    Parameters
    ----------
    group
        Which archive folder to read from. Defaults to the 15-minute site-level
        series, which offers the longest continuous coverage.
    """
    dataset = config.section("dataset")
    root = config.project_root_for("data") / "interim"
    path = root / dataset["pv_prefix"] / group / f"{station}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"station file not found: {path}\n"
            f"Run scripts/download_data.py first to extract the source archive.")

    raw = pd.read_csv(path)
    target = dataset["power_column_in_file"]
    schema_mod.validate_pv_frame(raw, target, station, report)

    frame = pd.DataFrame({TIME_COLUMN: parse_timestamps(raw[TIME_COLUMN], report, station)})
    frame[target] = pd.to_numeric(raw[target], errors="coerce")
    frame = frame.rename(columns={target: "pv_power_w"})

    detect_and_report_duplicates(pd.DatetimeIndex(frame[TIME_COLUMN]), report, station)
    frame = (frame.groupby(TIME_COLUMN, as_index=False)["pv_power_w"].mean()
                  .sort_values(TIME_COLUMN)
                  .reset_index(drop=True))
    frame.attrs["station"] = station
    frame.attrs["source_file"] = str(path.name)
    return frame


def load_all_stations(config: Config, stations: list[str],
                      report: schema_mod.ValidationReport,
                      group: str = SITE_LEVEL_OPTIMIZER) -> dict[str, pd.DataFrame]:
    """Load several stations, skipping (and recording) any that are unavailable."""
    out: dict[str, pd.DataFrame] = {}
    for station in stations:
        try:
            out[station] = load_pv_station(config, station, report, group=group)
        except FileNotFoundError as exc:
            report.add("warning", "loader/station_missing",
                       f"station {station!r} could not be loaded: {exc}")
    return out


def load_meteorology(config: Config, report: schema_mod.ValidationReport,
                    years: tuple[int, ...] = (2021, 2022, 2023)) -> pd.DataFrame:
    """Load and concatenate the 1-minute meteorological series.

    The archive stores one CSV per variable per year, and not every variable
    covers every year (rainfall begins in 2022). Columns absent from the archive
    are reported rather than fabricated, so downstream code can distinguish
    "missing variable" from "zero".
    """
    dataset = config.section("dataset")
    root = config.project_root_for("data") / "interim"
    met_root = root / dataset["met_prefix"]
    patterns = dataset["meteorological_variables"]
    positions = dataset["meteorological_value_columns"]

    parts: list[pd.DataFrame] = []
    present: set[str] = set()

    for variable, pattern in patterns.items():
        for year in years:
            path = met_root / pattern.format(year=year)
            if not path.exists():
                continue
            present.add(variable)
            position = positions.get(variable)
            if position is None:
                continue
            block = pd.read_csv(path)
            schema_mod.validate_met_frame(block, {variable: position}, report,
                                          label=f"met/{variable}/{year}")
            index = parse_timestamps(block[TIME_COLUMN], report, f"{variable}/{year}")
            values = pd.to_numeric(block.iloc[:, position], errors="coerce")
            parts.append(pd.DataFrame({"Time": index, variable: values.values}))

    if not parts:
        raise FileNotFoundError(
            f"no meteorological files found under {met_root}. "
            f"Run scripts/download_data.py first.")

    merged = pd.concat(parts, ignore_index=True)
    value_columns = [c for c in merged.columns if c != TIME_COLUMN]
    merged = (merged.groupby(TIME_COLUMN, as_index=False)[value_columns].mean()
                    .sort_values(TIME_COLUMN)
                    .reset_index(drop=True))
    merged.attrs["variables_present"] = sorted(present)

    absent = sorted(set(patterns) - present)
    if absent:
        report.add("warning", "loader/met_absent",
                   f"meteorological variables absent from the archive: {absent}")
    return merged


def load_station_metadata(config: Config) -> dict[str, dict[str, Any]]:
    """Parse the Brick metadata file and return per-station records."""
    dataset = config.section("dataset")
    path = config.project_root_for("data") / "interim" / dataset["metadata_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"metadata file not found: {path}. Run scripts/download_data.py first.")
    return parse_station_metadata(path)


def describe_station(stations: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    """Resolve a station's rated capacity, tolerating naming differences."""
    meta = lookup_station(stations, name)
    if meta is None:
        return {"station": name, "rated_w": None, "metadata_found": False}
    return {
        "station": name,
        "brick_name": meta.get("brick_name"),
        "location": meta.get("location"),
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "altitude_m": meta.get("altitude_m"),
        "rated_kw": meta.get("rated_kw"),
        "rated_w": meta.get("rated_w"),
        "connection_date": meta.get("connection_date"),
        "metadata_found": True,
    }


def reindex_to_grid(frame: pd.DataFrame, freq: str, value_columns: list[str],
                    start: str | pd.Timestamp | None = None,
                    end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Reindex onto a regular grid, leaving absent timestamps as NaN.

    This is the mechanism that makes missing-value analysis exact: gaps become
    explicit NaNs on a known grid instead of silently shortening the series.
    """
    index = pd.DatetimeIndex(frame[TIME_COLUMN])
    if start is None:
        start = index.min()
    if end is None:
        end = index.max()
    grid = pd.date_range(start=start, end=end, freq=freq)
    indexed = frame.set_index(TIME_COLUMN)[value_columns]
    out = indexed.reindex(grid)
    out.index.name = TIME_COLUMN
    return out.reset_index()


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    """Descriptive statistics used by the dataset-statistics table."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p01": float(values.quantile(0.01)),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
        "n_zero": int((values == 0).sum()),
        "fraction_zero": float((values == 0).mean()),
        "n_negative": int((values < 0).sum()),
    }
