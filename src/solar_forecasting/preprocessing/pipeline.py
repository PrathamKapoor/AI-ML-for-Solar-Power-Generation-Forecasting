"""The data pipeline: seventeen documented stages from raw archive to model-ready data.

Stage order (each stage's output is recorded in the pipeline report):

===  ========================================================================
 1   data acquisition        archive downloaded, checksummed, members extracted
 2   raw-data validation     file presence, sizes, readability
 3   schema validation       required columns, dtypes, parseability
 4   timestamp normalization two archive timestamp conventions -> naive local
 5   duplicate detection     counts and resolution policy
 6   missing-value analysis  per-column profile, run lengths, gap lengths
 7   missing-value handling  split-aware interpolation, indicators retained
 8   outlier analysis        irradiance spikes, night offset, negative power
 9   daylight/night handling solar position, daylight mask, clear-sky reference
10   weather-feature processing 1-minute meteorology -> working resolution
11   solar-position features elevation, azimuth, sin/cos
12   temporal features       calendar, cyclical hour and day-of-year
13   target construction     aligned supervised target at each horizon
14   chronological split     strictly time-ordered, disjoint
15   scaling                 fitted on train only, applied unchanged elsewhere
16   sequence construction   sliding windows for sequence models
17   experiment-ready output  parquet/CSV artefacts plus the pipeline report
===  ========================================================================

The pipeline is idempotent and stateless: it reads the raw archive and writes
artefacts. Any state that must persist (fitted scalers, split boundaries) is
written to ``data/processed`` so a later stage can reload exactly what was used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, data_dir, results_dir
from ..data import loader as loader_mod
from ..data import schema as schema_mod
from ..data.metadata import lookup_station
from ..features.builder import (SOLAR_COLUMNS, WEATHER_COLUMNS, build_features,
                               describe_feature_groups, feature_columns)
from ..preprocessing import daylight as daylight_mod
from ..preprocessing import missing as missing_mod
from ..preprocessing import outliers as outliers_mod
from ..preprocessing import splitting as split_mod
from ..utils.io import write_json, write_table


class PipelineError(RuntimeError):
    """Raised when a pipeline stage cannot produce a usable dataset."""


def _log(stage: int, name: str, message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[stage {stage:>2}] {name:<28} {message}")


def aggregate_meteorology(met_1min: pd.DataFrame, frequency: str,
                         columns: list[str], how: str = "mean") -> pd.DataFrame:
    """Roll 1-minute meteorology up to the working resolution.

    The bin label is taken to be the bin's **end** timestamp, matching the PV
    series convention. A bin labelled 12:00 therefore covers 11:45-12:00 and
    contains no observation later than its own label, which is what makes the
    aggregated series safe to use as a trailing input.
    """
    indexed = met_1min.set_index("Time").sort_index()
    present = [c for c in columns if c in indexed.columns]
    if how == "mean":
        rolled = indexed[present].resample(frequency, label="right", closed="right").mean()
    elif how == "sum":
        rolled = indexed[present].resample(frequency, label="right", closed="right").sum()
    else:
        raise ValueError(f"unsupported aggregation: {how!r}")
    rolled = rolled.reset_index()
    rolled.columns = ["Time", *[f"{c}_agg" if c not in present else c for c in rolled.columns[1:]]]
    return rolled


def build_target(frame: pd.DataFrame, target_column: str, horizon_steps: int,
                 time_column: str = "Time") -> pd.DataFrame:
    """Construct the supervised target for one horizon.

    The target at row ``t`` is the observed power at ``t + horizon_steps``. Any
    row whose target falls beyond the end of the record is dropped, because its
    label does not exist.
    """
    out = frame.copy()
    out["target"] = out[target_column].shift(-horizon_steps)
    horizon_hours = horizon_steps * (out[time_column].diff().dt.total_seconds().median() / 3600.0)
    out["target_lead_steps"] = horizon_steps
    out["target_lead_hours"] = float(horizon_hours)
    out["target_time"] = out[time_column].shift(-horizon_steps)
    return out[out["target"].notna()].reset_index(drop=True)


def run_pipeline(config: Config, quiet: bool = False,
                 persist: bool = True, station: str | None = None,
                 artifact_slug: str | None = None) -> dict[str, Any]:
    """Execute the full pipeline and return the report plus the prepared frames.

    Parameters
    ----------
    station
        Station to process. Defaults to ``dataset.primary_station``. The
        cross-site experiment passes each site in turn, which is why the
        station is a parameter rather than a constant read from the
        configuration: the processing is identical for every site, and any
        difference between them is a property of the data.
    artifact_slug
        Suffix for the persisted artefacts, so that processing a second station
        cannot overwrite the primary station's files. Defaults to
        ``_<slugified station>`` when ``station`` is given.
    """
    report: dict[str, Any] = {
        "pipeline_version": "1.0.0",
        "config_fingerprint": config.fingerprint(),
        "config_sources": [p.name for p in config.sources],
        "stages": {},
    }
    validation = schema_mod.ValidationReport()

    dataset = config.section("dataset")
    primary = station or dataset["primary_station"]
    slug = artifact_slug or (None if station is None else _slugify(primary))
    report["station"] = primary
    frequency = dataset["working_resolution"]
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(frequency))
    # A humid subtropical site: Linke turbidity of 4.0 is a standard mid-range
    # value for coastal Hong Kong. The resulting absolute clear-sky level is not
    # trusted (see the scale calibration below); only the ratio is used.
    linke_turbidity = float(config.get("clear_sky", {}).get("linke_turbidity", 4.0))
    min_clear_reference = float(
        config.get("clear_sky", {}).get("min_clear_reference_wm2", 50.0))

    # ---------------- Stage 2/3: raw validation and schema -------------------- #
    raw_root = config.project_root_for("data") / "interim"
    if not (raw_root / dataset["metadata_file"]).exists():
        raise PipelineError(
            "raw archive not extracted. Run `python scripts/download_data.py` first.")

    _log(2, "raw-data validation", f"checking archive layout under data/interim", quiet)
    report["stages"]["02_raw_validation"] = {
        "archive_root": str(raw_root),
        "metadata_present": (raw_root / dataset["metadata_file"]).exists(),
    }

    # ---------------- Stage 4/5/9: load, normalise timestamps, solar --------- #
    _log(4, "timestamp normalization", f"loading {primary}", quiet)
    pv_raw = loader_mod.load_pv_station(config, primary, validation)
    report["stages"]["04_timestamps"] = {
        "station": primary,
        "n_rows": int(len(pv_raw)),
        "format_detected": "YYYY/M/D H:MM (meteorology) and YYYY-MM-DD HH:MM:SS (PV)",
        "timezone": "naive local standard time (UTC+08:00, no daylight saving in Hong Kong)",
        "first": str(pv_raw["Time"].min()),
        "last": str(pv_raw["Time"].max()),
    }
    _log(5, "duplicate detection", "counts recorded in the validation report", quiet)
    report["stages"]["05_duplicates"] = {
        "n_duplicate_timestamps": int(validation.issues and 0 or 0),
        "policy": "duplicates averaged within the same timestamp; the first label is kept",
    }

    _log(3, "schema validation", f"{len(validation.issues)} checks recorded", quiet)
    validation.raise_on_errors()
    report["stages"]["03_schema"] = validation.to_dict()

    stations_meta = loader_mod.load_station_metadata(config)
    station_info = loader_mod.describe_station(stations_meta, primary)
    rated_w = station_info.get("rated_w")
    latitude = station_info.get("latitude")
    longitude = station_info.get("longitude")
    if latitude is None or longitude is None:
        raise PipelineError(
            f"no coordinates in the Brick metadata for station {primary!r}; "
            f"solar position cannot be computed")

    schema_mod.check_physical_plausibility(
        pv_raw["pv_power_w"], rated_w, validation, "target/plausibility", primary)
    report["stages"]["02b_capacity"] = station_info
    report["stages"]["05_duplicates"]["n_duplicate_timestamps"] = sum(
        1 for i in validation.issues if i.check == "timestamps/duplicates")

    # ---------------- Stage 9: solar position and daylight ------------------- #
    _log(9, "daylight/night handling",
         f"computing solar position at ({latitude}, {longitude})", quiet)
    solar = daylight_mod.compute_solar_position(pv_raw["Time"], latitude, longitude)
    clear_sky = daylight_mod.clear_sky_irradiance(
        pv_raw["Time"], solar, latitude, longitude, linke_turbidity=linke_turbidity,
        altitude_m=station_info.get("altitude_m"))
    daylight = daylight_mod.daylight_mask(solar.elevation)
    report["stages"]["09_daylight"] = daylight_mod.solar_summary(solar, daylight)

    # ---------------- Stage 8: irradiance cleaning needs solar elevation ----- #
    met_1min = loader_mod.load_meteorology(config, validation)
    _log(10, "weather-feature processing",
         f"rolling {len(met_1min)} 1-minute rows to {frequency}", quiet)

    met_met = daylight_mod.compute_solar_position(met_1min["Time"], latitude, longitude)
    ghi_raw = met_1min["ghi"] if "ghi" in met_1min.columns else pd.Series(np.nan, index=met_1min.index)
    ghi_clean, irradiance_summary = outliers_mod.analyse_irradiance(
        ghi_raw, met_met.elevation, config.section("validation"), report)
    met_1min = met_1min.copy()
    met_1min["ghi"] = ghi_clean
    report["stages"]["08_outliers"] = {
        "irradiance": irradiance_summary,
    }

    met_agg = aggregate_meteorology(
        met_1min, frequency,
        [c for c in WEATHER_COLUMNS if c in met_1min.columns],
        how=config.section("features").get("meteorology_aggregation", "mean"))
    report["stages"]["10_weather"] = {
        "native_resolution": "1 minute",
        "working_resolution": frequency,
        "aggregation": config.section("features").get("meteorology_aggregation", "mean"),
        "bin_convention": "label = bin end timestamp; closed='right'",
        "variables": [c for c in met_agg.columns if c != "Time"],
        "n_rows": int(len(met_agg)),
        "n_missing_after_rollup": int(met_agg.isna().sum().sum()),
    }

    # ---------------- Merge PV with weather and solar ------------------------ #
    merged = pv_raw.merge(met_agg, on="Time", how="left", validate="one_to_one")
    missing_before_merge = int(merged["ghi"].isna().sum())
    solar_frame = solar.as_frame()
    merged = pd.concat([merged.reset_index(drop=True),
                        solar_frame.reset_index(drop=True)], axis=1)
    merged["daylight"] = daylight.astype(int).to_numpy()
    merged["is_daylight"] = merged["daylight"]

    # Clear-sky reference recomputed on the PV grid so indices align exactly.
    ghi_clear_pv = daylight_mod.clear_sky_irradiance(
        merged["Time"], daylight_mod.SolarFrame(
            elevation=solar.elevation.to_numpy(), azimuth=solar.azimuth.to_numpy(),
            sin_elevation=solar.sin_elevation.to_numpy(),
            cos_elevation=solar.cos_elevation.to_numpy(),
            zenith=solar.zenith.to_numpy()),
        latitude, longitude, linke_turbidity=linke_turbidity,
        altitude_m=station_info.get("altitude_m"))
    merged["ghi_clear"] = ghi_clear_pv.to_numpy()
    merged["clear_sky_index_raw"] = daylight_mod.clearness_index(
        merged["ghi"], merged["ghi_clear"], merged["daylight"].astype(bool),
        min_clear_reference=min_clear_reference)

    schema_mod.summarise_missing_timestamps(
        pd.DatetimeIndex(merged["Time"]), frequency, "merge/alignment",
        validation, "merged PV + weather")
    report["stages"]["10b_merge"] = {
        "n_rows_after_merge": int(len(merged)),
        "n_pv_rows_without_weather": missing_before_merge,
        "fraction_pv_rows_without_weather": float(missing_before_merge / max(len(merged), 1)),
        "policy": "weather gaps are filled in stage 7 by split-aware interpolation",
    }

    # ---------------- Stage 6/7: missing values ------------------------------ #
    _log(6, "missing-value analysis", "profiling before imputation", quiet)
    boundaries = split_mod.boundaries_from_config(config.section("split"))
    missing_report: dict[str, Any] = {}
    weather_cols = [c for c in met_agg.columns if c != "Time"]
    merged = missing_mod.impute_frame(
        merged, "pv_power_w", weather_cols, missing_report,
        train_end=boundaries.train_end)
    _log(7, "missing-value handling", missing_report["imputation_policy"]["target"], quiet)
    report["stages"]["06_07_missing"] = missing_report

    # Recompute solar and daylight after rows with unfillable targets were dropped.
    solar = daylight_mod.compute_solar_position(merged["Time"], latitude, longitude)
    solar_columns = solar.as_frame()
    for column in solar_columns.columns:
        merged[column] = solar_columns[column].to_numpy()
    merged["daylight"] = (merged["solar_elevation"] > 0.0).astype(int)
    merged["is_daylight"] = merged["daylight"]
    merged["ghi_clear"] = daylight_mod.clear_sky_irradiance(
        merged["Time"], solar, latitude, longitude,
        linke_turbidity=linke_turbidity,
        altitude_m=station_info.get("altitude_m")).to_numpy()
    merged["clear_sky_index_raw"] = daylight_mod.clearness_index(
        merged["ghi"], merged["ghi_clear"], merged["daylight"].astype(bool),
        min_clear_reference=min_clear_reference)

    # The clear-sky scale factor is estimated on the training split only and then
    # applied unchanged to validation and test, so the regime labels never carry
    # information from the evaluation period.
    _log(9, "clear-sky calibration", "estimating sensor scale on the training split", quiet)
    in_train = merged["Time"] <= boundaries.train_end
    scale_info = daylight_mod.estimate_clear_sky_scale(
        merged.loc[in_train, "clear_sky_index_raw"],
        merged.loc[in_train, "daylight"].astype(bool),
        quantile=float(config.get("clear_sky", {}).get("scale_quantile", 0.99)))
    merged["clear_sky_index"] = daylight_mod.apply_clear_sky_scale(
        merged["clear_sky_index_raw"], scale_info["scale"])
    report["stages"]["09b_clearsky_calibration"] = scale_info

    # ---------------- Stage 8b: target sanity ------------------------------- #
    merged["pv_power_w"], target_summary = outliers_mod.analyse_target(
        merged["pv_power_w"], rated_w, report)
    report["stages"]["08b_target"] = target_summary

    # ---------------- Stage 12: temporal features ---------------------------- #
    _log(11, "solar-position features", f"{len(SOLAR_COLUMNS)} columns", quiet)
    _log(12, "temporal features", "calendar and cyclical encodings", quiet)

    feature_config = config.section("features")
    featured = build_features(merged, feature_config, target_column="pv_power_w")
    feature_config_local = dict(feature_config)
    lookback = int(feature_config_local.get("lookback_steps", 96))

    # Rows lacking a full lookback or rolling window cannot form a sequence.
    required_history = max(
        [lookback] + list(feature_config_local.get("pv_lags", []))
        + list(feature_config_local.get("pv_rolling_windows", []))
    )
    before_warmup = len(featured)
    featured = featured.iloc[required_history:].reset_index(drop=True)
    report["stages"]["12b_warmup"] = {
        "lookback_steps": lookback,
        "max_history_steps_required": required_history,
        "n_rows_dropped_for_warmup": before_warmup - len(featured),
        "policy": "rows without a complete lookback window are dropped rather than "
                  "partially padded, which would bias early rows",
    }
    featured["season"] = pd.DatetimeIndex(featured["Time"]).map(
        lambda t: ("winter" if t.month in (12, 1, 2) else
                   "spring" if t.month in (3, 4, 5) else
                   "summer" if t.month in (6, 7, 8) else "autumn"))
    featured["year"] = pd.DatetimeIndex(featured["Time"]).year

    # Weather regimes, assigned from forecast-time irradiance only. They are an
    # error-stratification device and are excluded from the model feature set.
    from ..evaluation import regimes as regime_mod

    featured = regime_mod.attach_regimes(featured, config.section("regimes"))
    report["stages"]["09c_regimes"] = {
        "distribution_daylight": regime_mod.regime_distribution(featured["regime"], True),
        "distribution_all_steps": regime_mod.regime_distribution(featured["regime"], False),
        "rules": regime_mod.describe_regime_rules(
            config.section("regimes"), scale_info["scale"], linke_turbidity),
    }
    _log(9, "weather-regime labels",
         f"clear={(featured['regime'] == 'Clear').mean():.1%} of all steps, "
         f"variable={(featured['regime'] == 'High-variability').mean():.1%}", quiet)

    # ---------------- Stage 14: chronological split -------------------------- #
    _log(14, "chronological split",
         f"train<= {boundaries.train_end.date()}, val<= {boundaries.val_end.date()}, "
         f"test<= {boundaries.test_end.date()}", quiet)
    split = split_mod.chronological_split(featured, boundaries)
    report["stages"]["14_split"] = split.audit

    # ---------------- Stage 15: train-only scaling --------------------------- #
    _log(15, "scaling", "fitting scalers on the training split only", quiet)
    columns = feature_columns(featured)
    scalers = split_mod.fit_scalers(split.train, columns, "pv_power_w",
                                    train_end=boundaries.train_end)
    report["stages"]["15_scaling"] = {
        "n_features": len(columns),
        "feature_groups": describe_feature_groups(columns),
        "target_mean_w": float(scalers.target_scaler.mean_[0]),
        "target_scale_w": float(scalers.target_scaler.scale_[0]),
        "fitted_on": scalers.fitted_on,
        "leakage_guard": "scaler statistics computed from training rows only",
    }

    # ---------------- Stage 16: sequences ------------------------------------ #
    _log(16, "sequence construction", f"lookback={lookback} steps "
         f"({lookback * 15 / 60:.0f} h)", quiet)
    report["stages"]["16_sequences"] = {
        "lookback_steps": lookback,
        "lookback_hours": lookback * 15 / 60.0,
        "n_features": len(columns),
        "window_definition": "row t supplies features; the target is observed power "
                             f"at t + h, for h in {config.section('horizons')}",
        "stride": 1,
    }

    dataset_summary = build_dataset_summary(
        featured, split, config, station_info, rated_w, steps_per_day)
    report["stages"]["17_dataset"] = dataset_summary

    if persist:
        _log(17, "experiment-ready output", f"writing parquet artefacts ({primary})", quiet)
        persist_artifacts(config, featured, split, columns, report, slug=slug)

    report["validation"] = validation.to_dict()
    report["target"] = dataset_summary
    write_json(report, results_dir("metrics") / f"pipeline_report{slug or ''}.json")

    if not quiet:
        print(f"\n  rows: train={len(split.train)} val={len(split.val)} test={len(split.test)}")
        print(f"  features: {len(columns)}   target mean={scalers.target_scaler.mean_[0]:.0f} W")
    return {"report": report, "featured": featured, "split": split,
            "columns": columns, "scalers": scalers, "station": station_info}


def build_dataset_summary(featured: pd.DataFrame, split: split_mod.SplitResult,
                          config: Config, station_info: dict[str, Any],
                          rated_w: float | None, steps_per_day: int) -> dict[str, Any]:
    """Dataset statistics table content, including daylight-only distributions."""
    daylight = featured["daylight"].astype(bool)
    train = split.train
    train_daylight = train["daylight"].astype(bool)
    return {
        "station": station_info,
        "rated_w": rated_w,
        "frequency": config.section("dataset")["working_resolution"],
        "n_rows_total": int(len(featured)),
        "n_rows_daylight": int(daylight.sum()),
        "fraction_daylight": float(daylight.mean()),
        "date_range": [str(featured["Time"].min()), str(featured["Time"].max())],
        "columns_present": [c for c in featured.columns],
        "target_all": loader_mod.numeric_summary(featured["pv_power_w"]),
        "target_daylight": loader_mod.numeric_summary(featured.loc[daylight, "pv_power_w"]),
        "target_train_daylight": loader_mod.numeric_summary(
            train.loc[train_daylight, "pv_power_w"]),
        "ghi": loader_mod.numeric_summary(featured["ghi"]),
        "temperature": loader_mod.numeric_summary(featured["temperature"]),
        "relative_humidity": loader_mod.numeric_summary(featured["relative_humidity"]),
        "wind_speed": loader_mod.numeric_summary(featured["wind_speed"]),
        "clear_sky_index_daylight": loader_mod.numeric_summary(
            featured.loc[daylight, "clear_sky_index"]),
        "outlier_profiles": [
            outliers_mod.describe_outliers(featured[c], c)
            for c in ("pv_power_w", "ghi", "temperature", "relative_humidity", "wind_speed")
        ],
        "split_sizes": split.sizes(),
    }


def _slugify(name: str) -> str:
    """A filename-safe form of a station name.

    Station names contain spaces and parentheses ("S H Ho Sports Hall",
    "UG Hall2 RF"), so they are reduced to lowercase alphanumerics joined by
    underscores before being used in a path.
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def persist_artifacts(config: Config, featured: pd.DataFrame,
                      split: split_mod.SplitResult, columns: list[str],
                      report: dict[str, Any], slug: str | None = None) -> None:
    """Write the experiment-ready artefacts and the dataset-statistics table.

    ``slug`` namespaces the files for a non-primary station, so processing a
    second site cannot overwrite the primary site's split files.
    """
    processed = data_dir("processed")
    suffix = f"_{slug}" if slug else ""
    if slug:
        featured.to_parquet(processed / f"featured{suffix}.parquet", index=False)
    else:
        featured.to_parquet(processed / "featured_primary_station.parquet", index=False)
    for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
        part.to_parquet(processed / f"split_{name}{suffix}.parquet", index=False)
    if not slug:
        write_json({"feature_columns": columns,
                    "lookback_steps": int(config.section("features")["lookback_steps"])},
                   processed / "feature_columns.json")
    else:
        write_json({"feature_columns": columns,
                    "lookback_steps": int(config.section("features")["lookback_steps"]),
                    "station": report.get("station")},
                   processed / f"feature_columns{suffix}.json")

    summary = report["stages"]["17_dataset"]
    rows = []
    for scope, stats in (("all_rows", summary["target_all"]),
                         ("daylight_only", summary["target_daylight"]),
                         ("train_daylight", summary["target_train_daylight"])):
        for key, value in stats.items():
            rows.append({"scope": scope, "statistic": key, "value": value})
    for profile in summary["outlier_profiles"]:
        for key, value in profile.items():
            rows.append({"scope": f"outliers::{profile.get('variable')}",
                         "statistic": key, "value": value})
    write_table(pd.DataFrame(rows), results_dir("tables") /
                f"dataset_statistics{suffix or ''}.csv")


def main(config: Config | None = None, quiet: bool = False) -> dict[str, Any]:
    from ..config import load_config

    return run_pipeline(config or load_config(), quiet=quiet)


if __name__ == "__main__":  # pragma: no cover
    main()
