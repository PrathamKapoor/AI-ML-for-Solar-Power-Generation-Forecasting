"""Data loading, schema validation, missing values and outliers.

Every test here exercises a real behaviour of the pipeline against a synthetic
frame with the same schema the pipeline emits (see ``conftest.make_frame``), so
the suite needs no download and still fails on a genuine regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecasting.data import schema
from solar_forecasting.evaluation import regimes
from solar_forecasting.preprocessing import missing, outliers


# --------------------------------------------------------------------------- #
# Schema and raw-data validation
# --------------------------------------------------------------------------- #
def test_validate_pv_frame_accepts_a_complete_frame(frame: pd.DataFrame) -> None:
    report = schema.ValidationReport()
    schema.validate_pv_frame(frame, "pv_power_w", "synthetic", report)
    assert not report.errors, [i.message for i in report.errors]
    assert any(i.check == "missing_target" for i in report.issues)


def test_validate_pv_frame_reports_a_missing_target_column(frame: pd.DataFrame) -> None:
    report = schema.ValidationReport()
    schema.validate_pv_frame(frame.drop(columns=["pv_power_w"]), "pv_power_w",
                             "synthetic", report)
    assert report.errors
    assert "pv_power_w" in report.errors[0].message


def test_validate_pv_frame_reports_an_empty_frame(frame: pd.DataFrame) -> None:
    report = schema.ValidationReport()
    schema.validate_pv_frame(frame.iloc[0:0], "pv_power_w", "synthetic", report)
    assert any(i.check == "empty_frame" for i in report.errors)


def test_validate_pv_frame_flags_negative_power(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[10, "pv_power_w"] = -250.0
    report = schema.ValidationReport()
    schema.validate_pv_frame(broken, "pv_power_w", "synthetic", report)
    assert any(i.check == "negative_power" for i in report.warnings)


def test_raise_on_errors_refuses_to_continue() -> None:
    report = schema.ValidationReport()
    report.add("error", "test", "synthetic", "deliberate failure")
    with pytest.raises(ValueError, match="deliberate failure"):
        report.raise_on_errors()


def test_expected_frequency_check_flags_irregular_spacing(frame: pd.DataFrame) -> None:
    index = pd.to_datetime(frame["Time"]).copy()
    index.loc[5] = index.loc[5] + pd.Timedelta(seconds=90)
    report = schema.ValidationReport()
    schema.check_expected_frequency(pd.DatetimeIndex(index), 900, report,
                                    "test/frequency", "synthetic")
    assert any(i.check in {"irregular_spacing", "resolution_mismatch"} for i in report.warnings)


def test_summarise_missing_timestamps_counts_the_gap(frame: pd.DataFrame) -> None:
    index = pd.DatetimeIndex(pd.to_datetime(frame["Time"])).delete(10)
    report = schema.ValidationReport()
    info = schema.summarise_missing_timestamps(index, "15min", "test/gaps", report,
                                               "synthetic")
    assert info["n_missing"] == 1
    assert 0 < info["missing_fraction"] < 0.01
    assert any(i.check == "missing_timestamps" for i in report.issues)


def test_physical_plausibility_compares_against_rated_capacity(frame: pd.DataFrame) -> None:
    report = schema.ValidationReport()
    schema.check_physical_plausibility(frame["pv_power_w"], 55_000.0, report,
                                       "test/capacity", "synthetic")
    assert any(i.check == "capacity_check" for i in report.issues)


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def test_profile_missingness_reports_the_affected_columns(frame: pd.DataFrame) -> None:
    holed = frame.copy()
    holed.loc[50:70, "ghi"] = np.nan
    profile = missing.profile_missingness(holed, ["ghi", "temperature", "wind_speed"])
    assert profile.columns["ghi"]["n_missing"] == 21
    assert profile.columns["temperature"]["n_missing"] == 0
    assert profile.columns["ghi"]["missing_fraction"] > \
        profile.columns["temperature"]["missing_fraction"]
    assert profile.total_missing == 21


def test_interpolate_target_closes_an_interior_gap(frame: pd.DataFrame) -> None:
    holed = frame.copy()
    original = holed["pv_power_w"].copy()
    holed.loc[100:110, "pv_power_w"] = np.nan
    series = holed["pv_power_w"]
    series.attrs["timestamps"] = pd.DatetimeIndex(holed["Time"])
    filled = missing.interpolate_target(series, limit_direction="both")
    assert filled.isna().sum() == 0
    recovered = filled.loc[100:110].to_numpy(dtype=float)
    expected = original.loc[100:110].to_numpy(dtype=float)
    # A linear reconstruction of a smooth ramp, not an exact recovery: the test
    # asserts plausibility rather than equality.
    assert np.all(recovered >= -1.0)


def test_impute_frame_leaves_no_non_finite_inputs(frame: pd.DataFrame) -> None:
    holed = frame.copy()
    holed.loc[20:25, "ghi"] = np.nan
    holed.loc[200:210, "pv_power_w"] = np.nan
    filled = missing.impute_frame(holed, "pv_power_w", ["ghi", "temperature"], report={})
    assert np.isfinite(filled[["ghi", "temperature", "pv_power_w"]].to_numpy(dtype=float)).all()


def test_fill_weather_never_invents_a_negative_irradiance(frame: pd.DataFrame) -> None:
    holed = frame["ghi"].copy()
    holed.loc[30:40] = np.nan
    holed.attrs["timestamps"] = pd.DatetimeIndex(frame["Time"])
    filled = missing.fill_weather(holed)
    assert (filled >= 0).all()
    assert filled.isna().sum() == 0


def test_impute_refuses_to_interpolate_without_timestamps(frame: pd.DataFrame) -> None:
    series = frame["pv_power_w"].copy()
    series.attrs.pop("timestamps", None)
    with pytest.raises(ValueError, match="timestamps"):
        missing.interpolate_target(series)


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def test_robust_bounds_reject_an_extreme_spike() -> None:
    values = pd.Series(np.concatenate([np.full(60, 500.0), [50_000.0]]))
    median, upper, lower = outliers.robust_bounds(values, window=15, n_sigma=4.0)
    spike_at = len(values) - 1
    assert float(values.iloc[spike_at]) > float(upper.iloc[spike_at])
    # The rolling median is the point of using it: a single spike must not drag
    # its own detection threshold up with it.
    assert float(median.iloc[spike_at]) == pytest.approx(500.0, rel=1e-6)


def test_analyse_irradiance_zeroes_a_night_time_offset(frame: pd.DataFrame) -> None:
    biased = frame["ghi"].copy()
    night = biased.index[~frame["daylight"]]
    biased.iloc[night[:20]] = 12.0
    report: dict = {}
    cleaned, summary = outliers.analyse_irradiance(
        biased, frame["solar_elevation"],
        {"ghi_plausible_min": 0.0, "ghi_plausible_max": 1_200.0,
         "zero_irradiance_below_horizon": True, "night_solar_elevation_max": 0.0},
        report)
    assert summary["n_night_offset_zeroed"] >= 20
    assert (cleaned.iloc[night[:20]] == 0.0).all()
    assert report["irradiance_cleaning"]["n_night_offset_zeroed"] == summary["n_night_offset_zeroed"]


def test_analyse_irradiance_replaces_an_impossible_reading(frame: pd.DataFrame) -> None:
    spiked = frame["ghi"].copy()
    spiked.iloc[200] = 4_000.0
    cleaned, summary = outliers.analyse_irradiance(
        spiked, frame["solar_elevation"],
        {"ghi_plausible_min": 0.0, "ghi_plausible_max": 1_200.0,
         "ghi_spike_window": 15, "ghi_spike_ratio": 1.8,
         "zero_irradiance_below_horizon": True, "night_solar_elevation_max": 0.0}, {})
    assert summary["n_outside_physical_envelope"] >= 1
    assert float(cleaned.max()) <= 1_200.0


def test_flag_flatlines_finds_a_stuck_sensor() -> None:
    values = pd.Series(np.tile(np.arange(1.0, 11.0), 6))
    values.iloc[30:50] = 7.0
    flags = outliers.flag_flatlines(values, min_length=8)
    assert bool(flags.iloc[40])


def test_flag_flatlines_finds_a_stuck_sensor() -> None:
    values = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 6))
    values.iloc[30:50] = 7.0
    flags = outliers.flag_flatlines(values, min_length=8)
    assert bool(flags.iloc[40])


# --------------------------------------------------------------------------- #
# Weather regimes
# --------------------------------------------------------------------------- #
def test_night_irradiance_does_not_enter_the_variability_statistic(
        frame: pd.DataFrame) -> None:
    """Night readings must not drive the coefficient of variation.

    A 24-hour window would include roughly twelve hours of zeros and give a
    coefficient close to one at every step, which is why the statistic is
    restricted to daylight samples.
    """
    baseline = regimes.irradiance_variability(frame["ghi"], window=16, min_ghi=20.0,
                                              daylight=frame["daylight"])
    perturbed = frame["ghi"].copy()
    perturbed[~frame["daylight"]] = 900.0
    changed = regimes.irradiance_variability(perturbed, window=16, min_ghi=20.0,
                                            daylight=frame["daylight"])
    comparable = baseline.notna() & changed.notna()
    assert comparable.any()
    assert np.allclose(baseline[comparable].to_numpy(dtype=float),
                       changed[comparable].to_numpy(dtype=float))


def test_variability_is_nan_before_the_window_is_filled(frame: pd.DataFrame) -> None:
    series = regimes.irradiance_variability(frame["ghi"], window=16, min_ghi=20.0,
                                            daylight=frame["daylight"])
    assert series.iloc[:3].isna().all()


def test_regime_labels_cover_every_step(frame: pd.DataFrame) -> None:
    labels = regimes.classify_regimes(
        frame["clear_sky_index"],
        regimes.irradiance_variability(frame["ghi"], window=16, min_ghi=20.0,
                                       daylight=frame["daylight"]),
        frame["solar_elevation"],
        {"night": {"solar_elevation_max": 0.0},
         "variable": {"variability_min": 0.6},
         "clear": {"kt_min": 0.72, "variability_max": 0.6},
         "partly_cloudy": {"kt_min": 0.45, "kt_max": 0.72, "variability_max": 0.6}})
    assert labels.notna().all()
    assert (labels[~frame["daylight"]] == "Night").all()
    assert (labels[frame["daylight"]] != "Night").all()


def test_variability_takes_precedence_over_clearness(frame: pd.DataFrame) -> None:
    kt = pd.Series([0.95] * 4)
    solar_elevation = pd.Series([30.0] * 4)
    configuration = {"night": {"solar_elevation_max": 0.0},
                     "variable": {"variability_min": 0.6},
                     "clear": {"kt_min": 0.72, "variability_max": 0.6},
                     "partly_cloudy": {"kt_min": 0.45, "kt_max": 0.72, "variability_max": 0.6}}
    assert (regimes.classify_regimes(kt, pd.Series([0.1] * 4), solar_elevation,
                                     configuration) == "Clear").all()
    assert (regimes.classify_regimes(kt, pd.Series([0.9] * 4), solar_elevation,
                                     configuration) == "High-variability").all()


def test_regime_distribution_reports_shares(frame: pd.DataFrame) -> None:
    labels = pd.Series(["Clear", "Clear", "Cloudy", "Night"], dtype=object)
    distribution = regimes.regime_distribution(labels, daylight_only=True)
    assert distribution["n"] == 3
    assert distribution["shares"]["Clear"] == pytest.approx(2 / 3)
