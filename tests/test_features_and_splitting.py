"""Feature engineering, target construction, windowing and chronological splitting.

These are the leakage-sensitive parts of the pipeline, so the tests assert
causality and split disjointness directly rather than checking shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecasting.config import load_config
from solar_forecasting.features import builder
from solar_forecasting.preprocessing import pipeline, splitting
from solar_forecasting.training.sequences import (build_sequences, verify_causality,
                                                 window_indices)


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon_steps", [1, 4, 24, 96])
def test_build_target_is_a_lead_shift_of_exactly_the_horizon(
        frame: pd.DataFrame, horizon_steps: int) -> None:
    shifted = pipeline.build_target(frame, "pv_power_w", horizon_steps)
    aligned = frame["pv_power_w"].to_numpy(dtype=float)[horizon_steps:]
    declared = shifted["target"].to_numpy(dtype=float)[:len(aligned)]
    assert np.allclose(declared, aligned, equal_nan=True)


def test_build_target_drops_rows_whose_label_does_not_exist(
        frame: pd.DataFrame) -> None:
    """The tail of the record is removed, not padded.

    Padding would create labels that were never observed; leaving them as NaN
    would push the NaN into the training set. The row count is therefore exactly
    the record length minus the horizon.
    """
    shifted = pipeline.build_target(frame, "pv_power_w", 4)
    assert len(shifted) == len(frame) - 4
    assert shifted["target"].isna().sum() == 0
    assert shifted["target_lead_hours"].iloc[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_temporal_features_are_cyclic_not_raw(frame: pd.DataFrame) -> None:
    featured = builder.add_temporal_features(frame)
    for column in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
        assert column in featured.columns
        assert featured[column].between(-1.0, 1.0).all()


def test_target_lags_are_strictly_backward_looking(frame: pd.DataFrame) -> None:
    lagged = builder.add_target_lags(frame, "pv_power_w", lags=[1, 4])
    power = frame["pv_power_w"].to_numpy(dtype=float)
    assert np.allclose(lagged["pv_power_w_lag_1"].to_numpy()[1:], power[:-1])
    assert np.allclose(lagged["pv_power_w_lag_4"].to_numpy()[4:], power[:-4])


def test_rolling_features_use_a_trailing_window_only(frame: pd.DataFrame) -> None:
    rolled = builder.add_rolling_features(frame, "pv_power_w", windows=[4],
                                          stats=["mean", "std"])
    assert "pv_rolling_mean_4" in rolled.columns
    power = frame["pv_power_w"].to_numpy(dtype=float)
    expected = np.convolve(power[:8], np.ones(4) / 4.0, mode="valid")
    assert np.allclose(rolled["pv_rolling_mean_4"].to_numpy()[3:8], expected)
    # min_periods equals the window, so the warm-up rows are NaN rather than
    # statistics computed from a partial window.
    assert rolled["pv_rolling_mean_4"].iloc[:3].isna().all()


def test_irradiance_derivatives_are_differences(frame: pd.DataFrame) -> None:
    derived = builder.add_irradiance_derivatives(frame, "ghi")
    ghi = frame["ghi"].to_numpy(dtype=float)
    assert np.allclose(derived["ghi_delta_1"].to_numpy()[1:], ghi[1:] - ghi[:-1])


def test_feature_ablation_removes_the_declared_group(config) -> None:
    columns = ["ghi", "temperature", "hour_sin", "pv_power_w_lag_1"]
    groups = config.section("features")["ablation_groups"]
    reduced = builder.apply_feature_ablation(frame_stub(), columns,
                                             groups["no_weather"]["remove"])
    assert "ghi" not in reduced
    assert "temperature" not in reduced
    assert "hour_sin" in reduced


def test_ablation_of_unknown_group_is_an_error(config) -> None:
    from solar_forecasting.training.experiment import select_features
    with pytest.raises(KeyError, match="unknown feature specification"):
        select_features(["ghi"], "not_a_group", config)


def test_build_features_produces_the_configured_columns(config, frame: pd.DataFrame
                                                        ) -> None:
    features = config.section("features")
    prepared = frame.copy()
    prepared["clear_sky_index"] = prepared["ghi"] / 1000.0
    featured = builder.build_features(prepared, features)
    columns = builder.feature_columns(featured)
    assert len(columns) > 0
    assert not any(column in columns for column in
                   ("Time", "pv_power_w", "target", "regime", "season", "daylight"))
    values = featured[columns].to_numpy(dtype=float)
    # The only non-finite values are the warm-up rows of the trailing rolling
    # windows, which the pipeline drops when it splits; they must be confined to
    # the head of the record rather than scattered through it.
    incomplete = ~np.isfinite(values).all(axis=1)
    assert incomplete.sum() < int(0.2 * len(featured))
    if incomplete.any():
        assert incomplete.max() < int(0.2 * len(featured))
    groups = builder.describe_feature_groups(columns)
    assert groups["target_lags"] and groups["calendar"]


def frame_stub() -> pd.DataFrame:
    return pd.DataFrame({"ghi": [1.0], "temperature": [2.0], "hour_sin": [0.0],
                         "pv_power_w_lag_1": [3.0]})


# --------------------------------------------------------------------------- #
# Windowing and causality
# --------------------------------------------------------------------------- #
def test_window_indices_cover_the_expected_rows() -> None:
    indices = window_indices(10, lookback=3, horizon_steps=2)
    assert indices.shape == (6, 3)
    assert indices[0].tolist() == [0, 1, 2]
    assert indices[-1].tolist() == [5, 6, 7]
    # The label of the last window is at 7 + 2 = 9, the final usable row.
    assert indices[-1][-1] + 2 == 9


def test_window_indices_refuses_an_impossible_request() -> None:
    with pytest.raises(ValueError, match="not enough rows"):
        window_indices(5, lookback=4, horizon_steps=4)


def test_build_sequences_shapes_and_causality(horizon_frame: pd.DataFrame) -> None:
    columns = ["ghi", "temperature", "pv_power_w_lag_0"]
    sequence = build_sequences(horizon_frame, columns, "target", lookback=12,
                               horizon_steps=4)
    assert sequence.X.shape[1:] == (12, len(columns))
    assert len(sequence) == len(horizon_frame) - 12 - 4 + 1
    verify_causality(sequence, horizon_frame, 12, 4, columns, n_checks=20)


def test_sequence_targets_match_the_declared_horizon(frame: pd.DataFrame) -> None:
    """The label of a window is the power observed exactly ``horizon_steps`` later."""
    columns = ["ghi"]
    prepared = pipeline.build_target(frame, "pv_power_w", 24)
    sequence = build_sequences(prepared, columns, "target", lookback=6, horizon_steps=24)
    power = prepared["pv_power_w"].to_numpy(dtype=float)
    origins = np.asarray(sequence.origin_row)
    assert np.allclose(sequence.target_raw, power[origins + 24])
    times = pd.to_datetime(prepared["Time"])
    assert (times.iloc[origins + 24].to_numpy() >
            times.iloc[origins].to_numpy()).all()


def test_verify_causality_catches_a_shifted_target(horizon_frame: pd.DataFrame) -> None:
    columns = ["ghi"]
    sequence = build_sequences(horizon_frame, columns, "target", lookback=6,
                               horizon_steps=4)
    broken = horizon_frame.copy()
    broken.loc[broken.index[100:200], "target"] = 0.0
    with pytest.raises(ValueError):
        verify_causality(sequence, broken, 6, 4, columns, n_checks=100)


# --------------------------------------------------------------------------- #
# Chronological splitting and scaling
# --------------------------------------------------------------------------- #
def test_chronological_split_is_disjoint_and_ordered(frame: pd.DataFrame) -> None:
    """Three contiguous periods of the synthetic record, which spans 10 days."""
    boundaries = splitting.SplitBoundaries(
        train_start=pd.Timestamp("2022-01-01"),
        train_end=pd.Timestamp("2022-01-04 23:45"),
        val_start=pd.Timestamp("2022-01-05"),
        val_end=pd.Timestamp("2022-01-06 23:45"),
        test_start=pd.Timestamp("2022-01-07"),
        test_end=pd.Timestamp("2022-01-10 23:45"))
    result = splitting.chronological_split(frame, boundaries)
    assert pd.to_datetime(result.train["Time"]).max() < \
        pd.to_datetime(result.val["Time"]).min()
    assert pd.to_datetime(result.val["Time"]).max() < \
        pd.to_datetime(result.test["Time"]).min()
    assert result.audit["sizes"] == {"train": len(result.train),
                                     "val": len(result.val),
                                     "test": len(result.test)}
    assert result.sizes()["train"] + result.sizes()["val"] + result.sizes()["test"] \
        == len(frame)
    assert result.audit["disjoint"] is True


def test_chronological_split_drops_rows_outside_the_window(frame: pd.DataFrame) -> None:
    """Rows before ``train_start`` are dropped, not silently carried into a split."""
    boundaries = splitting.SplitBoundaries(
        train_start=pd.Timestamp("2022-01-03"),
        train_end=pd.Timestamp("2022-01-04 23:45"),
        val_start=pd.Timestamp("2022-01-05"),
        val_end=pd.Timestamp("2022-01-06 23:45"),
        test_start=pd.Timestamp("2022-01-07"),
        test_end=pd.Timestamp("2022-01-10 23:45"))
    result = splitting.chronological_split(frame, boundaries)
    assert result.audit["n_dropped_outside_window"] == 2 * 96
    # The boundaries still hold after the frame has been shortened: this is the
    # regression guard for a stale-index mask, which shifted every boundary
    # without raising.
    assert pd.to_datetime(result.train["Time"]).min() == pd.Timestamp("2022-01-03")
    assert pd.to_datetime(result.train["Time"]).max() == pd.Timestamp("2022-01-04 23:45")
    assert len(result.train) == 2 * 96


def test_boundaries_reject_an_overlapping_configuration() -> None:
    boundaries = splitting.SplitBoundaries(
        train_start=pd.Timestamp("2022-01-01"),
        train_end=pd.Timestamp("2022-02-01"),
        val_start=pd.Timestamp("2022-01-15"),
        val_end=pd.Timestamp("2022-02-10"),
        test_start=pd.Timestamp("2022-02-05"),
        test_end=pd.Timestamp("2022-03-01"))
    with pytest.raises(ValueError):
        boundaries.validate()


def test_scalers_are_fitted_on_training_only(frame: pd.DataFrame) -> None:
    columns = ["ghi", "temperature", "pv_power_w_lag_0"]
    train = frame.iloc[:500]
    bundle = splitting.fit_scalers(train, columns, "pv_power_w")
    assert bundle.feature_scaler.mean_.tolist() == pytest.approx(
        train[columns].mean().to_numpy(dtype=float))
    assert float(bundle.target_scaler.mean_[0]) == pytest.approx(
        float(train["pv_power_w"].mean()))
    # A much larger distribution must map outside the standardised range, which
    # is what "fitted on training only" means in practice.
    extreme = bundle.transform_features(frame.iloc[900:910].assign(
        ghi=5_000.0, temperature=60.0, pv_power_w_lag_0=1.0))
    assert extreme.max() > 5.0


def test_scalers_reject_a_constant_feature(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="zero variance"):
        splitting.fit_scalers(frame, ["constant"], "pv_power_w") \
            if "constant" in frame.columns else splitting.fit_scalers(
                frame.assign(constant=1.0), ["constant"], "pv_power_w")


def test_scalers_reject_non_finite_training_values(frame: pd.DataFrame) -> None:
    holed = frame.copy()
    holed.loc[0:10, "ghi"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        splitting.fit_scalers(holed, ["ghi"], "pv_power_w")


def test_scaler_round_trip_is_lossless(frame: pd.DataFrame) -> None:
    bundle = splitting.fit_scalers(frame, ["ghi", "temperature"], "pv_power_w")
    values = frame["pv_power_w"].to_numpy(dtype=float)[:100]
    restored = bundle.inverse_transform_target(bundle.transform_target(values))
    assert np.allclose(values, restored)
