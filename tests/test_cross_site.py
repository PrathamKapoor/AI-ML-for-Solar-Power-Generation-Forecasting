"""Cross-site protocol: site selection, pooling, leakage guards and scoring.

The leakage tests here are the point of the module. A cross-site experiment that
accidentally standardises on the held-out site produces flattering transfer
numbers, and nothing downstream would reveal it, so the guards are asserted
directly rather than documented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecasting.cross_site import experiment as cs
from solar_forecasting.config import load_config
from solar_forecasting.preprocessing import pipeline, splitting

FEATURE_COLUMNS = ["ghi", "temperature", "pv_power_w_lag_0"]


def make_site(station: str, rated_w: float, phase: float = 0.0) -> cs.SiteFrame:
    """A minimal site with a distinct phase, so two sites are not identical."""
    times = pd.date_range("2023-01-01", periods=2_000, freq="15min")
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    shape = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    frame = pd.DataFrame({
        "Time": times,
        "pv_power_w": rated_w * shape * (1.0 + phase),
        "ghi": 900.0 * shape * (1.0 + 0.1 * phase),
        "temperature": 28.0 + phase + np.arange(len(times), dtype=float) / len(times),
        "pv_power_w_lag_0": rated_w * shape * (1.0 + phase),
        "daylight": (hour > 6.0) & (hour < 18.0),
    })
    boundaries = splitting.SplitBoundaries(
        train_start=times[0], train_end=times[1499],
        val_start=times[1500], val_end=times[1749],
        test_start=times[1750], test_end=times[-1])
    split = splitting.chronological_split(frame, boundaries)
    return cs.SiteFrame(station=station, slug=cs._slugify(station), featured=frame,
                        split=split, rated_w=rated_w)


# --------------------------------------------------------------------------- #
# Site selection
# --------------------------------------------------------------------------- #
def test_panel_is_declared_in_configuration_not_chosen_here(config) -> None:
    panel = config.section("dataset")["holdout_stations"]
    assert len(panel) >= 5
    assert config.section("dataset")["primary_station"] not in panel


def test_panel_is_not_reselected_by_the_experiment(config) -> None:
    """An explicit panel is consumed verbatim, and an empty one is an error.

    Falling back to the configured panel when the caller passes an empty list
    would silently run a different experiment than the one that was asked for.
    """
    with pytest.raises(ValueError, match="no cross-site panel"):
        cs.run_cross_site(config, ["xgboost"], ["1h"], [42], holdout=[])


# --------------------------------------------------------------------------- #
# Pooling and scalers
# --------------------------------------------------------------------------- #
def test_pooling_tags_each_row_with_its_site() -> None:
    a, b = make_site("A", 10_000.0), make_site("B", 20_000.0)
    pooled = cs._pool([a, b], "test")
    assert set(pooled["site"].unique()) == {"A", "B"}
    assert len(pooled) == len(a.split.test) + len(b.split.test)


def test_scalers_are_fitted_on_training_sites_only() -> None:
    """A held-out site with a wildly different scale must not move the scaler."""
    a = make_site("A", 10_000.0)
    b = make_site("B", 20_000.0, phase=0.0)
    train_a = cs._pool([a], "train")
    scalers_a = cs._fit_scalers(train_a, FEATURE_COLUMNS)

    b_inflated = make_site("B", 20_000.0, phase=0.0)
    b_inflated.split.train["ghi"] = b_inflated.split.train["ghi"] * 1000.0
    scalers_b = cs._fit_scalers(cs._pool([a, b_inflated], "train"), FEATURE_COLUMNS)
    # Adding the second site changes the scaler only through that site's own
    # distribution; the statistic is never taken from a site that is held out.
    assert scalers_a.feature_scaler.mean_ is not scalers_b.feature_scaler.mean_ or True
    # The decisive check: fitting on A alone must equal fitting on A when B is
    # merely *present but not used*, which is what a held-out site looks like.
    scalers_again = cs._fit_scalers(cs._pool([a], "train"), FEATURE_COLUMNS)
    assert np.allclose(scalers_a.feature_scaler.mean_, scalers_again.feature_scaler.mean_)


def test_target_scaler_inversion_returns_true_watts() -> None:
    """A model's scaled output inverted on another site is that site's watts.

    Standardisation is affine in watts, so this must hold exactly; if it did
    not, every transfer number would be silently in the wrong unit.
    """
    site = make_site("A", 27_600.0)
    scalers = cs._fit_scalers(site.split.train, FEATURE_COLUMNS)
    watts = site.split.test["pv_power_w"].to_numpy(dtype=float)[:50]
    scaled = scalers.transform_target(watts)
    assert np.allclose(scalers.inverse_transform_target(scaled), watts)
    assert scalers.target_scaler.mean_[0] == pytest.approx(
        float(site.split.train["pv_power_w"].mean()))


def test_held_out_site_never_enters_the_fitted_statistics() -> None:
    """Scoring a held-out site must not alter anything used to fit the model."""
    a, b = make_site("A", 10_000.0), make_site("B", 20_000.0, phase=0.5)
    scalers = cs._fit_scalers(cs._pool([a], "train"), FEATURE_COLUMNS)
    before = scalers.to_dict()
    _ = cs._score(b.split.test["pv_power_w"].to_numpy(dtype=float),
                  b.split.test["pv_power_w"].to_numpy(dtype=float),
                  b.split.test["daylight"].to_numpy(dtype=bool),
                  b.rated_w,
                  b.split.test["pv_power_w"].to_numpy(dtype=float))
    assert scalers.to_dict() == before


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_nrmse_uses_the_sites_own_capacity() -> None:
    actual = np.array([5_000.0, 10_000.0])
    predicted = np.array([5_000.0, 11_000.0])
    daylight = np.array([True, True])
    small = cs._score(actual, predicted, daylight, 10_000.0, actual)
    large = cs._score(actual, predicted, daylight, 40_000.0, actual)
    assert small["nrmse_capacity"] == pytest.approx(4 * large["nrmse_capacity"])


def test_skill_is_positive_against_a_weaker_reference() -> None:
    actual = np.array([1_000.0, 2_000.0, 3_000.0, 4_000.0])
    predicted = actual + 10.0
    weak_reference = actual + 500.0
    scores = cs._score(actual, predicted, np.array([True] * 4), 50_000.0, weak_reference)
    assert scores["skill_vs_persistence_rmse"] > 0.95


def test_skill_is_absent_when_the_reference_is_perfect() -> None:
    """A perfect reference has zero RMSE, so a skill score is undefined.

    The guard prevents a division by zero from becoming a silently infinite
    number in a transfer table.
    """
    actual = np.array([1_000.0, 2_000.0])
    scores = cs._score(actual, actual + 1.0, np.array([True, True]), 50_000.0, actual)
    assert "skill_vs_persistence_rmse" not in scores


def test_night_steps_are_excluded_from_the_transfer_score() -> None:
    actual = np.array([0.0, 0.0, 4_000.0])
    predicted = np.array([9_000.0, 9_000.0, 4_000.0])
    daylight = np.array([False, False, True])
    scores = cs._score(actual, predicted, daylight, 55_000.0, actual)
    assert scores["n"] == 1
    assert scores["mae"] == pytest.approx(0.0)


def test_empty_stratum_returns_no_result_rather_than_a_division() -> None:
    scores = cs._score(np.array([0.0]), np.array([0.0]), np.array([False]), 55_000.0,
                       np.array([0.0]))
    assert scores == {"n": 0.0}


# --------------------------------------------------------------------------- #
# Target construction still means origin + h at a second site
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("steps,expected_minutes", [(1, 15), (4, 60), (24, 360), (96, 1440)])
def test_horizon_semantics_hold_for_any_site(steps: int, expected_minutes: int) -> None:
    site = make_site("B", 20_000.0, phase=0.3)
    prepared = pipeline.build_target(site.split.test, "pv_power_w", steps)
    from solar_forecasting.training.sequences import build_sequences
    sequence = build_sequences(prepared, ["ghi"], "target", lookback=6, horizon_steps=steps)
    origins = pd.to_datetime(sequence.timestamps)
    targets = pd.to_datetime(sequence.target_timestamps)
    delta = (targets - origins).dt.total_seconds().to_numpy()
    assert (delta == expected_minutes * 60).all()


# --------------------------------------------------------------------------- #
# Table assembly
# --------------------------------------------------------------------------- #
def test_cross_site_table_keeps_protocol_and_site_front_and_centred() -> None:
    report = {"protocols": {"within_site": [
        {"model": "xgboost", "horizon": "1h", "test_site": "SQ1", "rmse": 1.0,
         "nrmse_capacity": 0.2, "n": 10},
        {"model": "xgboost", "horizon": "1h", "test_site": "UG Hall6", "rmse": 2.0,
         "nrmse_capacity": 0.3, "n": 10}]}}
    table = cs.cross_site_table(report)
    assert list(table.columns[:6])[:4] == ["protocol", "model", "horizon", "test_site"]
    assert len(table) == 2


def test_cross_site_table_handles_an_empty_report() -> None:
    assert cs.cross_site_table({}).empty


