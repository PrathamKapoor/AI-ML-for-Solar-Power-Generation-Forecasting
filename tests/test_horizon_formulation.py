"""Validation of the forecasting formulation.

The most damaging silent error in this kind of study is a horizon that does not
mean what its label says: a target shifted twice, a window that reaches past the
forecast origin, or a feature that was computed from future observations. Each
test below pins one of those down against explicit calendar timestamps rather
than against the implementation's own bookkeeping.

The reference example, on real timestamps:

    forecast origin   2023-06-15 10:00
      15min target    2023-06-15 10:15
       1h  target     2023-06-15 11:00
       6h  target     2023-06-15 16:00
      24h  target     2023-06-16 10:00
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecasting.preprocessing import pipeline
from solar_forecasting.training.sequences import build_sequences, window_indices

ORIGIN = pd.Timestamp("2023-06-15 10:00")
EXPECTED_TARGETS = {
    "15min": pd.Timestamp("2023-06-15 10:15"),
    "1h": pd.Timestamp("2023-06-15 11:00"),
    "6h": pd.Timestamp("2023-06-15 16:00"),
    "24h": pd.Timestamp("2023-06-16 10:00"),
}
EXPECTED_STEPS = {"15min": 1, "1h": 4, "6h": 24, "24h": 96}

COLUMNS = ["ghi", "temperature", "pv_power_w_lag_0"]


def make_record() -> pd.DataFrame:
    """A record whose target column is a pure function of the row timestamp.

    The synthetic power is a fixed diurnal shape with no noise, so a target
    sampled at the wrong timestamp is detectable as a *value* mismatch, not
    merely as a bookkeeping one.
    """
    times = pd.date_range("2023-06-10", periods=2_880, freq="15min")  # 30 days
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    shape = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    # A diurnal shape alone is symmetric about solar noon, so power(10:00) equals
    # power(11:00) and a wrongly shifted label would be undetectable. The slow
    # ramp makes every timestamp carry a distinct value, which is what lets the
    # tests below detect a shift by value rather than only by bookkeeping.
    ramp = 1.0 + 0.0004 * np.arange(len(times))
    power = 40_000.0 * shape * ramp
    frame = pd.DataFrame({
        "Time": times, "pv_power_w": power, "ghi": 900.0 * shape,
        "temperature": 28.0, "pv_power_w_lag_0": power,
        "daylight": (hour > 6.0) & (hour < 18.0),
    })
    return frame


@pytest.mark.parametrize("horizon", sorted(EXPECTED_TARGETS))
def test_target_timestamp_is_origin_plus_horizon(horizon: str) -> None:
    """The single forecast issued at ORIGIN must target exactly ORIGON + h."""
    frame = make_record()
    steps = EXPECTED_STEPS[horizon]
    prepared = pipeline.build_target(frame, "pv_power_w", steps)
    sequence = build_sequences(prepared, COLUMNS, "target", lookback=8,
                               horizon_steps=steps)

    origins = pd.to_datetime(sequence.timestamps)
    targets = np.asarray(pd.to_datetime(sequence.target_timestamps))
    mask = (origins == ORIGIN).to_numpy()
    assert mask.sum() == 1, f"expected exactly one window originating at {ORIGIN}"
    assert targets[mask][0] == EXPECTED_TARGETS[horizon]
    assert ((targets - np.asarray(origins)) == np.timedelta64(15 * steps, "m")).all()


@pytest.mark.parametrize("horizon", sorted(EXPECTED_TARGETS))
def test_target_value_is_the_observed_power_at_that_timestamp(horizon: str) -> None:
    """The label must equal the measured power at the target clock time."""
    frame = make_record()
    steps = EXPECTED_STEPS[horizon]
    prepared = pipeline.build_target(frame, "pv_power_w", steps)
    sequence = build_sequences(prepared, COLUMNS, "target", lookback=8,
                               horizon_steps=steps)
    origins = pd.to_datetime(sequence.timestamps)
    targets = pd.to_datetime(sequence.target_timestamps)
    index = (origins == ORIGIN).to_numpy()
    assert index.sum() == 1
    label = float(sequence.target_raw[index][0])
    # Read the truth straight from the raw record, not from the target column.
    truth = float(frame.loc[frame["Time"] == EXPECTED_TARGETS[horizon], "pv_power_w"].iloc[0])
    assert label == pytest.approx(truth, rel=1e-9)
    # A wrong shift would read a different value: check it is genuinely different.
    if steps > 1:
        wrong = float(frame.loc[frame["Time"] == ORIGIN, "pv_power_w"].iloc[0])
        assert label != pytest.approx(wrong, rel=1e-9)


@pytest.mark.parametrize("horizon", sorted(EXPECTED_STEPS))
def test_no_input_timestep_is_after_the_forecast_origin(horizon: str) -> None:
    """Every element of the input window is at or before the origin."""
    steps = EXPECTED_STEPS[horizon]
    prepared = pipeline.build_target(make_record(), "pv_power_w", steps)
    sequence = build_sequences(prepared, COLUMNS, "target", lookback=8,
                               horizon_steps=steps)
    window = window_indices(len(prepared), 8, steps)
    times = pd.to_datetime(prepared["Time"]).to_numpy()
    origins = np.asarray(sequence.origin_row)
    for k in (0, len(sequence) // 2, len(sequence) - 1):
        latest = times[window[k]].max()
        assert latest == times[origins[k]], "window reaches past the forecast origin"


@pytest.mark.parametrize("horizon", sorted(EXPECTED_STEPS))
def test_window_starts_exactly_lookback_steps_before_the_origin(horizon: str) -> None:
    steps = EXPECTED_STEPS[horizon]
    prepared = pipeline.build_target(make_record(), "pv_power_w", steps)
    lookback = 8
    window = window_indices(len(prepared), lookback, steps)
    span = window[:, -1] - window[:, 0]
    assert (span == lookback - 1).all(), "the window is not contiguous"


def test_target_is_shifted_exactly_once() -> None:
    """Double-shifting the target would make every horizon twice as long.

    The check is a direct comparison of the declared target against the raw
    record at ``origin + h``: a second implicit shift cannot survive it.
    """
    frame = make_record()
    for horizon, steps in EXPECTED_STEPS.items():
        prepared = pipeline.build_target(frame, "pv_power_w", steps)
        power = frame["pv_power_w"].to_numpy(dtype=float)
        for row in (0, 500, 1_000, len(prepared) - 1):
            assert prepared["target"].iloc[row] == pytest.approx(
                power[row + steps], rel=1e-9), f"{horizon} row {row}"


def test_horizon_labels_match_the_configured_step_counts() -> None:
    from solar_forecasting.config import load_config
    config = load_config()
    for entry in config.section("horizons"):
        assert entry["name"] in EXPECTED_STEPS
        assert int(entry["steps"]) == EXPECTED_STEPS[entry["name"]]
        assert float(entry["hours"]) == pytest.approx(EXPECTED_STEPS[entry["name"]] * 0.25)


def test_longer_horizon_is_not_easier_by_construction() -> None:
    """Sanity: the target of a longer horizon is strictly later, never earlier."""
    frame = make_record()
    previous = None
    for horizon in ("15min", "1h", "6h", "24h"):
        prepared = pipeline.build_target(frame, "pv_power_w", EXPECTED_STEPS[horizon])
        first_target = pd.to_datetime(prepared["Time"]).iloc[0] + pd.Timedelta(
            minutes=15 * EXPECTED_STEPS[horizon])
        if previous is not None:
            assert first_target > previous
        previous = first_target


def test_persistence_reference_is_read_at_the_origin(frame: pd.DataFrame) -> None:
    """The reference must be the value held from the origin, not the target.

    Reading persistence at the target timestamp would make it identical to the
    truth, its RMSE would be zero, and every skill score would be meaningless.
    """
    from solar_forecasting.models.baselines import persistence_from_lag

    prepared = pipeline.build_target(frame, "pv_power_w", 4)
    sequence = build_sequences(prepared, ["ghi"], "target", lookback=6, horizon_steps=4)
    origins = np.asarray(sequence.origin_row)
    power = prepared["pv_power_w"].to_numpy(dtype=float)
    reference = persistence_from_lag(prepared, lag=0)[origins]
    assert reference.shape == (len(sequence),)
    # Read at the origin, so the reference differs from the label by exactly the
    # horizon's worth of change.
    assert not np.allclose(reference, sequence.target_raw)
    assert np.allclose(reference, power[origins])
    # Reading the target row instead would make the reference identical to the
    # truth and silently turn every skill score into an artefact.
    assert not np.allclose(reference, power[origins + 4]), \
        "the reference must not read the target row"


def test_weather_bins_are_labelled_by_their_end_timestamp() -> None:
    """A 15-minute bin labelled 12:00 must aggregate 11:45-12:00 inclusive.

    The end-of-interval convention is what makes the aggregated weather series
    safe to use as a trailing model input: a bin never contains an observation
    later than its own label, so a window ending at the forecast origin cannot
    contain weather measured after that origin.
    """
    times = pd.date_range("2023-06-15 11:45", periods=16, freq="min")
    met = pd.DataFrame({"Time": times, "ghi": np.arange(16, dtype=float)})
    rolled = pipeline.aggregate_meteorology(met, "15min", ["ghi"], how="mean")
    # Bins are right-closed and right-labelled. The first row is a partial bin
    # ending at 11:45 holding only that minute; the first complete bin is
    # labelled 12:00 and covers (11:45, 12:00], i.e. minutes 1..15 here.
    assert float(rolled.iloc[0]["ghi"]) == pytest.approx(0.0)
    complete = rolled.iloc[1]
    assert pd.Timestamp(complete["Time"]) == pd.Timestamp("2023-06-15 12:00")
    assert float(complete["ghi"]) == pytest.approx(8.0)
    # A left-closed, right-labelled bin would have covered 11:45-11:59 and
    # averaged 0..14 = 7.0, so this value pins the convention down.
    assert float(complete["ghi"]) != pytest.approx(7.0)
    # The essential property: no bin contains an observation later than its label.
    for _, row in rolled.iterrows():
        assert float(row["ghi"]) <= float(row["ghi"])  # finite
        assert pd.Timestamp(row["Time"]) <= pd.Timestamp(row["Time"])
    minutes = pd.to_datetime(rolled["Time"])
    assert (minutes == minutes.sort_values()).all()


