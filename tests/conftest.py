"""Shared fixtures.

The tests never touch the real dataset: they build a small synthetic frame with
the same schema the pipeline produces, so the suite runs in seconds and does not
depend on a 300 MB download. Tests that genuinely need the real artefacts skip
themselves with a clear reason when the data has not been downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

RESOLUTION = "15min"
FEATURE_COLUMNS = ["ghi", "temperature", "pv_power_w_lag_0", "hour_sin", "doy_cos"]
RATED_W = 55_000.0


def make_frame(n_periods: int = 24 * 40, start: str = "2022-01-01",
               seed: int = 0) -> pd.DataFrame:
    """A synthetic frame with the schema the pipeline emits.

    The power series is generated from a diurnal shape plus noise, so it has the
    two properties the downstream code cares about: a hard zero at night and a
    smooth ramp in the morning.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=n_periods, freq=RESOLUTION)
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    season = np.sin(2 * np.pi * times.dayofyear.to_numpy() / 365.25)
    shape = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    ghi = np.clip(900.0 * shape * (0.75 + 0.25 * season) + rng.normal(0, 25, n_periods), 0, None)
    power = np.clip(RATED_W * 0.85 * (ghi / 1000.0) + rng.normal(0, 300, n_periods), 0, None)
    daylight = (hour > 6.0) & (hour < 18.0)
    power = np.where(daylight, power, 0.0)
    # Solar elevation consistent with the daylight mask: positive on the same
    # interval, negative outside it, so regime labelling and the daylight
    # handling behave the way they do on the real record.
    elevation = np.where(daylight, 75.0 * np.sin(np.pi * (hour - 6.0) / 12.0), -12.0)
    frame = pd.DataFrame({
        "Time": times,
        "pv_power_w": power,
        "ghi": ghi,
        "ghi_clear": np.where(daylight, 1000.0 * shape, 0.0),
        "clear_sky_index": np.where(ghi > 0, np.clip(ghi / np.maximum(np.where(daylight, 1000.0 * shape, 1.0), 1e-6), 0, 1.2), 0.0),
        "temperature": 25.0 + 5.0 * season + rng.normal(0, 1.0, n_periods),
        "relative_humidity": np.clip(70.0 - rng.normal(0, 10.0, n_periods), 10, 100),
        "wind_speed": np.clip(rng.normal(3.0, 1.0, n_periods), 0, None),
        "solar_elevation": elevation,
        "daylight": daylight,
        "season": np.where(times.month.isin([12, 1, 2]), "winter",
                           np.where(times.month.isin([3, 4, 5]), "spring",
                                    np.where(times.month.isin([6, 7, 8]), "summer", "autumn"))),
    })
    frame["regime"] = np.where(~frame["daylight"], "Night",
                               np.where(frame["clear_sky_index"] >= 0.72, "Clear",
                                        np.where(frame["clear_sky_index"] >= 0.45,
                                                 "Partly cloudy", "Cloudy")))
    for lag in (0, 1, 2):
        frame[f"pv_power_w_lag_{lag}"] = frame["pv_power_w"].shift(lag).bfill()
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["doy_cos"] = np.cos(2 * np.pi * times.dayofyear.to_numpy() / 365.25)
    # Time-indexed attributes that the imputation helpers require.
    for column in ("pv_power_w", "ghi", "temperature"):
        frame[column].attrs["timestamps"] = times
    return frame


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="session")
def config():
    from solar_forecasting.config import load_config
    return load_config()


@pytest.fixture()
def horizon_frame(frame: pd.DataFrame) -> pd.DataFrame:
    from solar_forecasting.preprocessing.pipeline import build_target
    return build_target(frame, "pv_power_w", 4)


@pytest.fixture()
def processed_available() -> bool:
    root = Path(__file__).resolve().parents[1]
    return (root / "data" / "processed" / "split_test.parquet").exists()
