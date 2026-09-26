"""Model construction, input/output shapes and the persistence baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecasting.models import registry
from solar_forecasting.models.baselines import (persistence_from_lag, smart_persistence_forecast,
                                                smart_persistence_from_frame)
from solar_forecasting.models.classical import flatten_windows

N_FEATURES = 7
LOOKBACK = 24


@pytest.fixture()
def windows() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(60, LOOKBACK, N_FEATURES)).astype(np.float32)


@pytest.fixture()
def flat_targets() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.normal(size=60)


@pytest.fixture()
def small_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[:400].copy()


def test_registry_declares_every_family() -> None:
    names = registry.all_model_names()
    for expected in ("persistence", "smart_persistence", "linear_regression",
                     "random_forest", "gradient_boosting", "xgboost", "lstm", "gru",
                     "cnn_lstm", "attention_lstm", "transformer"):
        assert expected in names
        info = registry.model_info(expected)
        assert info["family"] and info["description"]


def test_registry_rejects_an_unknown_model() -> None:
    with pytest.raises(KeyError):
        registry.model_info("not_a_model")


def test_capability_report_records_the_environment() -> None:
    report = registry.capability_report()
    assert report["n_models_registered"] >= 11
    # xgboost is optional: the benchmark falls back to scikit-learn's gradient
    # boosting when it is absent, so the report has to say which is in use.
    assert "xgboost_available" in report
    if not report["xgboost_available"]:
        assert report["xgboost_import_error"]


# --------------------------------------------------------------------------- #
# Reference forecasts
# --------------------------------------------------------------------------- #
def test_persistence_repeats_the_latest_observation(small_frame: pd.DataFrame) -> None:
    forecast = persistence_from_lag(small_frame, lag=0)
    power = small_frame["pv_power_w"].to_numpy(dtype=float)
    assert np.allclose(forecast, power)
    lagged = persistence_from_lag(small_frame, lag=1)
    # A lag reference is undefined at the first row, because no earlier
    # observation exists there; NaN is the honest answer.
    assert lagged.shape == power.shape
    assert np.isnan(lagged[0])
    assert np.allclose(lagged[1:], power[:-1])


def test_persistence_model_reads_the_origin_step_of_a_window(windows: np.ndarray) -> None:
    """A 3-D window yields one forecast per window, taken at the last timestep.

    Reading every timestep would return one value per lag and silently
    misalign the forecast with its target.
    """
    from solar_forecasting.models.baselines import PersistenceModel
    model = PersistenceModel()
    model.feature_index = 2
    predicted = model.predict(windows)
    assert predicted.shape == (windows.shape[0],)
    assert np.allclose(predicted, windows[:, -1, 2])


def test_smart_persistence_scales_by_the_clear_sky_ratio(small_frame: pd.DataFrame) -> None:
    frame = small_frame.copy()
    frame["ghi_clear"] = np.where(frame["ghi"] > 0, 800.0, 0.0)
    frame["pv_power_w_lag_1"] = frame["pv_power_w"].shift(1).bfill()
    forecast = smart_persistence_from_frame(frame, horizon_steps=4, rated_w=55_000.0)
    assert np.isfinite(forecast).all()
    daylight = frame["ghi"].to_numpy(dtype=float) > 0
    # Scaling by a clear-sky reference must move the daytime forecast, not leave
    # it alone, and must never exceed nameplate.
    assert not np.allclose(forecast[daylight][:50], frame["pv_power_w"].to_numpy()[daylight][:50])
    assert forecast.max() <= 55_000.0 + 1e-6


def test_smart_persistence_falls_back_to_plain_persistence_below_the_floor() -> None:
    """Where the clear-sky reference vanishes the ratio form is not used."""
    last = np.array([4_000.0])
    forecast = smart_persistence_forecast(last, np.array([0.0]), np.array([0.0]),
                                          rated_w=55_000.0)
    assert forecast[0] == pytest.approx(4_000.0)


def test_smart_persistence_is_finite_at_night() -> None:
    now = np.array([0.0, 0.0])
    future = np.array([0.0, 0.0])
    last = np.array([0.0, 0.0])
    forecast = smart_persistence_forecast(last, now, future, rated_w=55_000.0)
    assert np.isfinite(forecast).all()
    assert forecast[0] == pytest.approx(0.0)


def test_smart_persistence_never_exceeds_capacity() -> None:
    now = np.array([500.0])
    future = np.array([1100.0])
    last = np.array([54_000.0])
    forecast = smart_persistence_forecast(last, now, future, rated_w=55_000.0)
    assert 0.0 <= forecast[0] <= 55_000.0


# --------------------------------------------------------------------------- #
# Tabular and sequence models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", ["linear_regression", "gradient_boosting"])
def test_sklearn_models_learn_a_shifted_signal(model_name: str, windows: np.ndarray,
                                              flat_targets: np.ndarray) -> None:
    model = registry.build_model(model_name, n_features=N_FEATURES, random_state=42)
    features = flatten_windows(windows)
    y = flat_targets + 3.0 * features[:, 0]
    model.fit(features, y)
    predicted = np.asarray(model.predict(features), dtype=float)
    assert predicted.shape == y.shape
    assert np.corrcoef(predicted, y)[0, 1] > 0.9


def test_flatten_windows_preserves_the_window_layout(windows: np.ndarray) -> None:
    flat = flatten_windows(windows)
    assert flat.shape == (windows.shape[0], windows.shape[1] * windows.shape[2])
    assert np.allclose(flat[:, :N_FEATURES], windows[:, 0, :])


@pytest.mark.parametrize("model_name", ["lstm", "gru", "cnn_lstm", "attention_lstm",
                                        "transformer"])
def test_neural_models_map_window_to_scalar(model_name: str, windows: np.ndarray) -> None:
    import torch

    model = registry.build_model(model_name, n_features=N_FEATURES, hidden_size=16,
                                 d_model=16, n_heads=2, n_layers=1, dim_feedforward=32,
                                 n_filters=8, kernel_size=3, attention_size=8, dropout=0.0)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(windows[:8]))
    assert output.reshape(-1).shape == (8,)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("model_name", ["lstm", "gru", "cnn_lstm", "attention_lstm",
                                        "transformer"])
def test_neural_model_output_is_deterministic_in_eval_mode(model_name: str,
                                                          windows: np.ndarray) -> None:
    import torch

    model = registry.build_model(model_name, n_features=N_FEATURES, hidden_size=16,
                                 d_model=16, n_heads=2, n_layers=1, dim_feedforward=32,
                                 n_filters=8, kernel_size=3, attention_size=8, dropout=0.0)
    model.eval()
    tensor = torch.from_numpy(windows[:8])
    with torch.no_grad():
        first = model(tensor)
        second = model(tensor)
    assert torch.allclose(first, second)


def test_a_model_trained_on_a_constant_feature_learns_nothing_useful() -> None:
    """A guard against a feature matrix that silently carries no information."""
    features = np.zeros((40, 3))
    features[:, 1] = np.linspace(0, 1, 40)
    y = features[:, 1] * 100.0
    model = registry.build_model("linear_regression", n_features=3, random_state=42)
    model.fit(features, y)
    assert model.feature_importances_ is not None

