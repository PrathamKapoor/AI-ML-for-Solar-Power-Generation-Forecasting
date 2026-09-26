"""Baseline forecasting models.

The persistence family is the reference against which every learned model is
judged. In solar forecasting this is not a formality: a persistence forecast
is what a plant operator actually falls back on, and a model that cannot beat it
has no operational value. Two variants are implemented:

**Persistence** repeats the most recent observation:
``f(t + h) = y(t)``

**Smart persistence** repeats the most recent observation rescaled by the ratio
of the clear-sky reference at the two timestamps:
``f(t + h) = y(t) * GHI_clear(t + h) / GHI_clear(t)``

The second is the standard operational form. It captures the dominant diurnal
shape of PV output using only information available at forecast time, and it is
a genuinely strong benchmark: it is exactly zero at night, it cannot produce a
physically impossible value, and it needs no training. Any claimed improvement
of a neural network over a learned model has to clear this bar.

Both are implemented as estimators with the same interface as the learned
models so that the experiment runner treats every model identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class PersistenceModel:
    """Repeat the most recent observed value.

    The forecast is ``f(t+h) = y(t)``: whatever the plant produced at the
    forecast origin is what it will produce ``h`` steps later. The input that
    holds ``y(t)`` is :attr:`feature_name` -- ``pv_power_w_lag_0``, because the
    lag index counts steps *back from the origin*, and zero steps back is the
    origin itself.

    Parameters
    ----------
    lag
        Steps before the forecast origin to read. ``0`` is persistence and is
        the default. Larger values are staler by construction and are only
        meaningful as a deliberately degraded comparison.
    """

    family = "baseline"
    name = "persistence"
    requires_training = False
    sequence_input = False

    def __init__(self, lag: int = 0, **_: Any) -> None:
        if lag < 0:
            raise ValueError(f"persistence lag must be >= 0, got {lag}")
        self.lag = int(lag)
        self.name = "persistence" if lag == 0 else f"persistence_lag{lag}"
        self.feature_name = f"pv_power_w_lag_{lag}"
        self._last: float | None = None

    def fit(self, y_true: np.ndarray, X: np.ndarray | None = None) -> "PersistenceModel":
        values = np.asarray(y_true, dtype=float).ravel()
        self._last = float(values[-1]) if values.size else 0.0
        return self

    def predict(self, X: np.ndarray, last_values: np.ndarray | pd.Series | float = 0.0
                ) -> np.ndarray:
        """Predict from the lag feature present in ``X``.

        Accepts either a flattened 2-D window matrix, where :attr:`feature_index`
        selects one position, or a 3-D ``(n, lookback, n_features)`` tensor, from
        which the *most recent* timestep is taken -- the origin, which is where
        the lag-0 value lives. Reading every timestep instead would return one
        value per window step and silently misalign the forecast with its target.
        """
        if X.ndim == 3:
            column = X[:, -1, self.feature_index]
        elif X.ndim == 2:
            column = X[:, self.feature_index]
        else:
            raise ValueError(f"expected a 2-D or 3-D feature matrix, got shape {X.shape}")
        return np.asarray(column, dtype=float).reshape(-1)

    #: Index into the feature matrix, set by the experiment runner. Position 0 is
    #: only correct if ``pv_power_w_lag_0`` is the first selected feature, so the
    #: runner sets this explicitly rather than relying on the default.
    feature_index: int = 0
    feature_name: str = "pv_power_w_lag_0"

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "requires_training": self.requires_training,
            "formula": "f(t+h) = y(t)",
            "inputs": [self.feature_name],
            "parameters": 0,
        }


#: Below this clear-sky reference (W/m2) the ratio form of smart persistence is
#: numerically unstable: at sunrise and sunset the reference approaches zero while
#: the previous power reading need not, so the ratio can reach hundreds and the
#: forecast leaves any physically meaningful range. Measured on the study data,
#: the unguarded form reaches an RMSE of 142 kW against a 55 kW plant. Above the
#: threshold the ratio is well conditioned.
SMART_PERSISTENCE_REFERENCE_FLOOR_WM2 = 20.0

#: The ratio is additionally clipped to this range, which bounds the forecast
#: even where both the reference and the power reading are individually
#: well-behaved but their ratio is extreme.
SMART_PERSISTENCE_RATIO_LIMITS = (0.0, 10.0)


def smart_persistence_forecast(last_value: np.ndarray, ghi_clear_now: np.ndarray,
                               ghi_clear_target: np.ndarray,
                               rated_w: float | None = None,
                               reference_floor: float = SMART_PERSISTENCE_REFERENCE_FLOOR_WM2,
                               ratio_limits: tuple[float, float] = SMART_PERSISTENCE_RATIO_LIMITS
                               ) -> np.ndarray:
    """Compute the guarded smart-persistence forecast.

    ``f(t+h) = y(t) * clip(GHI_clear(t+h) / max(GHI_clear(t), floor), lo, hi)``

    Three guards, each necessary:

    1. **Reference floor.** Where ``GHI_clear(t)`` is below ``reference_floor``
       the ratio is not computed; the forecast falls back to plain persistence of
       the last reading, which is itself near zero at those times of day.
    2. **Ratio clip.** The ratio is bounded so a modest change in the clear-sky
       reference cannot produce an unbounded forecast.
    3. **Capacity clip.** When the rated capacity is known, the forecast is
       clipped to it, because a forecast above nameplate is impossible
       regardless of what the model believes.
    """
    last = np.asarray(last_value, dtype=float).ravel()
    now = np.asarray(ghi_clear_now, dtype=float).ravel()
    target = np.asarray(ghi_clear_target, dtype=float).ravel()
    if not (last.shape == now.shape == target.shape):
        raise ValueError(
            f"shape mismatch: last {last.shape}, now {now.shape}, target {target.shape}")

    usable = now > float(reference_floor)
    ratio = np.where(usable, target / np.where(usable, now, 1.0), 1.0)
    ratio = np.clip(ratio, float(ratio_limits[0]), float(ratio_limits[1]))

    forecast = last * ratio
    if rated_w and rated_w > 0:
        forecast = np.clip(forecast, 0.0, float(rated_w))
    else:
        forecast = np.clip(forecast, 0.0, None)
    return np.nan_to_num(forecast, nan=0.0, posinf=0.0, neginf=0.0)


def smart_persistence_from_frame(frame: pd.DataFrame, horizon_steps: int,
                                 ghi_clear_column: str = "ghi_clear",
                                 last_column: str = "pv_power_w_lag_1",
                                 rated_w: float | None = None,
                                 reference_floor: float = SMART_PERSISTENCE_REFERENCE_FLOOR_WM2
                                 ) -> np.ndarray:
    """Reference smart-persistence forecast computed directly from a frame.

    The clear-sky reference at the target timestamp is the reference shifted
    forward by ``horizon_steps``, which is available at forecast time because it
    is a deterministic function of the clock and the site.
    """
    now = frame[ghi_clear_column].to_numpy(dtype=float)
    target = frame[ghi_clear_column].shift(-horizon_steps).to_numpy(dtype=float)
    last = frame[last_column].to_numpy(dtype=float)
    return smart_persistence_forecast(last, now, target, rated_w=rated_w,
                                      reference_floor=reference_floor)


class SmartPersistenceModel:
    """Persistence rescaled by the change in the clear-sky reference.

    ``f(t + h) = y(t) * clip(GHI_clear(t + h) / max(GHI_clear(t), floor), lo, hi)``

    See :func:`smart_persistence_forecast` for the three guards that make the
    ratio form numerically safe; without them the forecast is numerically
    unstable at sunrise and sunset and can exceed nameplate.
    """

    family = "baseline"
    name = "smart_persistence"
    requires_training = False
    sequence_input = False

    def __init__(self, rated_w: float | None = None,
                 reference_floor: float = SMART_PERSISTENCE_REFERENCE_FLOOR_WM2,
                 **_: Any) -> None:
        self.rated_w = rated_w
        self.reference_floor = float(reference_floor)
        self._last: float | None = None

    def fit(self, y_true: np.ndarray, X: np.ndarray | None = None) -> "SmartPersistenceModel":
        values = np.asarray(y_true, dtype=float).ravel()
        self._last = float(values[-1]) if values.size else 0.0
        return self

    def predict(self, ghi_clear_now: np.ndarray, ghi_clear_target: np.ndarray,
                last_value: np.ndarray) -> np.ndarray:
        return smart_persistence_forecast(
            last_value, ghi_clear_now, ghi_clear_target,
            rated_w=self.rated_w, reference_floor=self.reference_floor)

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "requires_training": self.requires_training,
            "formula": "f(t+h) = y(t) * clip(GHI_clear(t+h) / max(GHI_clear(t), "
                       f"{self.reference_floor}), 0, 10), clipped to rated capacity",
            "inputs": ["pv_power_w_lag_0", "ghi_clear_now", "ghi_clear_target"],
            "parameters": 0,
            "guards": {
                "reference_floor_wm2": self.reference_floor,
                "ratio_limits": list(SMART_PERSISTENCE_RATIO_LIMITS),
                "rated_w": self.rated_w,
            },
        }



def persistence_from_lag(frame: pd.DataFrame, lag: int = 0,
                          column: str = "pv_power_w") -> np.ndarray:
    """Reference persistence forecast computed directly from a feature frame.

    ``lag`` is measured in the other direction from a lag *feature*: ``lag=0``
    (the default) is the observation **at** the row, which is the most recent
    value available at the forecast origin, and therefore the correct
    persistence forecast. ``lag=k`` for ``k > 0`` reproduces the value ``k`` steps
    earlier, which is what the ``pv_power_w_lag_k`` features hold and is a
    deliberately staler comparison.

    Getting this convention right matters. The aligned persistence forecast for a
    sample whose target is observed at frame row ``r`` is ``pv_power_w[r - h]``,
    which is the value at the forecast origin. Using ``pv_power_w_lag_1`` instead
    would be one step staler and would understate every learned model's skill.
    """
    if lag < 0:
        raise ValueError(f"persistence lag must be >= 0, got {lag}")
    # shift(+lag), not shift(-lag): "k steps earlier" must agree with the
    # pv_power_w_lag_k features, which hold the value k steps back. The tail
    # becomes NaN because no earlier observation exists there, which is correct
    # and is why the horizon comparison always reads this reference at the
    # forecast origin.
    return frame[column].shift(lag).to_numpy(dtype=float)


def make_baseline(name: str, **kwargs: Any):
    """Factory used by the model registry."""
    key = name.lower()
    if key in {"persistence", "naive"}:
        return PersistenceModel(**kwargs)
    if key in {"smart_persistence", "smart-persistence", "clear_sky_persistence"}:
        return SmartPersistenceModel(**kwargs)
    raise KeyError(f"unknown baseline model: {name!r}")
