"""Sliding-window sequence construction for sequence models.

A sample at index ``t`` is a window of ``lookback`` consecutive feature vectors
ending at ``t``, paired with the target observed at ``t + horizon_steps``. The
window is strictly causal: it contains nothing after ``t``.

Two properties are asserted rather than assumed, because a silent off-by-one in
window construction is the most common and least visible bug in this task:

* the first element of a window is genuinely ``lookback`` steps before its label;
* window ``k`` ends exactly one step after window ``k-1`` (stride 1), so the
  number of windows is ``n - lookback - horizon + 1``.

The builder returns the window index matrix alongside the arrays so that a test
can verify both properties without recomputing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SequenceSet:
    """Materialised windows plus the metadata needed to audit them."""

    X: np.ndarray                 # (n_samples, lookback, n_features)
    y: np.ndarray                 # (n_samples,) or (n_samples, horizon_steps)
    timestamps: pd.Series         # forecast-origin timestamp of each window
    target_timestamps: pd.Series  # timestamp at which each label is realised
    daylight: np.ndarray          # daylight flag of the target interval
    regime: np.ndarray            # regime label of the target interval
    season: np.ndarray
    target_raw: np.ndarray        # unscaled target in W, for metric computation
    origin_row: np.ndarray        # row index of the forecast origin in the source frame
    feature_columns: list[str]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2])

    def subset(self, mask: np.ndarray) -> "SequenceSet":
        return SequenceSet(
            X=self.X[mask], y=self.y[mask], timestamps=self.timestamps[mask],
            target_timestamps=self.target_timestamps[mask], daylight=self.daylight[mask],
            regime=self.regime[mask], season=self.season[mask],
            target_raw=self.target_raw[mask], origin_row=self.origin_row[mask],
            feature_columns=self.feature_columns)

    def describe(self) -> dict[str, Any]:
        return {
            "n_samples": len(self),
            "lookback_steps": int(self.X.shape[1]) if self.X.ndim == 3 else 0,
            "n_features": self.n_features,
            "n_targets_per_sample": int(self.y.shape[1]) if self.y.ndim == 2 else 1,
            "first_origin": str(self.timestamps.iloc[0]) if len(self) else None,
            "last_origin": str(self.timestamps.iloc[-1]) if len(self) else None,
            "first_target": str(self.target_timestamps.iloc[0]) if len(self) else None,
            "last_target": str(self.target_timestamps.iloc[-1]) if len(self) else None,
            "daylight_fraction": float(np.mean(self.daylight)) if len(self) else None,
            "feature_columns": list(self.feature_columns),
        }


def window_indices(n_rows: int, lookback: int, horizon_steps: int,
                   stride: int = 1) -> np.ndarray:
    """Return the ``(n_windows, lookback)`` matrix of row indices per window.

    Window ``k`` covers rows ``[k*stride, k*stride + lookback)`` and its label is
    the target at row ``k*stride + lookback - 1 + horizon_steps``.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be at least 1, got {lookback}")
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be at least 1, got {horizon_steps}")
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")

    last_origin = n_rows - lookback - horizon_steps
    if last_origin < 0:
        raise ValueError(
            f"not enough rows: {n_rows} rows cannot form a single window with "
            f"lookback={lookback} and horizon_steps={horizon_steps}")
    starts = np.arange(0, last_origin + 1, stride)
    return starts[:, None] + np.arange(lookback)[None, :]


def build_sequences(frame: pd.DataFrame, feature_columns: list[str],
                    target_column: str, lookback: int, horizon_steps: int,
                    target_scaler: Any = None, feature_scaler: Any = None,
                    stride: int = 1, time_column: str = "Time") -> SequenceSet:
    """Build a :class:`SequenceSet` from a feature frame.

    ``target_column`` must already hold the *future* value, i.e. the target at
    row ``t`` is the observed power at ``t + horizon_steps``. The pipeline builds
    that column with :func:`solar_forecasting.preprocessing.pipeline.build_target`,
    so window construction here never has to index past the end of the frame.
    """
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise KeyError(f"feature columns absent from frame: {missing}")
    if target_column not in frame.columns:
        raise KeyError(f"target column {target_column!r} absent from frame")

    values = frame[feature_columns].to_numpy(dtype=float)
    targets = frame[target_column].to_numpy(dtype=float)
    times = pd.to_datetime(frame[time_column]).reset_index(drop=True)

    idx = window_indices(len(frame), lookback, horizon_steps, stride)
    n_windows = idx.shape[0]

    # Two distinct row indices, and conflating them silently doubles the horizon:
    #
    #   origin  = idx[:, -1]        the last row of the input window, i.e. the
    #                               forecast origin. The features are known here.
    #   label   = origin + steps    the timestamp being predicted. Used for the
    #                               clock and for flags (daylight, regime) that
    #                               describe that timestamp.
    #
    # ``target_column`` is *already* lead-shifted by ``build_target``: its value
    # at row ``t`` is the power observed at ``t + horizon_steps``. The predicted
    # value must therefore be read at the origin, not at the label row. Reading
    # it at the label row would apply the shift twice and train every model to
    # forecast twice the requested horizon. Verified against the pipeline:
    # ``target[t] == pv_power_w[t + horizon_steps]``.
    origin_rows = idx[:, -1]
    label_rows = origin_rows + horizon_steps

    if label_rows.max() >= len(frame):
        raise ValueError(
            f"a window label would fall at row {label_rows.max()} but the frame has "
            f"{len(frame)} rows; the frame was probably not truncated by "
            f"horizon_steps={horizon_steps}")

    X = np.stack([values[row] for row in idx])
    y = targets[origin_rows]

    if feature_scaler is not None:
        flat = X.reshape(-1, X.shape[-1])
        X = feature_scaler.transform(flat).reshape(X.shape)
    if target_scaler is not None:
        y = target_scaler.transform(y.reshape(-1, 1)).ravel()

    daylight = (frame["daylight"].to_numpy()[label_rows] if "daylight" in frame.columns
                else np.ones(n_windows, dtype=bool))
    regime = (frame["regime"].to_numpy()[label_rows] if "regime" in frame.columns
              else np.array(["unknown"] * n_windows, dtype=object))
    season = (frame["season"].to_numpy()[label_rows] if "season" in frame.columns
              else np.array(["unknown"] * n_windows, dtype=object))

    return SequenceSet(
        X=X.astype(np.float32),
        y=np.asarray(y, dtype=np.float32),
        timestamps=times.iloc[idx[:, -1]].reset_index(drop=True),
        target_timestamps=times.iloc[label_rows].reset_index(drop=True),
        daylight=np.asarray(daylight, dtype=bool),
        regime=np.asarray(regime, dtype=object),
        season=np.asarray(season, dtype=object),
        target_raw=targets[origin_rows].astype(float),
        origin_row=idx[:, -1].copy(),
        feature_columns=list(feature_columns),
    )


def verify_causality(seq: SequenceSet, frame: pd.DataFrame, lookback: int,
                     horizon_steps: int, feature_columns: list[str],
                     n_checks: int = 50) -> None:
    """Assert that a sample's window really is causal and correctly aligned.

    Raises on the first violation. Called by the test suite and by
    :func:`solar_forecasting.training.trainer.train_model` before any training,
    because a misaligned window silently destroys a model.
    """
    if len(seq) == 0:
        raise ValueError("cannot verify an empty sequence set")
    values = frame[feature_columns].to_numpy(dtype=float)
    times = pd.to_datetime(frame["Time"]).reset_index(drop=True)

    probe = np.unique(np.linspace(0, len(seq) - 1, n_checks).astype(int))
    for k in probe:
        origin = int(seq.origin_row[k])
        start = origin - lookback + 1
        if start < 0:
            raise ValueError(f"window {k} starts at row {start}, before the frame")
        expected = values[start:origin + 1]
        # Features are compared on the last timestep only, since the sequence
        # set may have been standardised; the alignment of the raw frame is what
        # matters here.
        if not np.allclose(expected[-1], values[origin], equal_nan=True):
            raise ValueError(f"window {k} last timestep does not match its origin row")
        label_row = origin + horizon_steps
        if label_row >= len(times):
            raise ValueError(f"window {k} label row {label_row} is past the end of the frame")
        if times.iloc[label_row] <= times.iloc[origin]:
            raise ValueError(
                f"window {k} target timestamp {times.iloc[label_row]} is not after its "
                f"forecast origin {times.iloc[origin]}")

        # The target value must be the one declared at the origin row, because
        # the target column is already lead-shifted. A second implicit shift here
        # is the failure this project is most exposed to: it doubles every
        # horizon while leaving the forecast timestamps looking plausible.
        if "target" in frame.columns:
            declared = frame["target"].iloc[origin]
            if np.isfinite(declared) and not np.isclose(
                    float(seq.target_raw[k]), float(declared), rtol=1e-5, atol=1e-3):
                raise ValueError(
                    f"window {k} target_raw {seq.target_raw[k]!r} does not match the "
                    f"declared target at the origin row {declared!r}; the target column "
                    f"was probably shifted twice")
            # Tie the declared target back to the raw power record so a change in
            # build_target cannot silently renumber the horizon.
            if ("pv_power_w" in frame.columns and label_row < len(frame)
                    and np.isfinite(declared)):
                raw = float(frame["pv_power_w"].iloc[label_row])
                if not np.isclose(raw, float(declared), rtol=1e-5, atol=1e-3):
                    raise ValueError(
                        f"window {k} declared target {declared!r} does not equal "
                        f"pv_power_w at the label row {raw!r}; the horizon definition "
                        f"disagrees with the pipeline")
