"""Chronological splitting and train-only scaling.

Two leakage guards live here and are the reason this module exists as its own
unit:

1. **Splits are time-ordered, never shuffled.** Each split is a contiguous
   interval with explicit configured boundaries, and the boundaries are asserted
   to be non-overlapping and monotonically increasing.

2. **Scalers are fitted on the training split only.** The fitted statistics are
   then applied unchanged to validation and test. A test that scaler parameters
   match the training fit exactly is included in the test suite, because this is
   the single most common silent leak in time-series forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitBoundaries:
    """Explicit, resolved split boundaries."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def validate(self) -> None:
        ordered = [
            ("train_start", self.train_start), ("train_end", self.train_end),
            ("val_start", self.val_start), ("val_end", self.val_end),
            ("test_start", self.test_start), ("test_end", self.test_end),
        ]
        for (name_a, a), (name_b, b) in zip(ordered, ordered[1:]):
            if pd.isna(a) or pd.isna(b):
                raise ValueError(f"split boundary {name_a} or {name_b} is NaT")
            if b < a:
                raise ValueError(f"split boundaries are not monotonic: {name_a}={a} > {name_b}={b}")
        for earlier, later in (("train", "val"), ("val", "test"), ("train", "test")):
            start = getattr(self, f"{earlier}_start")
            end = getattr(self, f"{earlier}_end")
            other_start = getattr(self, f"{later}_start")
            other_end = getattr(self, f"{later}_end")
            if end >= other_start:
                raise ValueError(
                    f"{earlier} split ends at {end} which overlaps {later} split "
                    f"starting {other_start}; splits must be disjoint")
            _ = start, other_end

    def to_dict(self) -> dict[str, str]:
        return {k: str(getattr(self, k)) for k in (
            "train_start", "train_end", "val_start", "val_end", "test_start", "test_end")}


def boundaries_from_config(config_section: dict[str, Any]) -> SplitBoundaries:
    """Build and validate :class:`SplitBoundaries` from the YAML configuration."""
    strategy = config_section.get("strategy", "chronological")
    if strategy != "chronological":
        raise ValueError(
            f"only the 'chronological' split strategy is supported, got {strategy!r}. "
            f"Random splitting of a time series is a leakage bug, not a configuration option.")

    boundaries = SplitBoundaries(
        train_start=pd.Timestamp(config_section["train_start"]),
        train_end=pd.Timestamp(config_section["train_end"]),
        val_start=pd.Timestamp(config_section["val_start"]),
        val_end=pd.Timestamp(config_section["val_end"]),
        test_start=pd.Timestamp(config_section["test_start"]),
        test_end=pd.Timestamp(config_section["test_end"]),
    )
    boundaries.validate()
    return boundaries


@dataclass
class SplitResult:
    """The three disjoint partitions plus the audit trail."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    boundaries: SplitBoundaries
    audit: dict[str, Any] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def chronological_split(frame: pd.DataFrame, boundaries: SplitBoundaries,
                        time_column: str = "Time") -> SplitResult:
    """Split a time-indexed frame into train / validation / test.

    Rows outside the configured window are dropped and reported, so a
    configuration change can never silently include data the author did not
    intend to use.
    """
    boundaries.validate()
    if time_column not in frame.columns:
        raise KeyError(f"time column {time_column!r} not present in frame")

    times = pd.to_datetime(frame[time_column])
    in_window = (times >= boundaries.train_start) & (times <= boundaries.test_end)
    dropped_outside = int((~in_window).sum())
    frame = frame.loc[in_window].reset_index(drop=True)
    # The mask above is positional with respect to the *unfiltered* frame, so the
    # timestamps have to be recomputed after filtering. Reusing the original
    # series would reindex it against the shortened frame and silently shift
    # every split boundary, which is a leakage bug rather than a crash.
    times = pd.to_datetime(frame[time_column])

    train = frame[(times >= boundaries.train_start) & (times <= boundaries.train_end)]
    val = frame[(times >= boundaries.val_start) & (times <= boundaries.val_end)]
    test = frame[(times >= boundaries.test_start) & (times <= boundaries.test_end)]

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    audit = {
        "strategy": "chronological",
        "boundaries": boundaries.to_dict(),
        "n_rows_total": int(len(frame)),
        "n_dropped_outside_window": dropped_outside,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "fractions": {
            "train": round(len(train) / max(len(frame), 1), 4),
            "val": round(len(val) / max(len(frame), 1), 4),
            "test": round(len(test) / max(len(frame), 1), 4),
        },
        "date_ranges": {
            "train": [str(train[time_column].min()), str(train[time_column].max())] if len(train) else None,
            "val": [str(val[time_column].min()), str(val[time_column].max())] if len(val) else None,
            "test": [str(test[time_column].min()), str(test[time_column].max())] if len(test) else None,
        },
        "disjoint": True,
        "max_train_time": str(train[time_column].max()) if len(train) else None,
        "min_val_time": str(val[time_column].min()) if len(val) else None,
        "max_val_time": str(val[time_column].max()) if len(val) else None,
        "min_test_time": str(test[time_column].min()) if len(test) else None,
    }
    if audit["max_train_time"] and audit["min_val_time"]:
        audit["train_val_gap"] = str(
            pd.Timestamp(audit["min_val_time"]) - pd.Timestamp(audit["max_train_time"]))

    return SplitResult(train=train, val=val, test=test, boundaries=boundaries, audit=audit)


@dataclass
class ScalingBundle:
    """Feature and target scalers fitted on the training split only."""

    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: list[str]
    fitted_on: dict[str, Any]

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        return self.feature_scaler.transform(frame[self.feature_columns].to_numpy(dtype=float))

    def transform_target(self, values: np.ndarray | pd.Series) -> np.ndarray:
        return self.target_scaler.transform(np.asarray(values, dtype=float).reshape(-1, 1)).ravel()

    def inverse_transform_target(self, values: np.ndarray | pd.Series) -> np.ndarray:
        return self.target_scaler.inverse_transform(
            np.asarray(values, dtype=float).reshape(-1, 1)).ravel()

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "feature_mean": self.feature_scaler.mean_.tolist(),
            "feature_scale": self.feature_scaler.scale_.tolist(),
            "target_mean": float(self.target_scaler.mean_[0]),
            "target_scale": float(self.target_scaler.scale_[0]),
            "fitted_on": self.fitted_on,
        }


def fit_scalers(train_frame: pd.DataFrame, feature_columns: list[str],
                target_column: str,
                train_end: pd.Timestamp | None = None) -> ScalingBundle:
    """Fit feature and target scalers on the training split.

    Raises if a feature has zero variance in training, because the resulting
    scaler would divide by zero and silently produce inf/NaN at transform time.
    """
    missing = [c for c in feature_columns if c not in train_frame.columns]
    if missing:
        raise KeyError(f"feature columns absent from training frame: {missing}")

    features = train_frame[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        n_bad = int((~np.isfinite(features)).sum())
        raise ValueError(
            f"{n_bad} non-finite values in training features; resolve preprocessing before scaling")

    zero_variance = [feature_columns[i] for i in range(features.shape[1])
                     if float(np.std(features[:, i])) == 0.0]
    if zero_variance:
        raise ValueError(
            f"features with zero variance in the training split cannot be scaled: {zero_variance}")

    feature_scaler = StandardScaler().fit(features)

    target = train_frame[target_column].to_numpy(dtype=float)
    target_scaler = StandardScaler().fit(target.reshape(-1, 1))

    return ScalingBundle(
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=list(feature_columns),
        fitted_on={
            "n_train_rows": int(len(train_frame)),
            "train_start": str(train_frame["Time"].min()),
            "train_end": str(train_frame["Time"].max()),
            "configured_train_end": str(train_end) if train_end is not None else None,
            "n_features": len(feature_columns),
            "note": "fitted on training rows only; applied unchanged to validation and test",
        },
    )
