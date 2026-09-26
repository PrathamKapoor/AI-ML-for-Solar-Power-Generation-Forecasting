"""Training loop with early stopping, timing and full provenance capture.

The training protocol is fixed across every neural model so that the
computational-cost comparison is meaningful:

* loss: mean absolute error on the **standardised** target, which makes the loss
  scale comparable across horizons and prevents a model from optimising a metric
  that is not reported;
* optimiser: AdamW with weight decay, the same initial learning rate family for
  every architecture;
* early stopping on validation MAE in watts with a patience, restoring the best
  weights rather than the last weights, so early stopping cannot silently return
  an under-trained model;
* timing: wall-clock training duration and inference latency, both measured on
  the same hardware with the thread count pinned in
  :func:`solar_forecasting.config.set_thread_env`.

No hyperparameter is ever selected on the test split. The validation split is
the only selection signal, and the test split is touched exactly once, after
training is complete.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ..models.neural import ForecastNet, make_dataloader
from .sequences import SequenceSet, verify_causality


@dataclass
class TrainingConfig:
    """All training hyperparameters for one run."""

    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    min_delta: float = 0.0
    grad_clip: float = 1.0
    device: str = "cpu"
    verbose: bool = False
    shuffle: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingResult:
    """Everything needed to reproduce and audit one training run."""

    model: ForecastNet
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_mae_w: float = float("nan")
    epochs_run: int = 0
    stopped_early: bool = False
    train_seconds: float = 0.0
    inference_seconds_total: float = 0.0
    n_parameters: int = 0
    device: str = "cpu"
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_mae_w": self.best_val_mae_w,
            "stopped_early": self.stopped_early,
            "train_seconds": round(self.train_seconds, 3),
            "n_parameters": self.n_parameters,
            "device": self.device,
            "config": self.config,
        }


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def train_model(model: ForecastNet, train: SequenceSet, val: SequenceSet,
                config: TrainingConfig,
                target_scale: float = 1.0,
                verify: Callable[[], None] | None = None) -> TrainingResult:
    """Train one model with early stopping on validation MAE.

    Parameters
    ----------
    target_scale
        Standard deviation of the fitted target scaler. Used to express the
        training and validation losses in watts, which is the unit every reported
        metric uses.
    verify
        Zero-argument callable that asserts the sequence windows are causal and
        correctly aligned. Invoked before the first optimiser step, because a
        misaligned window produces a model that looks trained but is not.
    """
    if verify is not None:
        verify()

    device = resolve_device(config.device)
    model = model.to(device)

    criterion = nn.L1Loss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    train_loader = make_dataloader(train.X, train.y, config.batch_size,
                                   shuffle=config.shuffle)
    val_loader = make_dataloader(val.X, val.y, config.batch_size, shuffle=False)

    scale = float(target_scale) if target_scale else 1.0

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    stopped_early = False

    start = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            running += float(loss.detach()) * batch_x.shape[0]
            n_batches += 1

        train_loss = running / max(len(train), 1) * scale

        val_loss, n_val = predict_with_loss(model, val_loader, criterion, device)
        val_mae_w = val_loss * scale
        _ = n_val

        history.append({
            "epoch": epoch,
            "train_loss_scaled": running / max(n_batches, 1),
            "train_mae_w": train_loss,
            "val_mae_w": val_mae_w,
        })
        if config.verbose:
            print(f"    epoch {epoch:>3}  train MAE {train_loss:>9.1f} W  "
                  f"val MAE {val_mae_w:>9.1f} W")

        if val_mae_w < best_val - config.min_delta:
            best_val = val_mae_w
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_early = True
                break

    train_seconds = time.perf_counter() - start

    # Restoring the best weights, not the last, is what makes early stopping
    # safe: the returned model is the one that performed best on validation.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_mae_w=best_val,
        epochs_run=len(history),
        stopped_early=stopped_early,
        train_seconds=train_seconds,
        n_parameters=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        device=str(device),
        config=config.to_dict(),
    )


@torch.no_grad()
def predict_with_loss(model: ForecastNet, loader, criterion, device
                      ) -> tuple[float, int]:
    """Mean loss and sample count over a loader, in evaluation mode."""
    model.eval()
    total = 0.0
    n = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        loss = criterion(model(batch_x), batch_y)
        total += float(loss) * batch_x.shape[0]
        n += batch_x.shape[0]
    return total / max(n, 1), n


@torch.no_grad()
def predict_neural(model: ForecastNet, seq: SequenceSet, batch_size: int = 512,
                   device: str = "cpu") -> tuple[np.ndarray, float]:
    """Predict and return ``(predictions_scaled, inference_seconds)``.

    The reported latency is per window in milliseconds, which is the quantity
    that matters operationally: a forecaster must emit a value for every
    timestamp at the data cadence.
    """
    resolved = resolve_device(device)
    model = model.to(resolved)
    model.eval()
    loader = make_dataloader(seq.X, seq.y, batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    start = time.perf_counter()
    for batch_x, _ in loader:
        batch_x = batch_x.to(resolved)
        outputs.append(model(batch_x).cpu().numpy())
    elapsed = time.perf_counter() - start
    predictions = np.concatenate(outputs) if outputs else np.zeros(0)
    per_window_ms = (elapsed / max(len(seq), 1)) * 1000.0
    return predictions, per_window_ms


def training_curve_frame(result: TrainingResult) -> Any:
    """History as a DataFrame, for the training-curve figures and tables."""
    import pandas as pd

    return pd.DataFrame(result.history)
