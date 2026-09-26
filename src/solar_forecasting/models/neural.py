"""Neural forecasting architectures.

Five models, chosen to cover the progression the literature review traced from
recurrent models through hybrids to attention:

===================  ====================================================
``lstm``             single-layer LSTM, the field's reference architecture
``gru``              gated recurrent unit, fewer parameters than LSTM
``cnn_lstm``         1-D convolution then LSTM: local patterns then sequence
``attention_lstm``   LSTM with additive (Bahdanau) attention over time
``transformer``      encoder-only transformer with sinusoidal positions
===================  ====================================================

All five share one interface: ``(batch, lookback, n_features)`` in,
``(batch,)`` out, predicting the standardised target. The output is a single
step, because the project uses direct multi-horizon forecasting: one model is
trained per horizon rather than one model rolled forward recursively. Recursive
rolling was rejected because its error compounds with step, which would confound
the horizon comparison with error-accumulation artefacts.

Attention weights are exposed by :class:`AttentionLSTM` but are deliberately not
used as explanations anywhere in this project. The literature review found that
attention-weight-as-explanation is a widespread and unjustified practice
(research gap RG-07), and this project's own explainability analysis is
therefore built on SHAP and permutation importance instead.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ForecastNet(nn.Module):
    """Common base: input projection, backbone, regression head.

    Input normalisation and target de-normalisation are handled outside the
    network, by the trainer, so that a model can be inspected and its output
    interpreted in physical units.
    """

    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.1,
                 output_size: int = 1) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config(self) -> dict[str, Any]:
        return {"hidden_size": self.hidden_size, "n_features": self.n_features,
                "output_size": self.output_size,
                "n_parameters": self.n_parameters,
                "parameters_by_layer": {
                    name: int(p.numel()) for name, p in self.named_parameters()
                    if p.requires_grad
                }}


class LSTMForecaster(ForecastNet):
    """Single-layer LSTM with a linear head on the final hidden state."""

    family = "recurrent"
    requires_training = True
    sequence_input = True

    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.1,
                 n_layers: int = 1, **_: Any) -> None:
        super().__init__(n_features, hidden_size, dropout)
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


class GRUForecaster(ForecastNet):
    """Single-layer GRU with a linear head."""

    family = "recurrent"
    requires_training = True
    sequence_input = True

    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.1,
                 n_layers: int = 1, **_: Any) -> None:
        super().__init__(n_features, hidden_size, dropout)
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size,
                         num_layers=n_layers, batch_first=True,
                         dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


class CNNLSTMForecaster(ForecastNet):
    """1-D convolution for local temporal structure, then an LSTM.

    The convolution is causal: padding is applied only on the left of the time
    axis, so output step ``t`` depends on inputs up to and including ``t`` and
    never on future values. A non-causal convolution would leak information
    within the lookback window and inflate every result.
    """

    family = "hybrid"
    requires_training = True
    sequence_input = True

    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.1,
                 n_filters: int = 32, kernel_size: int = 5, **_: Any) -> None:
        super().__init__(n_features, hidden_size, dropout)
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_filters,
                              kernel_size=kernel_size)
        self.norm = nn.BatchNorm1d(n_filters)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=n_filters, hidden_size=hidden_size,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, time, features) -> (batch, features, time) for Conv1d
        h = x.transpose(1, 2)
        h = nn.functional.pad(h, (self.kernel_size - 1, 0))  # causal left padding
        h = self.conv(h)
        h = self.norm(h)
        h = torch.relu(h)
        h = self.pool(h)
        h = self.dropout_conv(h)
        h = h.transpose(1, 2)  # back to (batch, time, filters)
        out, _ = self.lstm(h)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


class AdditiveAttention(nn.Module):
    """Bahdanau-style (additive) attention over the time axis.

    A learned score ``e_t = v^T tanh(W h_t + b)`` is softmaxed over time and used
    to form a context vector. Exposed via :meth:`weights` for analysis of what
    the model attends to, with the explicit caveat that attention weights are
    not causal explanations.
    """

    def __init__(self, hidden_size: int, attention_size: int = 32) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = torch.tanh(self.projection(hidden))
        scores = self.score(energy).squeeze(-1)          # (batch, time)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)  # (batch, hidden)
        return context, weights

    def weights(self, hidden: torch.Tensor) -> torch.Tensor:
        energy = torch.tanh(self.projection(hidden))
        return torch.softmax(self.score(energy).squeeze(-1), dim=1)


class AttentionLSTMForecaster(ForecastNet):
    """LSTM with additive attention, head on the attention-weighted context."""

    family = "hybrid"
    requires_training = True
    sequence_input = True

    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.1,
                 attention_size: int = 32, **_: Any) -> None:
        super().__init__(n_features, hidden_size, dropout)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            batch_first=True)
        self.attention = AdditiveAttention(hidden_size, attention_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)
        self._last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(x)
        context, weights = self.attention(hidden)
        self._last_weights = weights.detach()
        combined = torch.cat([context, hidden[:, -1, :]], dim=-1)
        return self.head(self.dropout(combined)).squeeze(-1)

    def attention_weights(self) -> torch.Tensor | None:
        """Attention weights from the most recent forward pass, or None."""
        return self._last_weights


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Computed in closed form rather than learned, so a model trained on a shorter
    window can be evaluated on a longer one without shape errors, and so the
    encoding carries no information that is not a function of the position.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]].to(x.dtype)


class TransformerForecaster(ForecastNet):
    """Encoder-only transformer with a mean-pooled representation.

    The input projection maps ``n_features`` to ``d_model``. A CLS-free design
    mean-pools the encoded sequence and applies a linear head, which avoids
    introducing a learned token that has no counterpart in the data.
    """

    family = "attention"
    requires_training = True
    sequence_input = True

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1,
                 hidden_size: int | None = None, **_: Any) -> None:
        super().__init__(hidden_size or d_model, dropout)
        self.d_model = d_model
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.positional(self.input_projection(x))
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(self.dropout(pooled)).squeeze(-1)


MODEL_CLASSES: dict[str, type[ForecastNet]] = {
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "cnn_lstm": CNNLSTMForecaster,
    "attention_lstm": AttentionLSTMForecaster,
    "transformer": TransformerForecaster,
}

MODEL_FAMILY: dict[str, str] = {
    "lstm": "recurrent",
    "gru": "recurrent",
    "cnn_lstm": "hybrid",
    "attention_lstm": "hybrid",
    "transformer": "attention",
}


def make_neural_model(name: str, n_features: int, **kwargs: Any) -> ForecastNet:
    """Factory for the neural architectures."""
    key = name.lower()
    if key not in MODEL_CLASSES:
        raise KeyError(f"unknown neural model: {name!r}. "
                       f"Available: {sorted(MODEL_CLASSES)}")
    return MODEL_CLASSES[key](n_features=n_features, **kwargs)


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 256,
                    shuffle: bool = False, seed: int = 42) -> DataLoader:
    """Build a ``DataLoader`` with a seeded generator for reproducibility."""
    dataset = TensorDataset(
        torch.as_tensor(np.asarray(X), dtype=torch.float32),
        torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator if shuffle else None, drop_last=False,
                      num_workers=0)
