"""Model registry: one place where every model is declared and constructed.

Keeping construction in a single registry means the experiment runner, the
training script and the cross-site transfer code cannot drift apart in how they
instantiate a model, and a new model has exactly one integration point.
"""

from __future__ import annotations

from typing import Any

from .baselines import PersistenceModel, SmartPersistenceModel, make_baseline
from .classical import (XGBOOST_AVAILABLE, XGBOOST_IMPORT_ERROR, SklearnRegressorAdapter,
                        make_classical_model)
from .neural import MODEL_FAMILY, ForecastNet, make_neural_model

#: Every model available in the benchmark, grouped by the taxonomy used in the
#: literature review and in the results tables.
REGISTRY: dict[str, dict[str, str]] = {
    # Baselines
    "persistence": {"family": "baseline", "kind": "rule", "requires_training": "no",
                    "sequence_input": "no",
                    "description": "Repeat the most recent observed power."},
    "smart_persistence": {"family": "baseline", "kind": "rule", "requires_training": "no",
                          "sequence_input": "no",
                          "description": "Persistence rescaled by the change in clear-sky "
                                         "reference between the forecast origin and the target."},
    # Classical machine learning
    "linear_regression": {"family": "classical_ml", "kind": "sklearn", "requires_training": "yes",
                          "sequence_input": "yes",
                          "description": "Ordinary least squares on the flattened lookback."},
    "random_forest": {"family": "classical_ml", "kind": "sklearn", "requires_training": "yes",
                      "sequence_input": "yes",
                      "description": "Bagged decision trees on the flattened lookback."},
    "gradient_boosting": {"family": "classical_ml", "kind": "sklearn", "requires_training": "yes",
                          "sequence_input": "yes",
                          "description": "scikit-learn gradient-boosted trees; documented "
                                         "fallback for xgboost."},
    "xgboost": {"family": "classical_ml", "kind": "sklearn", "requires_training": "yes",
                "sequence_input": "yes",
                "description": "Second-order boosted trees with explicit regularisation."},
    "mlp": {"family": "classical_ml", "kind": "sklearn", "requires_training": "yes",
            "sequence_input": "yes",
            "description": "Multi-layer perceptron, the one non-tree neural baseline in the "
                           "classical family."},
    # Recurrent deep learning
    "lstm": {"family": "recurrent", "kind": "torch", "requires_training": "yes",
             "sequence_input": "yes",
             "description": "Single-layer LSTM, the field's reference architecture."},
    "gru": {"family": "recurrent", "kind": "torch", "requires_training": "yes",
            "sequence_input": "yes",
            "description": "Single-layer GRU."},
    # Hybrid deep learning
    "cnn_lstm": {"family": "hybrid", "kind": "torch", "requires_training": "yes",
                 "sequence_input": "yes",
                 "description": "Causal 1-D convolution then LSTM."},
    "attention_lstm": {"family": "hybrid", "kind": "torch", "requires_training": "yes",
                       "sequence_input": "yes",
                       "description": "LSTM with additive attention over the time axis."},
    # Attention / modern architectures
    "transformer": {"family": "attention", "kind": "torch", "requires_training": "yes",
                    "sequence_input": "yes",
                    "description": "Encoder-only transformer with sinusoidal positions and "
                                   "mean-pooled representation."},
}

#: Order used in every table and figure, so the presentation is consistent.
DEFAULT_ORDER: list[str] = [
    "persistence", "smart_persistence",
    "linear_regression", "random_forest", "gradient_boosting", "xgboost",
    "lstm", "gru", "cnn_lstm", "attention_lstm", "transformer",
]

#: Models whose results are the target of the project rather than part of the
#: headline comparison (they exist for the ablation and cost experiments).
SECONDARY_ORDER: list[str] = ["mlp"]


def all_model_names(include_secondary: bool = True) -> list[str]:
    names = list(DEFAULT_ORDER)
    if include_secondary:
        names += SECONDARY_ORDER
    return names


def model_info(name: str) -> dict[str, str]:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}. Available: {sorted(REGISTRY)}")
    return dict(REGISTRY[name])


def build_model(name: str, n_features: int, random_state: int = 42,
                param_grid: dict[str, list[Any]] | None = None,
                **kwargs: Any):
    """Construct a model by name.

    Returns either a rule-based baseline, a :class:`SklearnRegressorAdapter` or a
    :class:`~solar_forecasting.models.neural.ForecastNet`, all of which expose
    ``fit``/``predict`` through the runner's adapter layer.
    """
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown model {name!r}. Available: {sorted(REGISTRY)}")

    kind = REGISTRY[key]["kind"]
    if kind == "rule":
        return make_baseline(key, **kwargs)
    if kind == "sklearn":
        return make_classical_model(key, random_state=random_state,
                                    param_grid=param_grid, **kwargs)
    if kind == "torch":
        return make_neural_model(key, n_features=n_features, **kwargs)
    raise KeyError(f"unsupported model kind: {kind!r}")


def capability_report() -> dict[str, Any]:
    """Environment capabilities that affect which results can be reproduced."""
    import platform
    import sys

    import numpy as np
    import sklearn
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost_available": XGBOOST_AVAILABLE,
        "xgboost_import_error": XGBOOST_IMPORT_ERROR,
        "cuda_available": bool(torch.cuda.is_available()),
        "n_models_registered": len(REGISTRY),
    }
