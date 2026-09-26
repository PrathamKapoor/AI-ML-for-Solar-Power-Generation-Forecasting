"""Classical machine-learning regressors.

Four estimators, chosen to span the space of inductive biases that the literature
review identified as relevant:

``linear_regression``   the interpretable floor; also the reference that the
                        largest classical comparison in the corpus (Markovics and
                        Mayer 2022) uses as its baseline scenario
``random_forest``       bagged trees; an ensemble that cannot extrapolate
``gradient_boosting``   scikit-learn's histogram gradient boosting, the
                        documented fallback when XGBoost is unavailable
``xgboost``             second-order boosting with explicit regularisation

XGBoost is the primary choice for boosted trees. If it cannot be imported, the
runner substitutes :class:`~sklearn.ensemble.HistGradientBoostingRegressor` and
records the substitution in the experiment record, so no result is ever silently
attributed to the wrong estimator.

Every estimator consumes the *flattened* lookback window: the sequence models see
``(lookback, n_features)`` tensors, while the tree and linear models see the
concatenation of all ``lookback`` steps. That keeps the information content
identical across model families, which is what makes the comparison controlled
rather than confounded by input design.
"""

from __future__ import annotations

from typing import Any

import numpy as np

XGBOOST_IMPORT_ERROR: str | None = None

try:  # pragma: no cover - import-time capability check
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    xgb = None  # type: ignore[assignment]

from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from ..features.builder import LAG_PREFIX, ROLLING_PREFIX

FALLBACK_NOTE = (
    "xgboost was not importable, so HistGradientBoostingRegressor (scikit-learn) was used "
    "as the documented substitute. Both are second-order gradient-boosted tree ensembles; "
    "the substitute is reported in the experiment record so no result is attributed to the "
    "wrong estimator."
)


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Reshape ``(n, lookback, n_features)`` into ``(n, lookback * n_features)``."""
    array = np.asarray(X)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D array, got shape {array.shape}")
    n, lookback, n_features = array.shape
    return array.reshape(n, lookback * n_features)


class SklearnRegressorAdapter:
    """Uniform wrapper giving every classical estimator one interface.

    Exposes ``fit``, ``predict``, ``feature_importances_`` and ``describe`` so the
    experiment runner and the explainability module can treat linear regression
    and gradient boosting identically.
    """

    family = "classical_ml"
    requires_training = True
    sequence_input = False

    def __init__(self, name: str, estimator: Any, random_state: int = 42,
                 param_grid: dict[str, list[Any]] | None = None,
                 substitution_note: str | None = None, **_: Any) -> None:
        self.name = name
        self.estimator = estimator
        self.random_state = int(random_state)
        self.param_grid = param_grid or {}
        self.substitution_note = substitution_note
        self.feature_names_: list[str] = []
        self.lookback_: int = 0
        self.best_params_: dict[str, Any] | None = None
        self.search_log_: list[dict[str, Any]] = []

    # -- training ---------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnRegressorAdapter":
        features = flatten_windows(X)
        self.lookback_ = int(np.asarray(X).shape[1]) if np.asarray(X).ndim == 3 else 1
        self.estimator.fit(features, np.asarray(y, dtype=float).ravel())
        return self

    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
        """Select hyperparameters on the validation split, then refit on train.

        The grid is scored on the validation split only, and the winning
        configuration is refitted on the training split only. The test split is
        never passed to this method, so tuning cannot leak test information.
        """
        if not self.param_grid:
            return {"tuned": False, "reason": "no grid configured"}

        from itertools import product

        from sklearn.base import clone
        from sklearn.metrics import mean_absolute_error

        X_tr = flatten_windows(X_train)
        y_tr = np.asarray(y_train, dtype=float).ravel()
        X_v = flatten_windows(X_val)
        y_v = np.asarray(y_val, dtype=float).ravel()

        keys = list(self.param_grid)
        best_score = float("inf")
        best_params: dict[str, Any] = {}
        self.search_log_ = []

        for values in product(*(self.param_grid[k] for k in keys)):
            candidate = dict(zip(keys, values))
            estimator = clone(self.estimator).set_params(**candidate)
            estimator.fit(X_tr, y_tr)
            score = float(mean_absolute_error(y_v, estimator.predict(X_v)))
            self.search_log_.append({"params": dict(candidate), "val_mae_w": score})
            if score < best_score:
                best_score, best_params = score, candidate

        self.estimator.set_params(**best_params).fit(X_tr, y_tr)
        self.best_params_ = best_params
        return {"tuned": True, "best_params": best_params,
                "best_val_mae_w": best_score, "n_candidates": len(self.search_log_),
                "selection_split": "validation"}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(flatten_windows(X)), dtype=float)

    # -- introspection ------------------------------------------------------ #
    @property
    def feature_importances_(self) -> np.ndarray | None:
        if hasattr(self.estimator, "feature_importances_"):
            values = np.asarray(self.estimator.feature_importances_, dtype=float)
            return values if values.ndim == 1 else None
        if hasattr(self.estimator, "coef_"):
            coef = np.asarray(self.estimator.coef_, dtype=float).ravel()
            return np.abs(coef)
        return None

    @property
    def n_parameters(self) -> int:
        """A size proxy used by the computational-cost experiment.

        For linear models the exact count of fitted coefficients is reported. For
        tree ensembles the tree count and depth are reported, which is a proxy
        rather than an exact parameter count, and is labelled as such in the cost
        table.
        """
        coef = getattr(self.estimator, "coef_", None)
        if coef is not None:
            intercept = getattr(self.estimator, "intercept_", np.zeros(1))
            return int(np.asarray(coef).size + np.asarray(intercept).size)
        params = self.estimator.get_params() if hasattr(self.estimator, "get_params") else {}
        n_estimators = params.get("n_estimators") or params.get("n_estimators_")
        if n_estimators is not None:
            return int(n_estimators)
        return 0

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "estimator": type(self.estimator).__name__,
            "requires_training": self.requires_training,
            "substitution_note": self.substitution_note,
            "best_params": self.best_params_,
            "random_state": self.random_state,
        }


def make_classical_model(name: str, random_state: int = 42,
                         param_grid: dict[str, list[Any]] | None = None,
                         **kwargs: Any) -> SklearnRegressorAdapter:
    """Factory for the classical estimators."""
    key = name.lower()

    if key in {"linear_regression", "linear", "lr"}:
        return SklearnRegressorAdapter("linear_regression",
                                       LinearRegression(n_jobs=None),
                                       random_state, param_grid, **kwargs)

    if key in {"random_forest", "rf", "randomforest"}:
        estimator = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            max_features=kwargs.get("max_features", 0.5),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=random_state,
        )
        return SklearnRegressorAdapter("random_forest", estimator, random_state,
                                       param_grid, **kwargs)

    if key in {"gradient_boosting", "gbr", "sklearn_gb"}:
        # The histogram implementation is the default. It is the same
        # second-order gradient-boosted tree ensemble as GradientBoostingRegressor
        # but bins the features once instead of sorting them at every split,
        # which on the 768-column flattened window is the difference between
        # minutes and hours. GradientBoostingRegressor remains reachable as
        # 'sklearn_gradient_boosting' for comparison.
        estimator = HistGradientBoostingRegressor(
            max_iter=kwargs.get("n_estimators", 300),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_depth=kwargs.get("max_depth", 3),
            max_leaf_nodes=kwargs.get("max_leaf_nodes", 31),
            min_samples_leaf=kwargs.get("min_samples_leaf", 20),
            l2_regularization=kwargs.get("l2_regularization", 0.0),
            early_stopping=kwargs.get("early_stopping", False),
            random_state=random_state,
        )
        return SklearnRegressorAdapter("gradient_boosting", estimator, random_state,
                                       param_grid, **kwargs)

    if key in {"sklearn_gradient_boosting", "sklearn_gbr"}:
        estimator = GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_depth=kwargs.get("max_depth", 3),
            subsample=kwargs.get("subsample", 1.0),
            random_state=random_state,
        )
        return SklearnRegressorAdapter("sklearn_gradient_boosting", estimator,
                                       random_state, param_grid, **kwargs)

    if key in {"hist_gradient_boosting", "hgb"}:
        estimator = HistGradientBoostingRegressor(
            max_iter=kwargs.get("max_iter", 400),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_leaf_nodes=kwargs.get("max_leaf_nodes", 31),
            min_samples_leaf=kwargs.get("min_samples_leaf", 20),
            l2_regularization=kwargs.get("l2_regularization", 0.0),
            early_stopping=kwargs.get("early_stopping", False),
            random_state=random_state,
        )
        return SklearnRegressorAdapter("hist_gradient_boosting", estimator, random_state,
                                       param_grid, **kwargs)

    if key in {"xgboost", "xgb"}:
        if not XGBOOST_AVAILABLE:
            estimator = HistGradientBoostingRegressor(
                max_iter=kwargs.get("n_estimators", 400),
                learning_rate=kwargs.get("learning_rate", 0.05),
                max_leaf_nodes=kwargs.get("max_leaf_nodes", 31),
                early_stopping=False,
                random_state=random_state,
            )
            return SklearnRegressorAdapter("xgboost", estimator, random_state, param_grid,
                                          substitution_note=FALLBACK_NOTE + f" ({XGBOOST_IMPORT_ERROR})",
                                          **kwargs)
        estimator = xgb.XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 600),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_depth=kwargs.get("max_depth", 6),
            subsample=kwargs.get("subsample", 0.9),
            colsample_bytree=kwargs.get("colsample_bytree", 0.9),
            reg_lambda=kwargs.get("reg_lambda", 1.0),
            reg_alpha=kwargs.get("reg_alpha", 0.0),
            min_child_weight=kwargs.get("min_child_weight", 1),
            objective=kwargs.get("objective", "reg:squarederror"),
            tree_method=kwargs.get("tree_method", "hist"),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=random_state,
            verbosity=0,
        )
        return SklearnRegressorAdapter("xgboost", estimator, random_state, param_grid, **kwargs)

    if key in {"mlp", "mlp_regressor"}:
        estimator = MLPRegressor(
            hidden_layer_sizes=tuple(kwargs.get("hidden_layer_sizes", (128, 64))),
            learning_rate_init=kwargs.get("learning_rate_init", 1e-3),
            max_iter=kwargs.get("max_iter", 400),
            early_stopping=kwargs.get("early_stopping", True),
            n_iter_no_change=kwargs.get("n_iter_no_change", 15),
            validation_fraction=kwargs.get("validation_fraction", 0.1),
            random_state=random_state,
        )
        return SklearnRegressorAdapter("mlp", estimator, random_state, param_grid, **kwargs)

    raise KeyError(f"unknown classical model: {name!r}")


def flattened_feature_names(feature_columns: list[str], lookback: int) -> list[str]:
    """Names for the flattened window, in the exact order ``flatten_windows`` produces.

    ``flatten_windows`` reshapes ``(lookback, n_features)`` to row-major order,
    so step 0's features come first, then step 1's, and so on. The name records
    the offset of each step from the forecast origin, counting backwards.
    """
    if lookback <= 1:
        return list(feature_columns)
    return [f"{column}@t-{lookback - 1 - j}"
            for j in range(lookback)
            for column in feature_columns]


def dominant_feature_names(feature_columns: list[str], lookback: int) -> list[str]:
    """Collapse the flattened window to one name per original feature.

    The importance of a feature is summed across its ``lookback`` appearances,
    so the collapsed name is the original column name. This is what the
    feature-importance table reports, because a reader wants to know whether
    *irradiance* matters, not which of 96 lagged irradiance values matters most.
    """
    del lookback
    return list(feature_columns)


def is_lag_or_rolling(name: str) -> bool:
    return name.startswith(LAG_PREFIX) or name.startswith(ROLLING_PREFIX)
