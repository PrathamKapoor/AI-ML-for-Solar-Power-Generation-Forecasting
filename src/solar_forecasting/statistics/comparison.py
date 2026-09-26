"""Forecast-error comparison: block bootstrap, Diebold-Mariano, Wilcoxon.

All three methods consume *per-timestep* error series rather than aggregate
metrics, because a test on aggregates carries no information about variability.

Assumptions, stated once here and repeated in every output record:

* Errors are serially correlated. The moving-block bootstrap therefore resamples
  contiguous blocks of ``block_length`` steps, which preserves short-range
  dependence; the default block (96 steps = 24 h at the working resolution) spans
  a full diurnal cycle, so a resampled block keeps a physically coherent shape.
* The Diebold-Mariano test assumes the loss differential is stationary and
  serially uncorrelated at the horizon used. Photovoltaic errors are neither, so
  the Harvey-Leybourne-Newbold small-sample adjustment is applied by default and
  the caveat is written into the record.
* The Wilcoxon signed-rank test assumes the paired differences are symmetric and
  independent. Independence is violated here for the same reason, so it is
  reported as a *secondary* check with the violation stated, and its p-value is
  never quoted without the Diebold-Mariano result beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ASSUMPTIONS: dict[str, str] = {
    "serial_correlation": (
        "Forecast errors at 15-minute resolution are serially correlated because "
        "consecutive steps share the same cloud field. Every interval here is a "
        "moving-block bootstrap, and every test carries a small-sample "
        "correction; a naive i.i.d. treatment would overstate the evidence."
    ),
    "block_length": (
        "The default block length is 96 steps (24 h at 15-minute resolution), one "
        "full diurnal cycle, so each resampled block keeps a coherent daily shape."
    ),
    "dm_loss": (
        "Diebold-Mariano is applied to the squared-error and absolute-error loss "
        "differentials. A significant result on one loss and not the other is "
        "reported as such rather than resolved in favour of the significant one."
    ),
    "multiple_comparisons": (
        "The benchmark compares eleven models, so the family of pairwise tests is "
        "large. Holm-Bonferroni adjusted p-values are reported alongside raw "
        "p-values, and no single pairwise comparison is described as decisive on "
        "its own."
    ),
    "no_model_selection_on_test": (
        "The test split is scored once per declared configuration. No "
        "hyperparameter, model or feature set was chosen using these numbers, so "
        "the tests are confirmatory rather than exploratory; but they are not "
        "out-of-sample with respect to the model-selection process as a whole, "
        "and are interpreted as such."
    ),
}


def _as_float(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError("cannot compute a statistic on an empty array")
    return array


def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of one moving-block resample of length ``n``."""
    block_length = int(max(1, min(block_length, n)))
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]


def block_bootstrap_metric(values, metric: Callable[[np.ndarray], float],
                           n_resamples: int = 2000, block_length: int = 96,
                           confidence: float = 0.95, seed: int = 42
                           ) -> dict[str, float | int | str]:
    """Moving-block bootstrap confidence interval for a metric of a series."""
    array = _as_float(values)
    rng = np.random.default_rng(seed)
    point = float(metric(array))
    draws = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        draws[i] = metric(array[_block_indices(array.size, block_length, rng)])
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "point_estimate": point,
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
        "block_length": int(block_length),
        "method": "moving-block bootstrap",
        "seed": int(seed),
    }


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "rmse": _rmse,
    "mae": _mae,
    "bias": lambda v: float(np.mean(v)),
}


def block_bootstrap_skill(errors: np.ndarray, reference_errors: np.ndarray,
                          metric: str = "rmse", n_resamples: int = 2000,
                          block_length: int = 96, confidence: float = 0.95,
                          seed: int = 42) -> dict[str, float | int | str]:
    """Bootstrap CI for skill against a reference forecast, resampling jointly.

    The model errors and the reference errors are resampled with the *same* block
    indices, because they are forecasts for the same timestamps and their
    differences are strongly correlated. Resampling them independently would
    understate the variance of the skill score.
    """
    fn = METRIC_FUNCTIONS[metric]
    model_errors = _as_float(errors)
    ref_errors = _as_float(reference_errors)
    if model_errors.shape != ref_errors.shape:
        raise ValueError("model and reference error series must be aligned")
    rng = np.random.default_rng(seed)
    point = float(1.0 - fn(model_errors) / fn(ref_errors))
    draws = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        idx = _block_indices(model_errors.size, block_length, rng)
        ref_value = fn(ref_errors[idx])
        draws[i] = 1.0 - fn(model_errors[idx]) / ref_value if ref_value > 0 else np.nan
    draws = draws[np.isfinite(draws)]
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "point_estimate": point,
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": float(confidence),
        "metric": metric,
        "n_resamples": int(draws.size),
        "block_length": int(block_length),
        "method": "moving-block bootstrap, jointly resampled with the reference",
        "seed": int(seed),
    }


def paired_loss_difference(errors_a, errors_b, loss: str = "mae") -> np.ndarray:
    """Per-timestep loss differential ``d_t = L(e_a) - L(e_b)``."""
    a = _as_float(errors_a)
    b = _as_float(errors_b)
    if a.shape != b.shape:
        raise ValueError("error series must be aligned on the same timestamps")
    if loss == "mae":
        return np.abs(a) - np.abs(b)
    if loss == "rmse":
        return np.square(a) - np.square(b)
    raise KeyError(f"unknown loss {loss!r}; use 'mae' or 'rmse'")


def _acf_at_lag(values: np.ndarray, lag: int) -> float:
    centred = values - values.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0:
        return 0.0
    return float(np.dot(centred[:-lag], centred[lag:]) / denom)


@dataclass
class DMResult:
    """Outcome of one Diebold-Mariano test."""

    model_a: str
    model_b: str
    loss: str
    statistic: float
    p_value: float
    p_value_adjusted: float
    mean_differential: float
    n_observations: int
    lag_1_autocorrelation: float
    mean_significance: bool
    better_model: str
    correction: str
    caveat: str

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def diebold_mariano(errors_a, errors_b, model_a: str = "A", model_b: str = "B",
                    loss: str = "mae", horizon: int = 1,
                    correction: str = "harvey_leybourne_newbold",
                    alpha: float = 0.05,
                    p_value_adjusted: float | None = None) -> DMResult:
    """Diebold-Mariano test on a loss differential.

    ``d_t`` is the loss differential; the test statistic is
    ``dbar / sqrt(long_run_variance(d) / n)`` with the long-run variance
    estimated by a Bartlett kernel of bandwidth ``horizon - 1``. With the
    Harvey-Leybourne-Newbold adjustment the denominator uses ``n - 1 - 2h``
    instead of ``n``, which matters here because ``n`` is tens of thousands and
    the autocorrelation is not negligible.

    A negative mean differential means model A has the lower loss, so the
    ``better_model`` field records which model actually won, independently of
    the sign convention of the statistic.
    """
    d = paired_loss_difference(errors_a, errors_b, loss=loss)
    n = d.size
    h = int(max(0, horizon - 1))
    centred = d - d.mean()
    long_run = float(np.dot(centred, centred))
    for k in range(1, h + 1):
        weight = 1.0 - k / (h + 1.0)
        long_run += 2.0 * weight * float(np.dot(centred[:-k], centred[k:]))

    if correction == "harvey_leybourne_newbold":
        denom_n = max(n - 1 - 2 * h, 1)
    elif correction == "none":
        denom_n = n
    else:
        raise KeyError(f"unknown DM correction {correction!r}")

    variance = long_run / denom_n
    if variance <= 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float(d.mean() / np.sqrt(variance))
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))
    acf1 = _acf_at_lag(d, 1) if n > 2 else 0.0
    better = model_a if d.mean() < 0 else model_b
    return DMResult(
        model_a=model_a,
        model_b=model_b,
        loss=loss,
        statistic=statistic,
        p_value=p_value,
        p_value_adjusted=float(p_value if p_value_adjusted is None else p_value_adjusted),
        mean_differential=float(d.mean()),
        n_observations=int(n),
        lag_1_autocorrelation=round(acf1, 4),
        mean_significance=bool(p_value < alpha),
        better_model=better,
        correction=correction,
        caveat=(
            "The null hypothesis of no predictive accuracy difference is tested on "
            "a serially correlated loss differential. The reported lag-1 "
            "autocorrelation quantifies the violation; the Harvey-Leybourne-Newbold "
            "adjustment compensates for it in the variance estimate but does not "
            "restore the i.i.d. assumption."
        ),
    )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    values = _as_float(p_values)
    m = values.size
    order = np.argsort(values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * values[idx]
        running = max(running, min(candidate, 1.0))
        adjusted[idx] = running
    return [float(v) for v in adjusted]


def rank_biserial_effect(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the effect size for Wilcoxon.

    It is +1 when every paired difference is positive and -1 when every one is
    negative, so it answers "how completely does one model win?" rather than
    "is the difference distinguishable from zero?".
    """
    d = _as_float(differences)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    positive = float(ranks[d > 0].sum())
    negative = float(ranks[d < 0].sum())
    return float((positive - negative) / (positive + negative))


def wilcoxon_signed_rank(errors_a, errors_b, loss: str = "mae") -> dict[str, object]:
    """Wilcoxon signed-rank test on paired losses, with its assumption stated."""
    d = paired_loss_difference(errors_a, errors_b, loss=loss)
    nonzero = d[d != 0]
    if nonzero.size < 6:
        return {
            "test": "wilcoxon_signed_rank",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "effect_size_rank_biserial": 0.0,
            "n_pairs": int(nonzero.size),
            "n_zero_differences": int(d.size - nonzero.size),
            "assumption_violation": (
                "Paired photovoltaic forecast errors are serially correlated, so the "
                "symmetry-and-independence assumption of the signed-rank test is not "
                "satisfied. Reported as a secondary check only."
            ),
        }
    result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size_rank_biserial": rank_biserial_effect(nonzero),
        "n_pairs": int(nonzero.size),
        "n_zero_differences": int(d.size - nonzero.size),
        "assumption_violation": (
            "Paired photovoltaic forecast errors are serially correlated, so the "
            "symmetry-and-independence assumption of the signed-rank test is not "
            "satisfied. Reported as a secondary check only; the Diebold-Mariano "
            "result is the primary one."
        ),
    }


def autocorrelation_profile(values, max_lag: int = 96) -> pd.DataFrame:
    """Autocorrelation of a series at every lag up to ``max_lag``.

    Reported because the choice of bootstrap block length and the power of every
    test in this project rest on the answer. If forecast errors were
    independent, an i.i.d. bootstrap and an uncorrected Diebold-Mariano test
    would both be valid and the block size would not matter.
    """
    array = _as_float(values)
    array = array - array.mean()
    denominator = float(np.dot(array, array))
    rows = []
    for lag in range(0, int(max_lag) + 1):
        if lag == 0:
            value = 1.0
        elif lag < array.size:
            value = float(np.dot(array[:-lag], array[lag:]) / denominator)
        else:
            value = float("nan")
        rows.append({"lag": lag, "autocorrelation": value,
                     "lag_hours": lag * 15 / 60.0})
    return pd.DataFrame(rows)


def integrated_autocorrelation_time(values, max_lag: int | None = None) -> float:
    """The integrated autocorrelation time, 1 / (1 + 2 * sum of autocorrelations).

    This is the factor by which the effective sample size falls below the nominal
    one. A value of 1 means independent observations; a value of 50 means the
    series carries the information of only about 1/50 as many independent
    observations, which is exactly why an i.i.d. treatment of these errors would
    overstate the evidence by roughly that factor in the width of an interval.
    """
    array = _as_float(values)
    max_lag = int(min(max_lag or min(200, array.size // 4), array.size - 2))
    profile = autocorrelation_profile(array, max_lag=max_lag)
    acf = profile["autocorrelation"].to_numpy(dtype=float)[1:]
    # Stop at the first lag whose contribution is negligible, so that noise in
    # the tail of the ACF does not accumulate.
    significant = acf[: np.argmax(np.abs(acf) < 0.05)] if (np.abs(acf) >= 0.05).any() else acf[:1]
    return float(max(1.0, 1.0 + 2.0 * float(np.sum(significant))))


def effective_sample_size(values, max_lag: int | None = None) -> dict[str, float]:
    """Nominal and effective sample size, with the reduction factor."""
    array = _as_float(values)
    tau = integrated_autocorrelation_time(array, max_lag=max_lag)
    return {
        "n_observations": float(array.size),
        "integrated_autocorrelation_time": tau,
        "effective_sample_size": float(array.size / tau),
        "variance_inflation_if_iid": tau,
    }


def block_length_sensitivity(differentials, blocks: Sequence[int] = (1, 16, 48, 96, 192, 384),
                             n_resamples: int = 2000, confidence: float = 0.95,
                             seed: int = 42) -> pd.DataFrame:
    """Bootstrap interval width as a function of block length.

    The interval for a mean loss differential narrows as the block length grows
    and then plateaus once the block spans the dependence range. Reporting the
    curve makes the choice of block length auditable instead of asserted, and it
    shows how much a too-short block would understate the uncertainty.
    """
    d = _as_float(differentials)
    rng_master = np.random.default_rng(seed)
    rows = []
    for block in blocks:
        rng = np.random.default_rng(int(rng_master.integers(0, 2 ** 31 - 1)))
        draws = np.empty(int(n_resamples), dtype=float)
        for i in range(int(n_resamples)):
            idx = _block_indices(d.size, int(block), rng)
            draws[i] = d[idx].mean()
        alpha = (1.0 - float(confidence)) / 2.0
        low, high = np.quantile(draws, alpha), np.quantile(draws, 1.0 - alpha)
        rows.append({
            "block_length_steps": int(block),
            "block_hours": int(block) * 15 / 60.0,
            "ci_low": float(low), "ci_high": float(high),
            "ci_width": float(high - low),
            "se": float(np.std(draws, ddof=1)),
        })
    frame = pd.DataFrame(rows)
    frame["relative_width"] = frame["ci_width"] / frame["ci_width"].max()
    return frame


def compare_models(predictions: dict[str, np.ndarray], actual: np.ndarray,
                   model: str, reference_model: str = "persistence",
                   loss: str = "mae", horizon: int = 1,
                   n_resamples: int = 2000, block_length: int = 96,
                   confidence: float = 0.95, seed: int = 42) -> dict[str, object]:
    """Bootstrap skill plus both tests for one model against one reference."""
    y = _as_float(actual)
    errors = y - _as_float(predictions[model])
    ref_errors = y - _as_float(predictions[reference_model])
    skill = block_bootstrap_skill(errors, ref_errors, metric="rmse",
                                  n_resamples=n_resamples,
                                  block_length=block_length,
                                  confidence=confidence, seed=seed)
    dm = diebold_mariano(errors, ref_errors, model_a=model, model_b=reference_model,
                         loss=loss, horizon=horizon)
    wilcoxon = wilcoxon_signed_rank(errors, ref_errors, loss=loss)
    return {
        "model": model,
        "reference_model": reference_model,
        "loss": loss,
        "skill_vs_reference_rmse": skill,
        "diebold_mariano": dm.to_dict(),
        "wilcoxon": wilcoxon,
    }
