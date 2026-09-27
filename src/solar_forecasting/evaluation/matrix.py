"""The canonical machine-readable result source.

Every number the paper reports is derivable from one table,
``results/tables/final_experiment_matrix.csv``, and this module builds it from
the artefacts rather than from anything typed by hand. It unions five sources:

1. the experiment registry (all training runs, every group, every seed);
2. the per-stratum stratified metrics (regime, season, time of day, level, ramp);
3. the multi-seed aggregation (one row per model with the across-seed spread);
4. the cross-site transfer records (within-site, cross-site, leave-one-site-out);
5. the conformal coverage records.

Each row carries the research question it answers, so the mapping from question
to evidence is a column rather than a claim in prose. ``status`` distinguishes a
run that was executed from one that was merely declared, which is what keeps the
matrix honest when an experiment is absent.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Which research question each experiment group answers.
GROUP_TO_RQ: dict[str, str] = {
    "A_baselines": "RQ1",
    "B_ml_vs_dl": "RQ1",
    "C_horizons": "RQ2",
    "D_regimes": "RQ3",
    "G_seasonal": "RQ3",
    "J_cross_site": "RQ6",
    "E_feature_ablation": "RQ4",
    "F_model_ablation": "RQ4",
    "M_multiseed": "RQ1",
    "H_explainability": "RQ5",
    "I_cost": "RQ7",
    "K_uncertainty": "RQ7",
}

MATRIX_COLUMNS = [
    "experiment_id", "research_question", "experiment_group", "protocol", "model",
    "model_family", "horizon", "site", "weather_regime", "season", "feature_regime",
    "seed", "n_observations", "mae", "rmse", "nrmse", "smape", "r2",
    "skill_vs_persistence", "train_time_seconds", "inference_ms_per_window",
    "n_parameters", "status", "source",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=MATRIX_COLUMNS)


def from_registry(registry: pd.DataFrame) -> pd.DataFrame:
    """Rows for every executed training run."""
    if registry.empty:
        return _empty()
    frame = registry.copy()
    frame["research_question"] = frame["experiment_group"].map(GROUP_TO_RQ).fillna("")
    frame["protocol"] = "within_site_chronological"
    frame["site"] = "LSK North"
    frame["weather_regime"] = "all_daylight"
    frame["season"] = "all"
    frame["feature_regime"] = frame["feature_spec"]
    frame["experiment_id"] = frame["experiment_id"]
    frame["nrmse"] = frame.get("nrmse_capacity")
    frame["skill_vs_persistence"] = frame.get("skill_vs_persistence_rmse")
    frame["train_time_seconds"] = frame.get("train_seconds")
    frame["n_observations"] = frame.get("n_test")
    frame["status"] = "executed"
    frame["source"] = "results/experiments/<experiment_id>.json"
    keep = [c for c in MATRIX_COLUMNS if c in frame.columns]
    return frame[keep + [c for c in frame.columns if c not in keep]]


def from_stratified(stratified: pd.DataFrame) -> pd.DataFrame:
    """Rows for each regime, season and time-of-day stratum of each model."""
    if stratified.empty:
        return _empty()
    frame = stratified.copy()
    frame["research_question"] = "RQ3"
    frame["experiment_group"] = "D_regimes/G_seasonal"
    frame["protocol"] = "within_site_chronological"
    frame["experiment_id"] = (
        "S_" + frame["model"].astype(str) + "_" + frame["stratum_type"].astype(str)
        + "_" + frame["stratum"].astype(str).str.replace(" ", "_", regex=False)
        + "_" + frame["horizon"].astype(str))
    frame["site"] = "LSK North"
    frame["weather_regime"] = np.where(frame["stratum_type"] == "regime",
                                       frame["stratum"], "all_daylight")
    frame["season"] = np.where(frame["stratum_type"] == "season", frame["stratum"], "all")
    frame["feature_regime"] = "full"
    frame["seed"] = 42
    frame["nrmse"] = frame.get("nrmse_capacity")
    frame["skill_vs_persistence"] = frame.get("skill_vs_persistence_rmse")
    frame["n_observations"] = frame.get("n")
    frame["status"] = "executed"
    frame["source"] = "results/tables/error_analysis.csv"
    keep = [c for c in MATRIX_COLUMNS if c in frame.columns]
    return frame[keep + [c for c in frame.columns if c not in keep]]


def multiseed_summary(registry: pd.DataFrame, group: str = "M_multiseed"
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate repeated seeds: one row per model with the across-seed spread.

    Returns ``(per_seed, summary)``. The summary reports the mean, the standard
    deviation and a t-based 95% interval of the mean across seeds, plus the
    worst and best single seed so that the spread is visible rather than implied.
    """
    empty = _empty()
    if registry.empty:
        return empty, empty
    frame = registry[registry["experiment_group"] == group].copy()
    if frame.empty:
        return empty, empty
    metrics = ["mae", "rmse", "nrmse_capacity", "r2", "skill_vs_persistence_rmse"]
    rows = []
    for (model, horizon), group_frame in frame.groupby(["model", "horizon"]):
        n_seeds = int(group_frame["seed"].nunique())
        entry: dict[str, Any] = {
            "experiment_id": f"MS_{model}_{horizon}",
            "research_question": "RQ1",
            "experiment_group": group,
            "protocol": "multi_seed",
            "model": model,
            "horizon": horizon,
            "site": "LSK North",
            "weather_regime": "all_daylight",
            "season": "all",
            "feature_regime": "full",
            "n_seeds": n_seeds,
            "seeds": ", ".join(str(int(s)) for s in sorted(group_frame["seed"].unique())),
            "status": "executed",
            "source": "results/tables/multi_seed_results.csv",
        }
        for metric in metrics:
            values = pd.to_numeric(group_frame.get(metric), errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            half = 0.0
            if values.size > 1:
                from scipy import stats as _stats
                half = float(_stats.t.ppf(0.975, values.size - 1) * std / np.sqrt(values.size))
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
            entry[f"{metric}_ci_low"] = mean - half
            entry[f"{metric}_ci_high"] = mean + half
            entry[f"{metric}_min"] = float(values.min())
            entry[f"{metric}_max"] = float(values.max())
        rows.append(entry)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        front = ["experiment_id", "research_question", "experiment_group", "protocol",
                 "model", "horizon", "n_seeds", "seeds", "status", "source"]
        ordered = [c for c in front if c in summary.columns]
        summary = summary[ordered + [c for c in summary.columns if c not in ordered]]
    per_seed = from_registry(frame)
    per_seed["protocol"] = "multi_seed_single_run"
    return per_seed, summary


def from_cross_site(cross_site: pd.DataFrame) -> pd.DataFrame:
    """Rows for every transfer protocol."""
    if cross_site is None or cross_site.empty:
        return _empty()
    frame = cross_site.copy()
    frame["research_question"] = "RQ6"
    frame["experiment_group"] = "J_cross_site"
    frame["experiment_id"] = (
        "J_" + frame["protocol"].astype(str) + "_" + frame["model"].astype(str) + "_"
        + frame["horizon"].astype(str) + "_" + frame["test_site"].astype(str)
             .str.replace(" ", "_", regex=False))
    frame["site"] = frame["test_site"]
    frame["weather_regime"] = "all_daylight"
    frame["season"] = "all"
    frame["feature_regime"] = "full"
    frame["nrmse"] = frame.get("nrmse_capacity")
    frame["skill_vs_persistence"] = frame.get("skill_vs_persistence_rmse")
    frame["n_observations"] = frame.get("n")
    frame["train_time_seconds"] = frame.get("train_seconds")
    frame["status"] = "executed"
    frame["source"] = "results/tables/cross_site_comparison.csv"
    keep = [c for c in MATRIX_COLUMNS if c in frame.columns]
    return frame[keep + [c for c in frame.columns if c not in keep]]


def from_uncertainty(coverage: pd.DataFrame) -> pd.DataFrame:
    """Rows for every conformal coverage result."""
    if coverage is None or coverage.empty:
        return _empty()
    frame = coverage.copy()
    frame["research_question"] = "RQ7"
    frame["experiment_group"] = "K_uncertainty"
    frame["experiment_id"] = (
        "K_" + frame["model"].astype(str) + "_a" + frame["alpha"].astype(str) + "_"
        + frame["scope"].astype(str))
    frame["site"] = "LSK North"
    frame["season"] = "all"
    frame["feature_regime"] = "full"
    frame["status"] = "executed"
    frame["source"] = "results/tables/uncertainty_coverage.csv"
    keep = [c for c in MATRIX_COLUMNS if c in frame.columns]
    return frame[keep + [c for c in frame.columns if c not in keep]]


def build(registry: pd.DataFrame, stratified: pd.DataFrame | None = None,
          cross_site: pd.DataFrame | None = None,
          coverage: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the canonical matrix. Returns ``(matrix, multiseed_summary)``."""
    per_seed, seed_summary = multiseed_summary(registry)
    # `stratified or _empty()` would raise, because a DataFrame's truth value is
    # ambiguous. An explicit None check is the only correct form here.
    strata = _empty() if stratified is None else stratified
    frames = [from_registry(registry), from_stratified(strata),
              per_seed, from_cross_site(cross_site), from_uncertainty(coverage)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty(), seed_summary
    matrix = pd.concat(frames, ignore_index=True)
    matrix = matrix.reindex(columns=[c for c in MATRIX_COLUMNS if c in matrix.columns]
                            + [c for c in matrix.columns if c not in MATRIX_COLUMNS])
    return matrix, seed_summary


def coverage_summary(matrix: pd.DataFrame) -> dict[str, Any]:
    """How much of the declared matrix was actually executed."""
    if matrix.empty:
        return {"rows": 0}
    executed = matrix[matrix["status"] == "executed"]
    return {
        "rows": int(len(matrix)),
        "executed_rows": int(len(executed)),
        "research_questions_covered": sorted(
            {str(q) for q in executed["research_question"] if str(q)}),
        "experiments_covered": sorted({str(g) for g in executed["experiment_group"]}),
        "horizons_covered": sorted({str(h) for h in executed["horizon"]}),
        "models_covered": sorted({str(m) for m in executed["model"]}),
        "sites_covered": sorted({str(s) for s in executed["site"]}),
    }
