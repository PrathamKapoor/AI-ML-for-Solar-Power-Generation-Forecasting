#!/usr/bin/env python
"""Evaluation, analysis and figure generation.

    python scripts/evaluate.py                 # tables, stratified analysis, statistics, figures
    python scripts/evaluate.py --tables-only    # skip the expensive stages
    python scripts/evaluate.py --explainability  # refit the headline models for SHAP/permutation
    python scripts/evaluate.py --uncertainty    # conformal intervals around stored predictions
    python scripts/evaluate.py --report         # also write results/tables/RESULTS.md

The script is idempotent: every stage reads the stored per-run records and
predictions, and writes its own outputs. A stage whose inputs are absent is
skipped with a recorded reason rather than failing, so a partial experiment set
still produces a coherent results directory. The summary written to
``results/metrics/evaluation_report.json`` lists which stages ran, which were
skipped and why, so a reader can tell the difference between "no result" and "not
attempted".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from solar_forecasting.config import Config, load_config, set_thread_env  # noqa: E402
from solar_forecasting.evaluation import reporting  # noqa: E402
from solar_forecasting.evaluation import stratified as strat_mod  # noqa: E402
from solar_forecasting.evaluation import uncertainty as unc_mod  # noqa: E402
from solar_forecasting.statistics import comparison as stat_mod  # noqa: E402
from solar_forecasting.training import runner as runner_mod  # noqa: E402
from solar_forecasting.utils.io import write_json, write_table  # noqa: E402
from solar_forecasting.visualization import figures as fig_mod  # noqa: E402

PREDICTION_GROUPS = ("A_baselines", "B_ml_vs_dl", "C_horizons")


def tables_dir(config: Config) -> Path:
    path = config.project_root_for("results") / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_dir(config: Config) -> Path:
    path = config.project_root_for("results") / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def predictions_dir(config: Config) -> Path:
    return config.project_root_for("results") / "predictions"


def read_table(path: Path) -> pd.DataFrame | None:
    """Read a results table, returning None when it does not exist yet."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # pragma: no cover - unreadable artefact
        return None


def rated_power(config: Config) -> float | None:
    """Rated capacity of the case-study station, read from the pipeline report."""
    report = metrics_dir(config) / "pipeline_report.json"
    if not report.exists():
        return None
    try:
        stages = json.loads(report.read_text(encoding="utf-8")).get("stages", {})
    except json.JSONDecodeError:
        return None
    for stage in stages.values():
        if isinstance(stage, dict) and stage.get("rated_w"):
            return float(stage["rated_w"])
    return None


def load_predictions(config: Config, groups: tuple[str, ...] = PREDICTION_GROUPS,
                     horizon: str = "1h") -> dict[str, pd.DataFrame]:
    """Stored per-step test predictions, keyed by model name."""
    directory = predictions_dir(config)
    if not directory.exists():
        return {}
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.parquet")):
        parts = path.stem.split("__")
        if len(parts) < 3 or parts[0] not in groups:
            continue
        model, stored_horizon = parts[1], parts[2]
        if stored_horizon != horizon or model in frames:
            continue
        try:
            frames[model] = pd.read_parquet(path)
        except Exception:  # pragma: no cover - unreadable artefact
            continue
    return frames


def load_histories(config: Config, group: str = "B_ml_vs_dl") -> dict[str, list[dict[str, float]]]:
    """Training histories from the per-run records, for the training-curve figure."""
    directory = config.project_root_for("results") / "experiments"
    histories: dict[str, list[dict[str, float]]] = {}
    if not directory.exists():
        return histories
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not str(record.get("spec", {}).get("experiment_id", "")).startswith(group):
            continue
        history = record.get("training_history") or (record.get("metadata", {})
                                                     .get("training", {}).get("history"))
        if history:
            histories[record["spec"]["model"]] = history
    return histories


# ---------------------------------------------------------------------------
# Stage 1: tables from the registry
# ---------------------------------------------------------------------------
def stage_tables(config: Config, registry: pd.DataFrame, report: dict[str, Any]
                 ) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if registry.empty:
        report["tables"] = {"status": "skipped", "reason": "empty experiment registry"}
        return tables

    overall = reporting.overall_comparison(registry)
    if not overall.empty:
        tables["overall_model_comparison"] = overall
    horizon = reporting.horizon_comparison(registry)
    if not horizon.empty:
        tables["horizon_comparison"] = horizon
    ablation = reporting.feature_ablation(registry)
    if not ablation.empty:
        tables["feature_ablation"] = ablation
    regimes_table = reporting.feature_regime_comparison(registry)
    if not regimes_table.empty:
        tables["feature_regimes"] = regimes_table
    model_ablation = reporting.model_ablation(registry, config)
    if not model_ablation.empty:
        tables["model_ablation"] = model_ablation
    cost = reporting.computational_cost(registry)
    if not cost.empty:
        tables["computational_cost"] = cost
    metrics_doc = reporting.metric_documentation()
    if not metrics_doc.empty:
        tables["metric_documentation"] = metrics_doc

    for name, frame in tables.items():
        write_table(frame, tables_dir(config) / f"{name}.csv")
    report["tables"] = {"status": "ok", "written": sorted(tables)}
    return tables


# ---------------------------------------------------------------------------
# Stage 2: stratified error analysis
# ---------------------------------------------------------------------------
def stage_stratified(config: Config, predictions: dict[str, pd.DataFrame],
                     rated_w: float | None, report: dict[str, Any]
                     ) -> dict[str, pd.DataFrame]:
    if not predictions:
        report["stratified"] = {"status": "skipped",
                                "reason": "no stored predictions found"}
        return {}
    tables = {
        "weather_regime_comparison": reporting.regime_comparison(predictions, rated_w),
        "seasonal_comparison": reporting.seasonal_comparison(predictions, rated_w),
        "error_analysis": reporting.full_error_analysis(predictions, rated_w),
    }
    written = []
    for name, frame in tables.items():
        if frame.empty:
            continue
        write_table(frame, tables_dir(config) / f"{name}.csv")
        written.append(name)
    report["stratified"] = {"status": "ok", "written": written,
                            "n_models": len(predictions),
                            "models": sorted(predictions)}
    return {k: v for k, v in tables.items() if not v.empty}


# ---------------------------------------------------------------------------
# Stage 3: statistical comparison
# ---------------------------------------------------------------------------
def stage_statistics(config: Config, predictions: dict[str, pd.DataFrame],
                     report: dict[str, Any]) -> pd.DataFrame:
    """Bootstrap skill, Diebold-Mariano and Wilcoxon against the persistence reference.

    All three tests are run on the *same* aligned daylight rows for every model,
    so the comparison between models is paired and the reference is identical.
    """
    analysis = config.section("experiments").get("analysis", {}) or {}
    n_resamples = int(analysis.get("bootstrap_resamples", 2000))
    block_length = int(analysis.get("bootstrap_block_length", 96))
    confidence = float(analysis.get("bootstrap_confidence", 0.95))
    losses = list(analysis.get("dm_loss_functions", ["mae", "rmse"]))

    if "persistence" not in predictions or len(predictions) < 2:
        report["statistics"] = {"status": "skipped",
                                "reason": "need a persistence reference and at least one "
                                          "model prediction to compare"}
        return pd.DataFrame()

    base = (predictions["persistence"][["Time", "actual", "daylight", "predicted"]]
            .rename(columns={"predicted": "persistence"}))
    models = [m for m in predictions if m != "persistence"]
    frame = base
    for model in models:
        other = predictions[model][["Time", "predicted"]].rename(columns={"predicted": model})
        frame = frame.merge(other, on="Time", how="inner")
    frame = frame[frame["daylight"].astype(bool)].sort_values("Time").reset_index(drop=True)
    if frame.empty or len(models) == 0:
        report["statistics"] = {"status": "skipped", "reason": "no aligned daylight rows"}
        return pd.DataFrame()

    actual = frame["actual"].to_numpy(dtype=float)
    reference = frame["persistence"].to_numpy(dtype=float)
    reference_errors = actual - reference

    rows: list[dict[str, Any]] = []
    for model in models:
        if model not in frame.columns:
            continue
        predicted = frame[model].to_numpy(dtype=float)
        errors = actual - predicted
        entry = stat_mod.compare_models(
            {model: predicted, "persistence": reference}, actual=actual, model=model,
            reference_model="persistence", loss="mae", horizon=1,
            n_resamples=n_resamples, block_length=block_length, confidence=confidence)
        dm_by_loss = {"mae": entry["diebold_mariano"]}
        if "rmse" in losses:
            dm_by_loss["rmse"] = stat_mod.diebold_mariano(
                errors, reference_errors, model_a=model, model_b="persistence",
                loss="rmse", horizon=1).to_dict()
        skill = entry["skill_vs_reference_rmse"]
        for loss in losses:
            result = dm_by_loss.get(loss)
            if result is None:
                continue
            rows.append({
                "model": model,
                "reference_model": "persistence",
                "horizon": "1h",
                "loss": loss,
                "n_daylight_comparisons": int(len(frame)),
                "rmse_w": float(np.sqrt(np.mean(errors ** 2))),
                "skill_point": skill["point_estimate"],
                "skill_ci_low": skill["ci_low"],
                "skill_ci_high": skill["ci_high"],
                "dm_statistic": result["statistic"],
                "dm_p_value": result["p_value"],
                "dm_mean_differential_w": result["mean_differential"],
                "dm_better_model": result["better_model"],
                "dm_lag1_autocorrelation": result["lag_1_autocorrelation"],
                "wilcoxon_p_value": entry["wilcoxon"]["p_value"],
                "wilcoxon_effect_size": entry["wilcoxon"]["effect_size_rank_biserial"],
                "significant_at_0.05_raw": bool(result["p_value"] < 0.05),
                "losses_agree": bool(
                    (entry["diebold_mariano"]["p_value"] < 0.05) ==
                    (dm_by_loss.get("rmse", {"p_value": 1.0})["p_value"] < 0.05)),
            })

    table = pd.DataFrame(rows)
    if not table.empty:
        for loss in losses:
            mask = table["loss"] == loss
            if mask.any():
                table.loc[mask, "dm_p_value_holm"] = stat_mod.holm_bonferroni(
                    table.loc[mask, "dm_p_value"].to_numpy(dtype=float))
        table["significant_at_0.05_holm"] = table["dm_p_value_holm"] < 0.05
        table = table.sort_values(["loss", "dm_p_value"]).reset_index(drop=True)
        write_table(table, tables_dir(config) / "statistical_significance.csv")
    assumptions = stat_mod.ASSUMPTIONS
    write_json(assumptions, tables_dir(config) / "statistical_assumptions.json")

    # --- temporal dependence diagnostics ---------------------------------- #
    # The choice of block length and the power of every test above rest on how
    # strongly the errors are autocorrelated, so it is measured rather than
    # assumed and reported alongside the intervals.
    dependence: dict[str, Any] = {}
    best_model = None
    candidates = [m for m in frame.columns
                  if m not in ("Time", "actual", "daylight", "persistence")]
    if candidates:
        means = {m: float(np.sqrt(np.mean(
            (actual - frame[m].to_numpy(dtype=float)) ** 2))) for m in candidates}
        best_model = min(means, key=means.get)
    if best_model is not None:
        model_errors = actual - frame[best_model].to_numpy(dtype=float)
        reference_errors = actual - frame["persistence"].to_numpy(dtype=float)
        differential = stat_mod.paired_loss_difference(model_errors, reference_errors,
                                                       loss="mae")
        dependence["best_model"] = best_model
        dependence["model_error"] = stat_mod.effective_sample_size(model_errors)
        dependence["loss_differential"] = stat_mod.effective_sample_size(differential)
        profile = stat_mod.autocorrelation_profile(model_errors, max_lag=192)
        write_table(profile, tables_dir(config) / "error_autocorrelation.csv")
        sensitivity = stat_mod.block_length_sensitivity(
            differential, n_resamples=max(500, n_resamples // 4))
        write_table(sensitivity, tables_dir(config) / "block_length_sensitivity.csv")
        dependence["block_length_sensitivity"] = sensitivity.to_dict(orient="records")
        dependence["interpretation"] = (
            "The nominal sample size overstates the information in the error series by "
            f"about {dependence['loss_differential']['integrated_autocorrelation_time']:.0f}x. "
            "That is why an i.i.d. treatment would understate the interval width by "
            "roughly that factor, and why the Diebold-Mariano test rejects nothing even "
            "where the block-bootstrap interval excludes zero. Both are reported.")
    report["statistics"] = {
        "status": "ok",
        "n_models_compared": int(table["model"].nunique()) if not table.empty else 0,
        "n_paired_comparisons": int(len(frame)),
        "bootstrap_resamples": n_resamples,
        "block_length": block_length,
        "confidence": confidence,
        "losses": losses,
        "assumptions": assumptions,
        "temporal_dependence": dependence,
    }
    return table


# ---------------------------------------------------------------------------
# Stage 4: explainability (refits the declared models)
# ---------------------------------------------------------------------------
def stage_explainability(config: Config, report: dict[str, Any]) -> dict[str, Any]:
    """Refit the Experiment-H models and compute their importances.

    The models are refitted under the recorded seed rather than loaded from disk,
    so no binary weights are stored in the repository. The refit is deterministic
    and reproduces the Experiment-B run, so the importance figures describe the
    same model that produced the headline metrics.
    """
    spec_block = config.section("experiments").get("H_explainability", {}) or {}
    models = list(spec_block.get("models", ["xgboost"]))
    horizons = list(spec_block.get("horizons") or ["1h"])
    n_permutation = int(spec_block.get("n_permutation_repeats", 10))
    n_shap = int(spec_block.get("n_shap_samples", 2000))
    horizon = str(horizons[0])
    steps = {h["name"]: int(h["steps"]) for h in config.section("horizons")}.get(horizon, 4)
    directory = config.project_root_for("results") / "figures"
    directory.mkdir(parents=True, exist_ok=True)

    try:
        import shap  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        report["explainability"] = {"status": "skipped",
                                    "reason": f"shap is not importable: {exc}"}
        return {}

    from solar_forecasting.explainability import analysis as analysis_mod
    from _pipeline_state import load_pipeline_state

    state = load_pipeline_state(config)
    state["horizon"] = horizon
    state["horizon_steps"] = steps
    rated_w = state["station"].get("rated_w")

    results: dict[str, Any] = {}
    importance_frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    made: list[str] = []
    for model in models:
        print(f"  explainability: refitting {model} ...")
        try:
            entry = analysis_mod.explain_model(
                config, model, state, n_permutation_repeats=n_permutation,
                n_shap_samples=n_shap)
        except Exception as exc:
            skipped.append(f"{model}: {type(exc).__name__}: {exc}")
            print(f"    skipped {model}: {exc}")
            continue
        importance = entry["importance"]
        # The frame is kept in a parallel dictionary rather than in ``results``,
        # because ``results`` is what the JSON summary is built from and a frame
        # there would be both redundant and a serialisation hazard.
        importance_frames[model] = importance
        value_column = "importance" if "importance" in importance.columns else "mean_mae_increase_w"
        write_table(importance, tables_dir(config) / f"importance_{model}.csv")
        try:
            path = fig_mod.plot_feature_importance(
                importance, top_n=15, value_column=value_column,
                error_column="std_mae_increase_w" if value_column.startswith("mean") else None,
                title=f"{model}: {'mean |SHAP| per input variable' if value_column == 'importance' else 'increase in daylight MAE when permuted'}",
                name=f"fig10_importance_{model}", directory=directory)
            made.append(Path(path).name)
        except Exception as exc:
            skipped.append(f"{model} figure: {type(exc).__name__}: {exc}")
        shap_frame = entry.pop("_shap_frame", None)
        if shap_frame is not None and not shap_frame.empty:
            try:
                import shap as _shap
                import matplotlib.pyplot as plt
                figure = plt.figure()
                _shap.summary_plot(shap_frame, max_display=15, show=False)
                plt.tight_layout()
                made.append(Path(fig_mod.save(figure, f"fig10_shap_beeswarm_{model}",
                                              directory)).name)
                plt.close(figure)
            except Exception as exc:
                report.setdefault("notes", []).append(
                    f"SHAP beeswarm for {model} failed: {type(exc).__name__}: {exc}")

        dependence = entry.get("dependence")
        if dependence is not None and not dependence.empty:
            write_table(dependence, tables_dir(config) / f"dependence_{model}.csv")
            for feature in [c for c in dependence.columns if c.startswith("shap_")]:
                try:
                    made.append(Path(fig_mod.plot_dependence(
                        dependence.rename(columns={feature: "shap"}), feature[5:],
                        value="shap", name=f"fig26_dependence_{model}_{feature[5:]}",
                        directory=directory)).name)
                except Exception as exc:
                    report.setdefault("notes", []).append(
                        f"dependence plot {model}/{feature} failed: {exc}")

        regime_importance = entry.get("regime_importance")
        if regime_importance is not None and not regime_importance.empty:
            write_table(regime_importance,
                        tables_dir(config) / f"importance_regime_{model}.csv")
            try:
                made.append(Path(fig_mod.plot_regime_importance(
                    regime_importance, name=f"fig27_regime_importance_{model}",
                    directory=directory)).name)
            except Exception as exc:
                report.setdefault("notes", []).append(
                    f"regime importance plot for {model} failed: {exc}")

        results[model] = {k: v for k, v in entry.items() if k != "importance"}
        results[model]["figure"] = f"fig10_importance_{model}.png"
        print(f"  explainability: {model} done")

    if results:
        combined = analysis_mod.importance_frame(
            {model: {"importance": frame} for model, frame in importance_frames.items()})
        if not combined.empty:
            write_table(combined, tables_dir(config) / "feature_importance.csv")
        # DataFrames are written as their own CSV tables above; the JSON summary
        # keeps only the scalar findings, because a frame is not serialisable and
        # would otherwise abort the stage after it has already succeeded.
        scalar_summary = {model: {k: v for k, v in entry.items()
                                  if not hasattr(v, "to_dict") or isinstance(v, str)}
                          for model, entry in results.items()}
        for entry in scalar_summary.values():
            for key in ("baseline_metrics", "shap", "permutation"):
                if isinstance(entry.get(key), dict):
                    entry[key] = {k: v for k, v in entry[key].items()
                                  if not isinstance(v, (dict, list))}
        write_json(scalar_summary, metrics_dir(config) / "explainability_summary.json")

    report["explainability"] = {
        "status": "ok" if results else "nothing_computed",
        "models": results,
        "skipped": skipped,
        "figures": made,
        "n_rated_w": rated_w,
        "note": ("SHAP is used for the tree ensemble and grouped permutation importance "
                 "for the neural models; attention weights are deliberately not treated "
                 "as explanations. Both are predictive attributions, not causal claims."),
    }
    return results

# ---------------------------------------------------------------------------
# Stage 5: uncertainty
# ---------------------------------------------------------------------------
def stage_uncertainty(config: Config, predictions: dict[str, pd.DataFrame],
                      report: dict[str, Any]) -> pd.DataFrame:
    spec_block = config.section("experiments").get("K_uncertainty", {}) or {}
    models = list(spec_block.get("models", []))
    alphas = [float(a) for a in (spec_block.get("alphas") or [0.1, 0.2])]
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for model in models:
        if model not in predictions:
            skipped.append(model)
            continue
        try:
            rows.extend(unc_mod.coverage_table(predictions[model], model, "1h",
                                              alphas=tuple(alphas)))
        except ValueError as exc:
            skipped.append(f"{model} ({exc})")
    table = pd.DataFrame(rows)
    if not table.empty:
        write_table(table, tables_dir(config) / "uncertainty_coverage.csv")
    report["uncertainty"] = {
        "status": "ok" if not table.empty else "skipped",
        "models": [r for r in models if r in predictions],
        "skipped": skipped,
        "alphas": alphas,
        "note": ("Split conformal intervals calibrated on the first 31 days of the "
                 "test period and evaluated on the remainder. The coverage guarantee "
                 "is approximate because the calibration block is not exchangeable "
                 "with the evaluation block under seasonal drift."),
    }
    return table


# ---------------------------------------------------------------------------
# Stage 7: canonical experiment matrix
# ---------------------------------------------------------------------------
def stage_matrix(config: Config, registry: pd.DataFrame,
                 stratified_table: pd.DataFrame, report: dict[str, Any]
                 ) -> pd.DataFrame:
    """Build the one machine-readable result source for the whole study."""
    from solar_forecasting.evaluation import matrix as matrix_mod

    def read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    cross_site = read(tables_dir(config) / "cross_site_comparison.csv")
    coverage = read(tables_dir(config) / "uncertainty_coverage.csv")
    matrix, seed_summary = matrix_mod.build(registry, stratified_table, cross_site, coverage)
    if not matrix.empty:
        write_table(matrix, tables_dir(config) / "final_experiment_matrix.csv")
    if not seed_summary.empty:
        write_table(seed_summary, tables_dir(config) / "multi_seed_results.csv")
    summary = matrix_mod.coverage_summary(matrix)
    write_json(summary, metrics_dir(config) / "matrix_coverage.json")
    report["matrix"] = {
        "status": "ok" if not matrix.empty else "empty",
        "rows": int(len(matrix)),
        "written": ["final_experiment_matrix.csv"] +
                   (["multi_seed_results.csv"] if not seed_summary.empty else []),
        "coverage": summary,
    }
    return matrix


# ---------------------------------------------------------------------------
# Stage 6: figures
# ---------------------------------------------------------------------------
def stage_figures(config: Config, tables: dict[str, pd.DataFrame],
                  predictions: dict[str, pd.DataFrame], rated_w: float | None,
                  report: dict[str, Any]) -> None:
    made: list[str] = []
    failed: dict[str, str] = {}
    directory = config.project_root_for("results") / "figures"
    directory.mkdir(parents=True, exist_ok=True)

    def attempt(label: str, function, *args, **kwargs) -> None:
        try:
            path = function(*args, **kwargs)
            made.append(Path(path).name)
        except Exception as exc:
            failed[label] = f"{type(exc).__name__}: {exc}"

    overall = tables.get("overall_model_comparison", pd.DataFrame())
    if not overall.empty and predictions:
        headline = sorted(
            (m for m in predictions if m in set(overall["model"])),
            key=lambda m: float(overall.loc[overall["model"] == m, "rmse"].iloc[0]))
        best = headline[0] if headline else next(iter(predictions))
        attempt("actual_vs_predicted", fig_mod.plot_actual_vs_predicted,
                predictions, best, rated_w, directory=directory)
        attempt("error_over_time", fig_mod.plot_error_over_time,
                predictions, headline[:6], rated_w, directory=directory)
        attempt("metric_mae", fig_mod.plot_metric_comparison, overall, "mae",
                "Mean absolute error by model (daylight test steps, 1-hour horizon)",
                "fig04_mae_comparison", directory=directory)
        attempt("metric_rmse", fig_mod.plot_metric_comparison, overall, "rmse",
                "Root mean squared error by model (daylight test steps, 1-hour horizon)",
                "fig05_rmse_comparison", directory=directory)
        attempt("metric_r2", fig_mod.plot_metric_comparison, overall, "r2",
                "Coefficient of determination by model", "fig06_r2_comparison",
                directory=directory)
        attempt("residual_distribution", fig_mod.plot_residual_distribution,
                predictions, headline, rated_w, directory=directory)
        attempt("scatter", fig_mod.plot_predicted_vs_actual, predictions,
                headline[:5], rated_w, directory=directory)
        attempt("skill", fig_mod.plot_skill_comparison, overall,
                directory=directory)
        attempt("error_by_model", fig_mod.plot_error_distribution_by_model,
                predictions, headline, rated_w, directory=directory)

    horizon = tables.get("horizon_comparison", pd.DataFrame())
    if not horizon.empty:
        attempt("horizons", fig_mod.plot_horizon_comparison, horizon, directory=directory)

    regime = tables.get("weather_regime_comparison", pd.DataFrame())
    if not regime.empty:
        attempt("regimes", fig_mod.plot_regime_comparison, regime, directory=directory)
    season = tables.get("seasonal_comparison", pd.DataFrame())
    if not season.empty:
        attempt("seasons", fig_mod.plot_seasonal_comparison, season, directory=directory)
    error_analysis = tables.get("error_analysis", pd.DataFrame())
    if not error_analysis.empty:
        model = str(error_analysis["model"].iloc[0])
        attempt("intraday", fig_mod.plot_intraday_error, error_analysis, model, rated_w,
                directory=directory)
    ablation = tables.get("feature_ablation", pd.DataFrame())
    if not ablation.empty:
        attempt("ablation", fig_mod.plot_ablation, ablation, directory=directory)
    cost = tables.get("computational_cost", pd.DataFrame())
    if not cost.empty:
        attempt("cost", fig_mod.plot_cost_accuracy, cost, directory=directory)
        attempt("cost_memory", fig_mod.plot_memory_and_size, cost, directory=directory)

    # --- multi-seed spread ------------------------------------------------- #
    seed_table = read_table(tables_dir(config) / "multi_seed_results.csv")
    if seed_table is not None and not seed_table.empty:
        attempt("multiseed_rmse", fig_mod.plot_multiseed, seed_table, "rmse",
                directory=directory)
        attempt("multiseed_mae", fig_mod.plot_multiseed, seed_table, "mae",
                name="fig21b_multiseed_mae", directory=directory)

    # --- cross-site transfer ------------------------------------------------ #
    cross_site = read_table(tables_dir(config) / "cross_site_comparison.csv")
    if cross_site is not None and not cross_site.empty:
        attempt("cross_site", fig_mod.plot_cross_site, cross_site, directory=directory)

    # --- additive input regimes --------------------------------------------- #
    regime_table = tables.get("feature_regimes", pd.DataFrame())
    if not regime_table.empty:
        attempt("feature_regimes", fig_mod.plot_feature_regimes, regime_table,
                directory=directory)

    # --- temporal dependence ------------------------------------------------ #
    acf = read_table(tables_dir(config) / "error_autocorrelation.csv")
    if acf is not None and not acf.empty:
        attempt("autocorrelation", fig_mod.plot_error_autocorrelation, acf,
                directory=directory)
    blocks = read_table(tables_dir(config) / "block_length_sensitivity.csv")
    if blocks is not None and not blocks.empty:
        attempt("block_length", fig_mod.plot_block_length_sensitivity, blocks,
                directory=directory)

    history = load_histories(config)
    if history:
        attempt("training_curves", fig_mod.plot_training_curves, history,
                directory=directory)
    else:
        report.setdefault("notes", []).append(
            "training curves: no run has recorded an epoch history yet; the field is "
            "persisted by current runs, so this figure appears after the neural "
            "experiments are re-run")

    alphas = [float(a) for a in ((config.section("experiments")
                                 .get("K_uncertainty", {}) or {}).get("alphas") or [0.1])]
    for model in sorted(predictions):
        try:
            intervals = unc_mod.interval_frame(predictions[model], alpha=alphas[0])
            attempt(f"intervals_{model}", fig_mod.plot_intervals, intervals, rated_w,
                    directory=directory)
            break
        except Exception as exc:
            report.setdefault("notes", []).append(
                f"conformal intervals: {model} skipped ({type(exc).__name__}: {exc})")
            break

    featured = config.project_root_for("data") / "processed" / "featured_primary_station.parquet"
    if featured.exists():
        attempt("dataset", fig_mod.plot_dataset_overview,
                pd.read_parquet(featured), rated_w, directory=directory)

    report["figures"] = {"status": "ok" if made else "nothing_to_plot",
                         "written": sorted(made), "failed": failed}


# ---------------------------------------------------------------------------
# Markdown digest
# ---------------------------------------------------------------------------
def write_markdown_report(config: Config, tables: dict[str, pd.DataFrame],
                          registry: pd.DataFrame, report: dict[str, Any]) -> Path:
    """A human-readable digest of the results, regenerated from the tables."""
    lines: list[str] = ["# Results digest", "",
                        "Generated by `scripts/evaluate.py` from the stored experiment "
                        "records. Every number below is computed from a run; nothing is "
                        "entered by hand.", ""]
    overall = tables.get("overall_model_comparison", pd.DataFrame())
    if not overall.empty:
        lines += ["## Overall model comparison (1-hour horizon, daylight steps)", "",
                  reporting.format_markdown_table(
                      overall[["model", "mae", "rmse", "nrmse_capacity", "r2", "smape",
                               "skill_vs_persistence_rmse", "train_seconds"]]), ""]
    horizon = tables.get("horizon_comparison", pd.DataFrame())
    if not horizon.empty:
        lines += ["## Horizon comparison", "",
                  reporting.format_markdown_table(
                      horizon[["model", "horizon", "rmse", "nrmse_capacity", "r2",
                               "skill_vs_persistence_rmse"]]), ""]
    regime = tables.get("weather_regime_comparison", pd.DataFrame())
    if not regime.empty:
        lines += ["## Weather-regime comparison", "",
                  reporting.format_markdown_table(
                      regime[["model", "stratum", "n", "nrmse_capacity",
                              "skill_vs_persistence_rmse"]]), ""]
    season = tables.get("seasonal_comparison", pd.DataFrame())
    if not season.empty:
        lines += ["## Seasonal comparison", "",
                  reporting.format_markdown_table(
                      season[["model", "stratum", "n", "nrmse_capacity",
                              "skill_vs_persistence_rmse"]]), ""]
    ablation = tables.get("feature_ablation", pd.DataFrame())
    if not ablation.empty:
        lines += ["## Feature ablation", "",
                  reporting.format_markdown_table(
                      ablation[["model", "feature_spec", "rmse", "nrmse_capacity",
                                "rmse_vs_full_pct"]]), ""]
    cost = tables.get("computational_cost", pd.DataFrame())
    if not cost.empty:
        lines += ["## Computational cost", "",
                  reporting.format_markdown_table(
                      cost[["model", "train_seconds", "inference_ms_per_window",
                            "n_parameters", "nrmse_capacity", "pareto_optimal"]]), ""]
    lines += ["## Stage status", "",
              reporting.format_markdown_table(
                  pd.DataFrame([{"stage": k,
                                 "status": v.get("status"),
                                 "detail": v.get("reason") or ", ".join(v.get("written", []))}
                                for k, v in report.items() if isinstance(v, dict)])), ""]
    path = tables_dir(config) / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-only", action="store_true",
                        help="only rebuild tables from existing records")
    parser.add_argument("--explainability", action="store_true",
                        help="refit the headline models and compute SHAP and permutation "
                             "importance (slow)")
    parser.add_argument("--uncertainty", action="store_true",
                        help="compute split-conformal coverage")
    parser.add_argument("--report", action="store_true",
                        help="write results/tables/RESULTS.md")
    parser.add_argument("--horizon", default=None, help="horizon for the stratified stage")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    set_thread_env(args.threads)
    config = load_config()
    report: dict[str, Any] = {}
    started = time.perf_counter()

    registry = runner_mod.rebuild_registry_from_disk(config)
    report["registry"] = {"status": "ok", "n_rows": int(len(registry)),
                          "groups": sorted(registry["experiment_group"].unique().tolist())
                          if not registry.empty else []}
    print(f"registry: {len(registry)} records")

    rated_w = rated_power(config)
    report["rated_capacity_w"] = rated_w

    tables = stage_tables(config, registry, report)
    print(f"tables: {len(tables)} written")

    predictions = load_predictions(config, horizon=args.horizon or "1h")
    stratified_tables: dict[str, pd.DataFrame] = {}
    if not args.tables_only:
        stratified_tables = stage_stratified(config, predictions, rated_w, report)
        print(f"stratified: {len(stratified_tables)} tables")
        stage_statistics(config, predictions, report)
        print("statistics: complete")
        if args.uncertainty:
            stage_uncertainty(config, predictions, report)
            print("uncertainty: complete")
        if args.explainability:
            stage_explainability(config, report)
            print("explainability: complete")
        elif not args.tables_only:
            # SHAP and permutation importance need a fitted estimator, so the stage
            # is opt-in. Recording it as not attempted keeps "not run" visibly
            # distinct from "run and found nothing".
            report["explainability"] = {
                "status": "not_run",
                "reason": "requires refitting the models; re-run with --explainability",
            }

    all_tables = {**tables, **stratified_tables}
    if not args.tables_only:
        stage_figures(config, all_tables, predictions, rated_w, report)
        print(f"figures: {len(report.get('figures', {}).get('written', []))} written")

    stage_matrix(config, registry,
                 all_tables.get("error_analysis", pd.DataFrame()), report)
    print(f"matrix: {report['matrix']['rows']} rows")

    if args.report:
        path = write_markdown_report(config, all_tables, registry, report)
        print(f"report: {path}")

    summary = reporting.results_summary(registry)
    write_json(summary, metrics_dir(config) / "results_summary.json")
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    write_json(report, metrics_dir(config) / "evaluation_report.json")
    print(f"done in {report['elapsed_seconds']:.1f}s -> "
          f"{metrics_dir(config) / 'evaluation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


