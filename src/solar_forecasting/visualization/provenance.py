"""Machine-readable provenance for every published figure.

A figure that cannot be traced to the table and experiment it came from cannot be
audited, and a figure whose caption has drifted from its data is worse than no
figure. This module records, for every figure written during an evaluation run,
the source table, the experiment groups that produced the underlying records, and
the research question the figure serves, and writes the result to
``results/metrics/figure_provenance.json``.

Run implicitly by ``scripts/evaluate.py``; run directly to re-audit an existing
figure directory without recomputing anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.io import write_json

#: figure stem -> (source table, experiment groups, research question, caption)
FIGURE_PROVENANCE: dict[str, tuple[str, str, str, str]] = {
    "fig01_actual_vs_predicted": (
        "results/predictions/*.parquet", "B_ml_vs_dl", "RQ1",
        "Observed and forecast power over the first week of the test year, "
        "daylight steps, with the persistence reference."),
    "fig02_forecast_error_over_time": (
        "results/predictions/*.parquet", "B_ml_vs_dl", "RQ3",
        "Daily mean absolute error and signed error over the first three weeks, "
        "showing when the models fail."),
    "fig04_mae_comparison": (
        "results/tables/overall_model_comparison.csv", "B_ml_vs_dl", "RQ1",
        "Mean absolute error by model."),
    "fig05_rmse_comparison": (
        "results/tables/overall_model_comparison.csv", "B_ml_vs_dl", "RQ1",
        "Root mean squared error by model."),
    "fig06_r2_comparison": (
        "results/tables/overall_model_comparison.csv", "B_ml_vs_dl", "RQ1",
        "Coefficient of determination by model."),
    "fig07_horizon_comparison": (
        "results/tables/horizon_comparison.csv", "C_horizons", "RQ2",
        "RMSE, capacity-normalised error and skill against the four horizons."),
    "fig07b_horizon_mae": (
        "results/tables/horizon_comparison.csv", "C_horizons", "RQ2",
        "MAE against forecast horizon for every model."),
    "fig07c_horizon_ranking": (
        "results/tables/horizon_comparison.csv", "C_horizons", "RQ2",
        "Model rank by horizon, showing where the ranking changes."),
    "fig07d_horizon_cost": (
        "results/tables/horizon_comparison.csv", "C_horizons", "RQ2",
        "Training time and inference latency against horizon."),
    "fig08_weather_regime_comparison": (
        "results/tables/weather_regime_comparison.csv", "B_ml_vs_dl", "RQ3",
        "Capacity-normalised error and skill inside each weather regime."),
    "fig09_seasonal_comparison": (
        "results/tables/seasonal_comparison.csv", "B_ml_vs_dl", "RQ3",
        "Generalisation across the four seasons of the test year."),
    "fig10_importance_xgboost": (
        "results/tables/importance_xgboost.csv", "H_explainability", "RQ5",
        "SHAP mean absolute attribution per input variable, collapsed over the "
        "lookback window."),
    "fig10_importance_attention_lstm": (
        "results/tables/importance_attention_lstm.csv", "H_explainability", "RQ5",
        "Grouped permutation importance for the attention-recurrent model."),
    "fig10_shap_beeswarm_xgboost": (
        "results/tables/dependence_xgboost.csv", "H_explainability", "RQ5",
        "SHAP beeswarm summary of the attribution distribution."),
    "fig11_residual_distribution": (
        "results/predictions/*.parquet", "B_ml_vs_dl", "RQ1",
        "Distribution of normalised daylight residuals against the normal density."),
    "fig12_predicted_vs_actual_scatter": (
        "results/predictions/*.parquet", "B_ml_vs_dl", "RQ1",
        "Parity plots coloured by observation density."),
    "fig13_persistence_skill": (
        "results/tables/overall_model_comparison.csv", "B_ml_vs_dl", "RQ1",
        "Skill against the persistence reference, with bootstrap intervals."),
    "fig14_training_curves": (
        "results/experiments/*.json", "B_ml_vs_dl", "RQ1",
        "Training and validation loss per epoch with the restored best epoch."),
    "fig15_error_distribution_by_model": (
        "results/predictions/*.parquet", "B_ml_vs_dl", "RQ3",
        "Absolute-error box plots and exceedance curves."),
    "fig16_intraday_error_profile": (
        "results/tables/error_analysis.csv", "D_regimes", "RQ3",
        "Error by time-of-day band, locating the morning and evening ramps."),
    "fig17_cost_accuracy": (
        "results/tables/computational_cost.csv", "I_cost", "RQ7",
        "Accuracy against training cost with the Pareto frontier."),
    "fig18_conformal_intervals": (
        "results/tables/uncertainty_coverage.csv", "K_uncertainty", "RQ7",
        "Split-conformal prediction intervals and their coverage."),
    "fig19_dataset_overview": (
        "results/tables/dataset_statistics.csv", "-", "RQ0",
        "The case-study record: power, irradiance and a sample week."),
    "fig19b_temporal_coverage": (
        "results/tables/dataset_statistics.csv", "-", "RQ0",
        "Temporal coverage and the chronological split boundaries."),
    "fig19c_generation_distribution": (
        "results/tables/dataset_statistics.csv", "-", "RQ0",
        "Distribution of generated power, with the night-time mass shown."),
    "fig19d_weather_distribution": (
        "results/metrics/pipeline_report.json", "-", "RQ0",
        "Distribution of the weather-regime taxonomy over the record."),
    "fig20_feature_ablation": (
        "results/tables/feature_ablation.csv", "E_feature_ablation", "RQ4",
        "Change in error when each input group is removed."),
    "fig21_multiseed_spread": (
        "results/tables/multi_seed_results.csv", "M_multiseed", "RQ1",
        "Across-seed spread for each neural model."),
    "fig21b_multiseed_mae": (
        "results/tables/multi_seed_results.csv", "M_multiseed", "RQ1",
        "Across-seed spread in MAE for each neural model."),
    "fig22_cross_site_transfer": (
        "results/tables/cross_site_comparison.csv", "J_cross_site", "RQ6",
        "Within-site against cross-site performance per held-out site."),
    "fig23_feature_regimes": (
        "results/tables/feature_regimes.csv", "N_feature_regimes", "RQ4",
        "Additive input regimes: what each information type is worth alone."),
    "fig24_error_autocorrelation": (
        "results/tables/error_autocorrelation.csv", "-", "RQ0",
        "Autocorrelation of the forecast error."),
    "fig25_block_length_sensitivity": (
        "results/tables/block_length_sensitivity.csv", "-", "RQ0",
        "Bootstrap interval width against block length."),
    "fig26_residual_vs_generation": (
        "results/predictions/*.parquet", "B_ml_vs_dl", "RQ3",
        "Residual against generated power, showing where the error grows."),
    "fig26b_residual_vs_irradiance": (
        "results/predictions/*.parquet", "B_ml_vs_dl", "RQ3",
        "Residual against irradiance at the forecast origin."),
    "fig27_regime_importance_attention_lstm": (
        "results/tables/importance_regime_attention_lstm.csv", "H_explainability",
        "RQ5", "Permutation importance computed separately inside each regime."),
    "fig28_cost_memory": (
        "results/tables/computational_cost.csv", "I_cost", "RQ7",
        "Training time, inference latency, parameter count and model size."),
    "fig10c_integrated_gradients": (
        "results/explainability/integrated_gradients_<model>.csv", "H_explainability",
        "RQ5", "Mean absolute integrated gradient per input variable, with a "
               "completeness check against the baseline prediction."),
    "fig27b_condition_importance": (
        "results/explainability/importance_condition_<model>.csv", "H_explainability",
        "RQ5", "Permutation importance inside each weather regime and generation level."),
    "fig29_error_by_horizon": (
        "results/tables/horizon_comparison.csv", "C_horizons", "RQ2",
        "Error by forecast horizon, with the reference alongside."),
}


def build(figure_directory: Path) -> dict[str, Any]:
    """Match the figures on disk against the declared provenance."""
    if not figure_directory.exists():
        return {"figures": {}, "unmatched": []}
    present = {p.stem: p.name for p in sorted(figure_directory.glob("*.png"))}
    matched: dict[str, Any] = {}
    for stem, (source, groups, question, caption) in FIGURE_PROVENANCE.items():
        if stem not in present:
            continue
        # A model-specific figure name embeds the model, e.g. fig10_importance_lstm.
        matched[present[stem]] = {
            "source": source, "experiment_groups": groups,
            "research_question": question, "caption": caption}

    # A template whose stem is never itself written, because every figure made
    # from it carries a model suffix: fig10c_integrated_gradients_attention_lstm
    # exists and fig10c_integrated_gradients does not. Matching exact stems only
    # would leave those undocumented, while the prefix rule that declared them
    # matched hid the gap, so a figure could escape provenance by having a suffix.
    for stem, (source, groups, question, caption) in FIGURE_PROVENANCE.items():
        if stem in present:
            continue
        for name_stem, name in present.items():
            if name in matched or not name_stem.startswith(stem + "_"):
                continue
            suffix = name_stem[len(stem) + 1:]
            matched[name] = {
                "source": source.replace("<model>", suffix),
                "experiment_groups": groups,
                "research_question": question,
                "caption": caption + " Shown for " + suffix + ".",
                "model": suffix,
                "inherits_template": stem,
            }

    unmatched = [name for name in present.values() if name not in matched]
    return {
        "figures": matched,
        "unmatched": unmatched,
        "note": ("Figures whose name embeds a model, such as fig10_importance_<model> "
                 "or fig26_dependence_<model>_<feature>, inherit the provenance of the "
                 "corresponding template and differ only in the model named in the "
                 "filename."),
        "n_figures": len(present),
        "n_documented": len(matched),
    }


def write(figure_directory: Path, target: Path) -> dict[str, Any]:
    provenance = build(figure_directory)
    write_json(provenance, target)
    return provenance


def write_paper_index(provenance: dict[str, Any], target: Path) -> Path:
    """Write ``paper/figures/README.md`` from the provenance manifest.

    The index used to be maintained by hand and drifted: it listed a figure that
    was never produced and omitted most of the ones that were. It is generated
    here from the same manifest the figures are validated against, so it cannot
    name a figure that has no provenance, and cannot omit one that does.
    """
    figures = provenance.get("figures", {})
    if not figures:
        raise ValueError(
            "refusing to write an empty figure index: that would replace a stale "
            "list with a blank one. Build the manifest first.")

    lines = [
        "# Paper figures",
        "",
        "Generated by `python scripts/evaluate.py` from "
        "`results/metrics/figure_provenance.json`. Do not edit by hand.",
        "",
        "The figures themselves live in `results/figures/` and are the files the",
        "paper references. They are not duplicated into this directory: a second",
        "copy is a second thing that can go stale.",
        "",
        f"{len(figures)} figures, each traced to the table it is drawn from, the "
        "experiment group that produced it, and the research question it answers.",
        "",
        "| Figure | Research question | Experiment group | Drawn from |",
        "| --- | --- | --- | --- |",
    ]
    for name in sorted(figures):
        entry = figures[name]
        lines.append(
            f"| `{name}` | {entry.get('research_question', '')} "
            f"| {entry.get('experiment_groups', '')} "
            f"| `{entry.get('source', '')}` |")
    unmatched = provenance.get("unmatched") or []
    if unmatched:
        lines += ["", "## Undocumented", "",
                  "These files are in the figure directory but match no declared "
                  "template, so they carry no provenance:", ""]
        lines += [f"- `{name}`" for name in sorted(unmatched)]
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


