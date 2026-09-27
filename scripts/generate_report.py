#!/usr/bin/env python
"""Assemble the project report from computed artefacts.

    python scripts/generate_report.py

Writes ``docs/RESULTS_REPORT.md``: a single document containing the dataset
summary, the metric definitions, every results table in markdown, the list of
figures with their captions, the statistical summary, and an explicit statement
of which experiment groups were run and which were not.

The script contains no numbers of its own. Every value is read from
``results/tables/*.csv``, ``results/metrics/*.json`` and the experiment registry,
so the report cannot drift from the runs that produced it. If a table is absent,
the corresponding section says so instead of being omitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

import pandas as pd  # noqa: E402

from solar_forecasting.config import load_config  # noqa: E402
from solar_forecasting.evaluation import reporting  # noqa: E402
from solar_forecasting.utils.io import write_json  # noqa: E402

TABLE_SECTIONS: list[tuple[str, str, str, list[str]]] = [
    ("Overall model comparison (1-hour horizon, daylight steps)",
     "overall_model_comparison.csv",
     "Every model at the default horizon, on identical inputs and an identical split. "
     "Skill is measured against the persistence reference on the same timestamps.",
     ["model", "model_family", "n_features", "mae", "rmse", "nrmse_capacity", "r2",
      "smape", "bias", "skill_vs_persistence_rmse", "train_seconds"]),
    ("Horizon comparison", "horizon_comparison.csv",
     "Accuracy at 15 minutes, 1 hour, 6 hours and 24 hours ahead. The 1-hour rows are "
     "taken from the headline experiment, whose configuration is identical to the "
     "horizon experiment at 1 hour; the source of each row is recorded in the table.",
     ["model", "horizon", "source_group", "rmse", "nrmse_capacity", "r2",
      "skill_vs_persistence_rmse"]),
    ("Weather-regime comparison", "weather_regime_comparison.csv",
     "Daylight errors stratified by the regime taxonomy defined in "
     "`configs/data.yaml` and `docs/methodology.md`.",
     ["model", "stratum", "n", "nrmse_capacity", "rmse", "r2",
      "skill_vs_persistence_rmse"]),
    ("Seasonal comparison", "seasonal_comparison.csv",
     "The test period is a full calendar year, so each season appears exactly once.",
     ["model", "stratum", "n", "nrmse_capacity", "rmse", "skill_vs_persistence_rmse"]),
    ("Error analysis by time of day, generation level and ramping",
     "error_analysis.csv",
     "The mandatory error analysis: where each model fails, not only how much it "
     "fails on average.",
     ["model", "stratum_type", "stratum", "n", "nrmse_capacity", "mae", "bias"]),
    ("Feature-group ablation", "feature_ablation.csv",
     "Each input group removed in turn. The percentage columns are relative to the "
     "same model with the full input set, so they are directly comparable.",
     ["model", "feature_spec", "n_features", "rmse", "nrmse_capacity",
      "rmse_vs_full_pct", "skill_vs_persistence_rmse"]),
    ("Model ablation: composite architectures against their backbones",
     "model_ablation.csv",
     "All models here share a seed, a split and an input set, so a difference is "
     "attributable to the architecture.",
     ["composite", "backbone", "composite_rmse", "backbone_rmse", "rmse_change_w",
      "rmse_change_pct"]),
    ("Statistical comparison against persistence", "statistical_significance.csv",
     "Moving-block bootstrap skill intervals, the Diebold-Mariano test with the "
     "Harvey-Leybourne-Newbold small-sample correction, and the Wilcoxon signed-rank "
     "test as a secondary check. Holm-Bonferroni adjusted p-values are reported "
     "because several models are compared at once.",
     ["model", "loss", "n_daylight_comparisons", "rmse_w", "skill_point", "skill_ci_low",
      "skill_ci_high", "dm_p_value", "dm_p_value_holm", "dm_lag1_autocorrelation",
      "wilcoxon_p_value", "wilcoxon_effect_size"]),
    ("Computational cost", "computational_cost.csv",
     "Training duration, per-window inference latency and parameter count, with the "
     "Pareto-optimal models flagged. Thread counts are pinned to one, so the "
     "durations are comparable within a machine; only the ranking is portable.",
     ["model", "model_family", "train_seconds", "inference_ms_per_window",
      "n_parameters", "nrmse_capacity", "pareto_optimal"]),
    ("Conformal interval coverage", "uncertainty_coverage.csv",
     "Split conformal intervals calibrated on an initial block of the test period. "
     "The coverage guarantee is approximate; see the module documentation.",
     ["model", "alpha", "scope", "nominal_coverage", "empirical_coverage",
      "coverage_error", "interval_half_width_w", "n"]),
    ("Feature importance", "feature_importance.csv",
     "SHAP for the tree ensemble, grouped permutation importance for the neural "
     "models. Predictive attributions only.",
     ["model", "feature", "value", "method"]),
    ("Metric definitions", "metric_documentation.csv",
     "Formula, interpretation and limitation of every metric computed, including why "
     "MAPE is not used for this target.",
     ["metric", "formula", "units", "interpretation", "limitations", "primary"]),
    ("Dataset statistics", "dataset_statistics.csv",
     "Descriptive statistics of the case-study record, for the dataset section.",
     ["quantity", "value", "unit", "note"]),
]

FIGURE_CAPTIONS: list[tuple[str, str]] = [
    ("fig01_actual_vs_predicted", "Observed and forecast power over the first week of the "
                                 "test year, daylight steps, with the persistence reference."),
    ("fig02_forecast_error_over_time", "Daily mean absolute error and signed error over the "
                                       "first three weeks, showing when the models fail."),
    ("fig04_mae_comparison", "Mean absolute error by model."),
    ("fig05_rmse_comparison", "Root mean squared error by model."),
    ("fig06_r2_comparison", "Coefficient of determination by model."),
    ("fig07_horizon_comparison", "RMSE, capacity-normalised error and skill against the "
                                 "four forecast horizons."),
    ("fig08_weather_regime_comparison", "Capacity-normalised error and skill inside each "
                                        "weather regime."),
    ("fig09_seasonal_comparison", "Generalisation across the four seasons of the test year."),
    ("fig10_feature_importance", "Feature importance (model-specific variants are written "
                                 "as fig10_importance_<model>)."),
    ("fig11_residual_distribution", "Distribution of normalised daylight residuals against "
                                    "the normal density."),
    ("fig12_predicted_vs_actual_scatter", "Forecast versus observed parity plots, coloured "
                                          "by observation density."),
    ("fig13_persistence_skill", "Skill against the persistence reference, with bootstrap "
                                "intervals where computed."),
    ("fig14_training_curves", "Training and validation loss per epoch for the neural models, "
                              "with the restored best epoch marked."),
    ("fig15_error_distribution_by_model", "Absolute-error box plots and exceedance curves."),
    ("fig16_intraday_error_profile", "Error by time-of-day band, locating the morning and "
                                     "evening ramps."),
    ("fig17_cost_accuracy", "Accuracy against training cost with the Pareto frontier."),
    ("fig18_conformal_intervals", "Conformal prediction intervals and their coverage "
                                  "through the evaluation period."),
    ("fig19_dataset_overview", "The case-study record: power, irradiance and a sample week."),
    ("fig20_feature_ablation", "Change in error when each input group is removed."),
]

GROUP_DESCRIPTIONS = {
    "A_baselines": "Baseline comparison: the two persistence references against a linear fit.",
    "B_ml_vs_dl": "Machine learning versus deep learning at the default horizon, on identical "
                  "inputs and an identical split. This is the headline experiment.",
    "C_horizons": "Horizon comparison at 15 minutes, 1 hour, 6 hours and 24 hours ahead.",
    "D_regimes": "Weather-regime stratification of the stored predictions (analysis stage).",
    "E_feature_ablation": "Feature-group ablation for a tree model and a recurrent model.",
    "F_model_ablation": "Composite architectures against their backbones (analysis of "
                        "Experiment B).",
    "G_seasonal": "Seasonal stratification of the stored predictions (analysis stage).",
    "H_explainability": "SHAP and permutation importance, with integrated gradients for "
                        "the recurrent models (executed).",
    "I_cost": "Training and inference cost, assembled from the registry (analysis stage).",
    "J_cross_site": "Cross-site transfer to held-out stations: within-site reference, "
                    "train-on-A/test-on-B, and leave-one-site-out (executed).",
    "K_uncertainty": "Split conformal prediction intervals, with coverage and interval "
                     "width for three nominal levels (executed).",
    "M_multiseed": "Five seeds per recurrent model, aggregated to mean, standard deviation "
                   "and a 95% confidence interval (executed).",
    "N_feature_regimes": "Additive feature regimes, from power history alone to the full "
                         "engineered set (executed).",
    "S_stratified": "Error stratified by regime, season, time of day, generation level and "
                    "ramping state (analysis of stored predictions).",
}


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=None,
                        help="output path (default: docs/RESULTS_REPORT.md)")
    args = parser.parse_args()

    config = load_config()
    results = config.project_root_for("results")
    tables_dir = results / "tables"
    figures_dir = results / "figures"

    registry_path = results / "experiments.csv"
    registry = read_table(registry_path)

    lines: list[str] = [
        "# Results report",
        "",
        "Generated by `python scripts/generate_report.py` from the artefacts in "
        "`results/`. Every number in this document is read from a computed file; "
        "nothing is transcribed by hand. Sections whose inputs are absent are marked "
        "as missing rather than filled in.",
        "",
    ]

    lines += ["## 1. Experiment coverage", ""]
    if registry is None or registry.empty:
        lines += ["_No experiment records found._", ""]
    else:
        coverage = (registry.groupby(["experiment_group", "horizon", "feature_spec"])
                    .agg(n_runs=("model", "count"),
                         models=("model", lambda s: ", ".join(sorted(set(s)))))
                    .reset_index())
        coverage["description"] = coverage["experiment_group"].map(
            lambda g: GROUP_DESCRIPTIONS.get(str(g), ""))
        columns = ["experiment_group", "description", "horizon", "feature_spec", "n_runs",
                   "models"]
        lines += [reporting.format_markdown_table(coverage[columns]), "",
                  f"Total runs recorded: **{int(len(registry))}**.", ""]
        missing = [g for g in GROUP_DESCRIPTIONS
                   if g not in set(registry["experiment_group"])]
        if missing:
            lines += ["> Experiment groups with no recorded run: "
                      + ", ".join(f"`{g}`" for g in missing) +
                      ". Their status is described in `docs/experiments.md`.", ""]

    lines += ["## 2. Dataset", ""]
    dataset_table = read_table(tables_dir / "dataset_statistics.csv")
    if dataset_table is None:
        lines += ["_`results/tables/dataset_statistics.csv` not found._", ""]
    else:
        lines += [reporting.format_markdown_table(dataset_table), ""]

    for index, (title, filename, caption, columns) in enumerate(TABLE_SECTIONS, start=3):
        lines += [f"## {index}. {title}", "", caption, ""]
        frame = read_table(tables_dir / filename)
        if frame is None or frame.empty:
            lines += [f"_`results/tables/{filename}` not available._", ""]
            continue
        subset = [c for c in columns if c in frame.columns]
        lines += [reporting.format_markdown_table(frame[subset]), "",
                  f"Full table: `results/tables/{filename}`.", ""]

    lines += ["## Figures", ""]
    present = sorted(p.name for p in figures_dir.glob("*.png")) if figures_dir.exists() else []
    if not present:
        lines += ["_No figures found. Run `python scripts/evaluate.py` to generate them._", ""]
    else:
        lookup = dict(FIGURE_CAPTIONS)
        for name in present:
            stem = name[:-4]
            lines += [f"- `results/figures/{name}` — {lookup.get(stem, 'generated figure')}",
                      f"  {stem}"]
        lines.append("")

    lines += ["## Statistical assumptions", ""]
    assumptions_path = tables_dir / "statistical_assumptions.json"
    if assumptions_path.exists():
        import json
        assumptions = json.loads(assumptions_path.read_text(encoding="utf-8"))
        for key, text in assumptions.items():
            lines += [f"- **{key.replace('_', ' ')}** — {text}"]
    else:
        lines += ["_`results/tables/statistical_assumptions.json` not found._"]
    lines.append("")

    evaluation_report = results / "metrics" / "evaluation_report.json"
    if evaluation_report.exists():
        import json
        payload: dict[str, Any] = json.loads(evaluation_report.read_text(encoding="utf-8"))
        lines += ["## Evaluation stage status", "",
                  reporting.format_markdown_table(pd.DataFrame(
                      [{"stage": k, "status": v.get("status"),
                        "detail": v.get("reason") or ", ".join(v.get("written", []))}
                       for k, v in payload.items() if isinstance(v, dict)])), ""]

    output = Path(args.output) if args.output else config.project_root_for("docs") / "RESULTS_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output} ({len(lines)} lines, {len(present)} figures referenced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
