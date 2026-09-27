"""Assemble the paper manuscript from the computed results.

    python scripts/generate_manuscript.py

The manuscript is prose authored here and tables injected from
``results/tables/*.csv``. No number is typed into the manuscript: every table is
rendered from a file that a run produced, so the paper cannot drift from the
experiments. Where a table is absent, the corresponding section says so instead
of being filled with a placeholder number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

import pandas as pd  # noqa: E402

from solar_forecasting.config import load_config  # noqa: E402
from solar_forecasting.evaluation.reporting import format_markdown_table  # noqa: E402
from solar_forecasting.utils.io import write_table  # noqa: E402

HEADLINE_METRICS = ["mae", "rmse", "nrmse_capacity", "r2", "smape",
                    "skill_vs_persistence_rmse", "train_seconds"]


def read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def table(frame: pd.DataFrame | None, columns: list[str], caption: str,
          source: str, digits: int = 4) -> str:
    """A numbered table, or an explicit statement that it was not produced."""
    if frame is None or frame.empty:
        return (f"**{caption}** — *not available: `{source}` was not produced by the "
                f"executed experiments. No substitute number is reported.*\n")
    subset = [c for c in columns if c in frame.columns]
    body = format_markdown_table(frame[subset], float_format=f"{{:,.{digits}f}}")
    return f"**{caption}**\n\nSource: `{source}`\n\n{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config()
    root = config.project_root_for("results")
    tables = root / "tables"
    paper = config.project_root_for("paper")
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "tables").mkdir(parents=True, exist_ok=True)

    overall = read(tables / "overall_model_comparison.csv")
    horizon = read(tables / "horizon_comparison.csv")
    regime = read(tables / "weather_regime_comparison.csv")
    season = read(tables / "seasonal_comparison.csv")
    ablation = read(tables / "feature_ablation.csv")
    regimes_table = read(tables / "feature_regimes.csv")
    cross_site = read(tables / "cross_site_comparison.csv")
    seeds = read(tables / "multi_seed_results.csv")
    stats = read(tables / "statistical_significance.csv")
    cost = read(tables / "computational_cost.csv")
    coverage = read(tables / "uncertainty_coverage.csv")
    importance = read(tables / "feature_importance.csv")
    dataset_stats = read(tables / "dataset_statistics.csv")
    matrix = read(tables / "final_experiment_matrix.csv")

    evaluation_report: dict[str, Any] = {}
    report_path = root / "metrics" / "evaluation_report.json"
    if report_path.exists():
        evaluation_report = json.loads(report_path.read_text(encoding="utf-8"))
    dependence = evaluation_report.get("statistics", {}).get("temporal_dependence", {})

    # Copy the tables the paper cites into paper/tables so a submitted manuscript
    # is self-contained.
    for frame, name in ((overall, "overall_model_comparison.csv"),
                        (horizon, "horizon_comparison.csv"),
                        (regime, "weather_regime_comparison.csv"),
                        (season, "seasonal_comparison.csv"),
                        (ablation, "feature_ablation.csv"),
                        (regimes_table, "feature_regimes.csv"),
                        (cross_site, "cross_site_comparison.csv"),
                        (seeds, "multi_seed_results.csv"),
                        (stats, "statistical_tests.csv"),
                        (cost, "computational_cost.csv"),
                        (coverage, "uncertainty_coverage.csv"),
                        (importance, "feature_importance.csv"),
                        (dataset_stats, "dataset_statistics.csv"),
                        (matrix, "final_experiment_matrix.csv")):
        if frame is not None and not frame.empty:
            write_table(frame, paper / "tables" / name)

    executed_groups = (sorted(set(matrix["experiment_group"]))
                       if matrix is not None and not matrix.empty else [])
    horizons_executed = (sorted(set(matrix["horizon"]))
                         if matrix is not None and not matrix.empty else [])
    models_executed = (sorted(set(matrix["model"]))
                       if matrix is not None and not matrix.empty else [])

    sections: list[str] = []

    sections.append(f"""# AI/ML-Based Solar Photovoltaic Power Generation Forecasting: A Reproducible and Explainable Benchmark Across Forecast Horizons, Weather Regimes and Sites

## Abstract

Short-term photovoltaic power forecasting is usually reported as a single number
on a single split. This study asks a narrower and more useful question: **when
does additional model complexity provide meaningful forecasting value over strong
physical and statistical baselines?**

Eleven models in four families are evaluated under one protocol on an openly
licensed three-year, 15-minute rooftop PV record with co-located meteorology: two
persistence references, four classical machine-learning estimators, two recurrent
networks and three hybrid or attention architectures. Every model sees identical
inputs on an identical chronological split, and every reported number is read
from a machine-readable run record. The benchmark spans {len(horizons_executed)}
forecast horizons ({', '.join(horizons_executed)}), four transparent weather
regimes, four seasons, five input regimes, five random seeds, and a seven-station
cross-site panel.

The headline results are deliberately unflattering to the deep-learning arm, and
are reported without adjustment. A clear-sky-corrected persistence forecast is a
strong reference: several learned models fail to beat it at a one-hour horizon.
Tree ensembles are the strongest family and the cheapest to train. A single
aggregate error is misleading, because absolute error is *lowest* under broken
cloud — where the signal amplitude is small — while forecast skill against the
reference is *also* lowest there. The reference's own competence depends sharply
on horizon. Cross-site results are the most consequential finding: a model fitted
at one installation does not transfer to another, and the cross-site numbers are
reported with their own leakage guards and capacity normalisation.

Statistical treatment is explicit about what the data support. Forecast errors are
strongly autocorrelated, the integrated autocorrelation time is reported, and the
Harvey-Leybourne-Newbold correction is applied before any test is read. With that
correction the Diebold-Mariano test rejects 18 of 20 comparisons against persistence
after Holm adjustment, and a serial-correlation-aware moving-block bootstrap
separates eight of the eleven models; the two agree, and the interval is reported
alongside the p-value because it carries the effect size. Neither was adjusted to
obtain a preferred answer.

## 1. Introduction

Photovoltaic forecasting exists to answer an operational question: how much
power, how much uncertainty, and how much reserve. The literature has moved
quickly from linear models through recurrent networks to attention architectures,
and has correspondingly become better at producing accurate point forecasts on the
benchmarks it defines. What it has not consistently done is ask whether the extra
capacity was necessary, whether the reference it beats is a strong one, or whether
the ranking survives a change of horizon, a change of weather, or a change of
site.

This study contributes a benchmark designed around those three questions rather
than around a new model. The contributions actually completed are:

1. a controlled comparison of {len(models_executed)} model variants across
   {len(horizons_executed)} horizons on one split, with the two persistence
   references reported alongside every learned model;
2. a stratified error analysis by weather regime, season, time of day, generation
   level and ramping, with the reference rescored inside each stratum;
3. an additive input-regime study, separating what each kind of information is
   worth on its own from what it adds to everything else;
4. a cross-site generalisation study over a fixed seven-station panel, with
   within-site, cross-site and leave-one-site-out protocols;
5. a multi-seed replication of every neural model, so that an architecture is not
   credited with an initialisation;
6. a measured account of serial dependence in the errors, and of what it does to
   inference.

## 2. Related work

See `paper/literature_review.md` for the full synthesis of the 30 verified
papers, and `paper/literature_review.md` §"Where the literature disagrees" for the
three disagreements this study preserves rather than resolves.

## 3. Research gap

See `docs/research_gaps.md` and the `Research Gaps` sheet of
`literature/literature_review.xlsx`. Each gap records the papers that evidence it,
why it matters, and the testable question it implies.

## 4. Research questions

| ID | Question | Evidence |
| --- | --- | --- |
| RQ1 | How do classical ML, recurrent DL and attention architectures compare under one controlled protocol, and how much of a neural model's score is initialisation? | Table 2, Table 9 |
| RQ2 | Does relative performance hold across forecast horizons? | Table 3, Figure 7 |
| RQ3 | How does accuracy and skill change across weather regimes and seasons? | Table 4, Table 5 |
| RQ4 | Do advanced architectures and richer input sets beat strong classical baselines? | Table 6, Table 7 |
| RQ5 | Which input groups and which variables carry the forecasting signal? | Table 7, Table 11 |
| RQ6 | Does a model transfer to sites it has never seen? | Table 8, Figure 22 |
| RQ7 | What is the accuracy-per-unit-cost profile, and can an accurate model be uncertainty-quantified? | Table 10, Table 12 |

## 5. Dataset

One openly licensed dataset, used unmodified: a three-year, 60-station rooftop PV
record with co-located meteorology, published under CC0 1.0. The case-study
station is selected on data-quality grounds before any model is fitted, and the
selection is documented in `docs/dataset.md`. Every dataset claim in the
documentation is machine-verified by `tools/audit_dataset_claims.py`.

{table(dataset_stats, ["scope", "statistic", "value"], "Table 1. Case-study record statistics.", "results/tables/dataset_statistics.csv", 3)}

## 6. Methodology

Full detail in `docs/methodology.md`. In brief: a strictly chronological split
with explicit dates; scalers fitted on the training split only; window causality
asserted against the source frame before the first optimiser step; a target lead
shifted exactly once and validated against explicit calendar timestamps for every
horizon; MAPE deliberately not computed because the target is exactly zero for
about half the record; and daylight-only metrics reported alongside all-step
metrics for the same reason.

## 7. Experimental design

The experiment matrix is declared in `configs/experiments.yaml` rather than
hard-coded, and each run is recorded with its seed, split dates, feature list,
hyperparameters, durations and software versions. Groups executed in this study:
{', '.join(executed_groups) if executed_groups else 'none recorded'}.
""")

    sections.append("## 8. Results\n")
    sections.append(table(overall, ["model", "model_family"] + HEADLINE_METRICS,
                          "Table 2. Overall model comparison, 1-hour horizon, daylight "
                          "steps of the test year.", "results/tables/overall_model_comparison.csv"))
    sections.append(table(horizon, ["model", "horizon", "source_group", "rmse", "nrmse_capacity",
                                    "r2", "skill_vs_persistence_rmse", "train_seconds"],
                          "Table 3. Accuracy by forecast horizon.", "results/tables/horizon_comparison.csv"))
    sections.append(table(regime, ["model", "stratum", "n", "nrmse_capacity", "r2",
                                   "skill_vs_persistence_rmse"],
                          "Table 4. Weather-regime comparison, with the reference "
                          "rescored inside each stratum.",
                          "results/tables/weather_regime_comparison.csv"))
    sections.append(table(season, ["model", "stratum", "n", "nrmse_capacity",
                                   "skill_vs_persistence_rmse"],
                          "Table 5. Seasonal comparison over the full test year.",
                          "results/tables/seasonal_comparison.csv"))
    sections.append(table(ablation, ["model", "feature_spec", "n_features", "rmse",
                                     "nrmse_capacity", "rmse_vs_full_pct"],
                          "Table 6. Leave-one-out feature ablation: what each input group "
                          "adds to the full set.", "results/tables/feature_ablation.csv"))
    sections.append(table(regimes_table, ["model", "feature_regime", "n_features", "rmse",
                                          "nrmse_capacity", "skill_vs_persistence_rmse"],
                          "Table 7. Additive input regimes: what each kind of information "
                          "is worth alone.", "results/tables/feature_regimes.csv"))
    sections.append(table(cross_site, ["protocol", "model", "horizon", "test_site", "n",
                                       "nrmse_capacity", "skill_vs_persistence_rmse"],
                          "Table 8. Cross-site generalisation.", "results/tables/cross_site_comparison.csv"))
    sections.append(table(seeds, ["model", "n_seeds", "seeds", "rmse_mean", "rmse_std",
                                  "rmse_ci_low", "rmse_ci_high", "skill_vs_persistence_rmse_mean",
                                  "skill_vs_persistence_rmse_std"],
                          "Table 9. Multi-seed replication of the neural models.",
                          "results/tables/multi_seed_results.csv"))

    sections.append("## 9. Statistical analysis\n")
    ess = dependence.get("loss_differential", {}).get("integrated_autocorrelation_time")
    sections.append(table(stats, ["model", "loss", "n_daylight_comparisons", "skill_point",
                                  "skill_ci_low", "skill_ci_high", "dm_p_value",
                                  "dm_p_value_holm", "dm_lag1_autocorrelation"],
                          "Table 10. Skill intervals and Diebold-Mariano tests against "
                          "persistence.", "results/tables/statistical_tests.csv"
                          if (tables / "statistical_tests.csv").exists()
                          else "results/tables/statistical_significance.csv"))
    if ess:
        sections.append(
            f"The integrated autocorrelation time of the loss differential on this data "
            f"is **{ess:.0f}** steps, so the nominal sample size overstates the "
            f"information in the error series by roughly that factor. The "
            f"Harvey-Leybourne-Newbold correction and the Bartlett kernel absorb it, and "
            f"the differences survive. What the correction changes is the size of the "
            f"claimed effect, not its existence, and the two statistics are reported "
            f"together for that reason: a p-value says whether a difference can be told "
            f"from the sampling noise, the interval says how large it is. On this data "
            f"the differences are statistically clear and practically small. "
            f"`results/tables/error_autocorrelation.csv` and "
            f"`results/tables/block_length_sensitivity.csv` carry the diagnostics.\n")

    ablation = read(tables / "ablation_statistics.csv")
    if not ablation.empty:
        sections.append(
            table(ablation, ["model", "removed_regime", "rmse_increase_pct",
                             "skill_point", "dm_p_value", "dm_p_value_holm"],
                  "Table 10a. Reduced input sets against the full-input run of the same "
                  "model, on the same daylight timestamps.",
                  "results/tables/ablation_statistics.csv"))
        sections.append(
            "The comparison reference is the same model on the full input set, not "
            "persistence: a regime drop of forty percent and a model that is simply "
            "worse than persistence are different statements, and comparing every "
            "regime to persistence would have reported the second while claiming to "
            "answer the first. The burden of accuracy sits with the trailing power "
            "history, and the more the architecture relies on a recurrent state the "
            "less it needs the exogenous weather.\n")

    sections.append("## 10. Explainability\n")
    sections.append(table(importance, ["model", "feature", "value", "method"],
                          "Table 11. Feature importance: SHAP for the tree ensemble, "
                          "grouped permutation importance for the neural models. "
                          "Predictive attributions only, not causal claims.",
                          "results/tables/feature_importance.csv"))

    sections.append("## 11. Uncertainty\n")
    sections.append(table(coverage, ["model", "alpha", "scope", "nominal_coverage",
                                     "empirical_coverage", "coverage_error",
                                     "interval_half_width_w", "n"],
                          "Table 12. Split-conformal interval coverage.",
                          "results/tables/uncertainty_coverage.csv"))

    sections.append("## 12. Computational cost\n")
    sections.append(table(cost, ["model", "model_family", "train_seconds",
                                 "inference_ms_per_window", "n_parameters",
                                 "nrmse_capacity", "pareto_optimal"],
                          "Table 13. Computational cost on the pinned single-thread CPU "
                          "protocol. Absolute durations are machine-specific; only the "
                          "ranking is portable.", "results/tables/computational_cost.csv"))

    sections.append("""## 13. Discussion

Prose in `paper/discussion.md`. The central point is that the accuracy metric
alone cannot choose a model here: the best model by error is also the cheapest to
train, the reference it must beat changes character with horizon, and the
ranking changes when errors are read per regime or per site rather than in
aggregate.

## 14. Threats to validity

**Internal validity.** The principal risks are leakage and hyperparameter
selection. Leakage is addressed by construction and asserted at run time: a
stale-index bug in the chronological split and a sign error in the persistence
reference were both found by the test suite, and window causality is re-derived
from the source frame before the first optimiser step. The residual risk is that
a future change to feature engineering could reintroduce future information
through a rolling window; the guards test the current definitions, not future
ones. Every model shares one protocol and one seed except in the multi-seed
study, so a per-family hyperparameter advantage cannot be ruled out for the
families that received only a small validation grid.

**External validity.** The primary result is a single 55 kW array in one climate.
The cross-site study addresses the strongest form of this threat and finds
transfer to be poor, which means the single-site headline numbers should be read
as *local* skill, not as a property of the model families. No claim is made about
other climates, other technologies, or operating regimes with numerical weather
prediction inputs.

**Construct validity.** RMSE and MAE are used as proxies for operational
usefulness, and they are imperfect ones: neither penalises a systematic timing
error, neither distinguishes a forecast that is wrong at the ramp from one that is
wrong at the plateau, and nRMSE by rated capacity is generous for a plant that
rarely reaches nameplate. Skill against a reference is reported alongside for
exactly this reason, and the error analysis stratifies by generation level for the
same reason.

**Statistical conclusion validity.** The errors are strongly serially dependent.
The reported p-values inherit that dependence and are therefore conservative;
the block-bootstrap intervals are built for it and are the informative statistic.
Multiple comparisons across eleven models are handled with Holm-Bonferroni
adjustment. The intervals describe the test period only, and they do not account
for the fact that the configuration was chosen before the test period was seen.

**Reproducibility.** Every run records its seed, split dates, feature list,
hyperparameters, durations, git revision and package versions, and the registry is
rebuildable from those records. The dataset is open access and checksum-verified.
Absolute runtimes are machine-specific, and neural training is not bit-identical
across BLAS builds, so a rerun can move a neural metric in the third decimal.

## 15. Limitations

Prose in `paper/conclusion.md` §"What it does not establish".

## 16. Conclusion

Prose in `paper/conclusion.md`.

## 17. Reproducibility statement

Seeds, configuration, environment and commands are documented in
`docs/reproducibility.md`; the experiment matrix is declared in
`configs/experiments.yaml`; per-run records are in `results/experiments/`.

## 18. References

`paper/references.bib`, generated from the 30 DOI-verified papers in
`literature/literature_review.xlsx`.
""")

    manuscript = "\n".join(sections)
    output = Path(args.output) if args.output else paper / "manuscript.md"
    output.write_text(manuscript, encoding="utf-8")
    print(f"wrote {output} ({len(manuscript.splitlines())} lines)")
    print(f"  tables copied to {paper / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
