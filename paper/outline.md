# Paper outline

Target: a conference or journal submission in the renewable-energy / applied ML
area. The structure below is conventional; every section states which artefact in
this repository it draws on, so no claim in the paper is untraceable to a computed
file.

## Title

AI/ML-Based Solar Photovoltaic Power Forecasting: A Reproducible and Explainable
Benchmark Across Forecast Horizons and Weather Conditions

## Abstract

Problem, gap, contribution, protocol, headline result, the persistence result, the
statistical caveat, limitations. Numbers come from
`results/tables/overall_model_comparison.csv` and
`results/tables/statistical_tests.csv`. The abstract must state the
information set (strictly historical, no NWP) and the single-station scope.

## 1. Introduction

* PV forecasting matters for operations; short-term horizons drive reserve and
  ramping decisions.
* Deep learning is widely reported as superior, but usually against weak baselines
  on one split with one aggregate metric.
* Contributions list (seven items, matching `README.md`).
* Scope statement: this is a controlled single-station benchmark, not a survey.

## 2. Related work

Synthesised from `paper/literature_review.md` (30 verified papers, categories A–G).
Organise by *theme*, not by paper: the move from classical ML to DL; the role of
meteorological inputs; hybrid convolutional-recurrent designs; attention and
transformers; physics-informed and domain-aware models; probabilistic forecasting
and uncertainty; explainability; multi-site transfer; reproducibility. Preserve
disagreements — see the review's "Where the literature disagrees" section.

## 3. Methodology

From `docs/methodology.md`. Research questions RQ1–RQ7, hypotheses H1–H5, the
information set, the leakage controls table, the regime taxonomy with threshold
provenance, metric definitions with the MAPE rationale, statistical procedures and
their assumptions, explainability methods and their limits, conformal calibration
and its cost.

## 4. Dataset

From `docs/dataset.md` and `results/tables/dataset_statistics.csv`. Citation,
licence, selection rationale, the LSK North choice and its nameplate cross-check,
the 50.1% zero fraction, the chronological split, the known data issues and how
each was handled.

## 5. Experimental setup

From `docs/experiments.md`. The 11 models, the fixed protocol, the experiment
matrix with per-group status, seeds, thread pinning, and the compute environment.
**State per-group execution status here**, in the setup, not in a footnote: every
group in the declared matrix was executed, and where a group is narrower than the
design -- the horizon group runs the eleven models at all four horizons, but the
feature ablation and the cross-site protocols run a documented subset -- say which
models and sites were in it rather than implying the whole grid was filled.

## 6. Results

`paper/results.md`, generated from the tables. Subsections: headline comparison;
horizon analysis; weather-regime and seasonal analysis; error analysis; feature
and model ablation; statistical comparison; explainability; uncertainty;
computational cost. Every number cited to a CSV.

## 7. Discussion

`paper/discussion.md`. The trade-offs: accuracy vs robustness vs interpretability
vs cost. Why the persistence result is the most interesting finding. Why
architecture did not pay here, and what would change that. The Diebold-Mariano
power discussion.

## 8. Research limitations

`docs/methodology.md` §13 plus the incomplete matrix and the single seed.

## 9. Conclusion

What the benchmark establishes, what it cannot, and the three next experiments
that would matter most.

## 10. References

`paper/references.bib` (generated from the verified corpus, with DOIs).

## Figures

Copies in `paper/figures/`, referenced by their `results/figures/` names. Caption
each from the corresponding entry in `scripts/generate_report.py`.

## Tables

Numbered from `results/tables/`:

| Table | Source |
| --- | --- |
| 1 | dataset statistics |
| 2 | overall model comparison |
| 3 | horizon comparison |
| 4 | weather-regime comparison |
| 5 | seasonal comparison |
| 6 | error analysis |
| 7 | feature ablation |
| 8 | model ablation |
| 9 | statistical significance |
| 10 | computational cost |
| 11 | conformal coverage |
