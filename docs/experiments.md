# Experimental design and status

The experiment matrix is declared in `configs/experiments.yaml` rather than
hard-coded, so the set of runs is auditable in one file. This document records
what each experiment asks, how it is implemented, and its execution status.

## Status summary

Status is read from the registry (`results/experiments.csv`, rebuilt from the
per-run JSON records by `scripts/evaluate.py`).

| Group | Question | Implementation | Status |
| --- | --- | --- | --- |
| A | How strong are the persistence references? | `scripts/run_experiments.py --only A_baselines` | **executed** |
| B | Classical ML vs recurrent DL vs attention, one protocol | `--only B_ml_vs_dl` | **executed** (11/11) |
| C | Does relative performance hold across horizons? | `--only C_horizons [--horizons H]` | **partially executed** — see below |
| D | Does the ranking change across weather regimes? | post-hoc on B's predictions | **executed** (`--uncertainty`-independent stage) |
| E | Which input groups matter? | `--only E_feature_ablation` | **partially executed** — see below |
| F | What does each architectural component add? | analysis of B (backbone vs composite) | **executed** |
| G | Does performance hold across seasons? | post-hoc on B's predictions | **executed** |
| H | Which variables drive the forecasts? | `scripts/evaluate.py --explainability` | **implemented, opt-in** (requires refitting) |
| I | What is the accuracy-per-unit-cost profile? | assembled from the registry | **executed** |
| K | Can intervals be calibrated? | `scripts/evaluate.py --uncertainty` | **executed** |
| J | Does a model transfer to unseen sites? | — | **not implemented** |

### Why C and E are partial, and what was done instead

Both groups are implemented and runnable in full. The partial state is a
**time-budget** artefact of this implementation session, not a methodological
choice, and it is recorded here rather than hidden:

* **C (horizons)** — the 1-hour horizon is complete for all eleven models (in
  group B). The 15-minute, 6-hour and 24-hour runs were launched in parallel and
  sixteen runs completed before the session's compute budget was reached. The
  horizon table reads each `(model, horizon)` cell from whichever group recorded
  it, and the `source_group` column in
  `results/tables/horizon_comparison.csv` shows the provenance of every row.
  Completing the matrix needs one command per horizon:

  ```bash
  python scripts/run_experiments.py --only C_horizons --horizons 15min
  python scripts/run_experiments.py --only C_horizons --horizons 6h
  python scripts/run_experiments.py --only C_horizons --horizons 24h
  ```

  `--horizons` exists precisely so that one long group can be split across
  parallel processes without editing code.

* **E (feature ablation)** — configured for one tree model and one recurrent
  model across six input sets (12 runs). Fewer than twelve completed. The choice
  of models is documented in the configuration: a tree model and a recurrent
  model test the two structurally different ways the inputs are consumed
  (flattened window versus ordered sequence), and retraining four models six times
  would multiply the training budget without changing the answer to which input
  groups matter. Attention and transformer inputs are covered by the permutation
  importance analysis of Experiment H instead.

* **H (explainability)** is opt-in because SHAP and permutation importance need a
  fitted estimator and the repository stores no binary weights. The stage refits
  the declared models under the recorded seed, which reproduces the Experiment-B
  models. Until it is run, `results/metrics/evaluation_report.json` records
  `explainability: not_run`, so "not attempted" is visibly distinct from "run and
  found nothing".

* **J (cross-site)** is declared with its holdout panel in `configs/data.yaml`
  (`dataset.holdout_stations`) and is **not implemented**. It requires per-station
  pipeline runs, per-station scalers and a capacity-aware scoring path, which is
  a separate piece of work rather than a flag. RQ6's cross-site half is therefore
  open, and the paper says so. The six holdout stations have never been used for
  fitting or tuning, so implementing it later will not contaminate the existing
  results.

## Experiments in detail

### A — Baseline comparison (executed)

*Question.* Do the two persistence references behave differently, and by how much
does a linear fit improve on them?

*Runs.* `persistence`, `smart_persistence`, `linear_regression` at 1 hour.

*Note.* Smart persistence is the guarded clear-sky ratio form
`f(t+h) = y(t) · clip(GHI_clear(t+h)/max(GHI_clear(t), 20 W/m²), 0, 10)`, clipped
to rated capacity. The three guards are each necessary: measured on this data, the
unguarded ratio form reaches an RMSE of 142 kW against a 55 kW plant, because at
sunrise and sunset the reference approaches zero while the previous reading need
not.

### B — Machine learning versus deep learning (executed, headline)

*Question.* Under one protocol, one split and one input set, does a recurrent or
attention architecture beat well-tuned gradient boosting?

*Runs.* All eleven models at 1 hour with the full input set. Every other
experiment is read relative to this one.

*Controls.* Identical split, identical feature list, identical lookback, identical
training protocol, seed 42, thread count pinned to 1.

### C — Forecast horizons (partially executed)

*Question.* Is relative performance consistent across horizons, and does any
architectural advantage shrink as the horizon grows?

*Design.* Every model at 15 minutes, 1 hour, 6 hours and 24 hours. All four
horizons are exact integer multiples of the 15-minute working resolution, so the
lookback window spans the same diurnal phase at every horizon and the only thing
that varies is the gap being predicted.

*Analysis.* `results/tables/horizon_comparison.csv` plus
`fig07_horizon_comparison.png`: RMSE, capacity-normalised error and skill against
the two persistence references.

### D — Weather regimes (executed)

*Question.* Does the ranking of models change across weather regimes?

*Implementation.* Post-hoc stratification of the stored test predictions by the
regime label at the **target** timestamp, with the persistence reference rescored
inside each stratum. Stratifying a skill score without rescoring the reference in
the same stratum would be meaningless, since a regime in which persistence is
nearly perfect cannot hide behind a high absolute error.

*Outputs.* `weather_regime_comparison.csv`, `fig08_weather_regime_comparison.png`.

### E — Feature-group ablation (partially executed)

*Question.* Which input groups carry the signal: weather, solar geometry, PV
history, calendar, or almost nothing?

*Design.* Each group removed in turn, plus a `minimal` set (current power, GHI and
temperature only). Percentages are reported relative to the same model with the
full input set, so they are directly comparable between models.

*Outputs.* `feature_ablation.csv`, `fig20_feature_ablation.png`.

### F — Model ablation (executed)

*Question.* What does each architectural component contribute?

*Implementation.* Comparison of the composite architectures against the best of
their recurrent backbones within Experiment B. The composites and the backbones
were trained under an identical seed, split and input set, so a difference in
accuracy is attributable to the architecture. No significance is claimed here; the
paired test on the same rows is in the statistical tables.

*Why it is not re-trained.* Re-running these five models under their own group id
would reproduce byte-identical runs and cost about 40 minutes of CPU for no
additional information. The configuration marks the group `analysis_only` for this
reason.

### G — Seasonal generalisation (executed)

*Question.* Does performance hold across seasons, and does it degrade on a test
year that follows the training period?

*Implementation.* Post-hoc stratification of the stored test predictions. Because
the test period is a full calendar year, every season appears exactly once and the
season mix cannot be confounded with the split.

*Outputs.* `seasonal_comparison.csv`, `fig09_seasonal_comparison.png`.

### H — Explainability (implemented, opt-in)

*Question.* Which variables dominate the forecast, and do tree and sequence models
agree?

*Implementation.* SHAP for XGBoost; grouped permutation importance for the neural
models. Attention weights are deliberately not treated as explanations. Both
methods are predictive attributions, not causal claims, and this is stated in the
figure titles and the output JSON.

*Command.* `python scripts/evaluate.py --explainability`

### I — Computational cost (executed)

*Question.* What is the accuracy-per-unit-cost profile, and which models are
Pareto-optimal?

*Implementation.* Assembled from the recorded training duration, per-window
inference latency and parameter count, with a Pareto flag. BLAS and OpenMP thread
counts are pinned to 1 for every run so the durations are comparable *within* a
machine; absolute values are machine-specific and only the ranking is portable.
This is stated in the table itself.

*Outputs.* `computational_cost.csv`, `fig17_cost_accuracy.png`.

### K — Uncertainty (executed)

*Question.* Can a deterministic model be given well-calibrated intervals at
acceptable cost?

*Implementation.* Split conformal prediction, symmetric and unadapted, using the
finite-sample `ceil((n+1)(1-α))/n` quantile of the absolute residuals.

**The calibration choice and its cost, stated plainly.** A conformal interval is
valid under exchangeability between calibration and evaluation residuals. The
honest options were to calibrate on the validation split — which requires refitting
every model and re-predicting the validation period — or to calibrate on an initial
block of the test period. The implemented option is the second, so that intervals
can be derived from the stored test predictions without refitting. The consequence
is that the coverage guarantee is **approximate**: the calibration block is earlier
in the year than the evaluation block, so under seasonal drift the two are not
exchangeable. The reported coverage is therefore an empirical description of
interval behaviour, not a proof of validity, and the caveat is carried in the output
record, in the figure and here.

*Outputs.* `uncertainty_coverage.csv`, `fig18_conformal_intervals.png`.

## Mandatory error analysis (executed)

Aggregate metrics are not sufficient, so every model's daylight errors are
stratified by regime, season, time-of-day band, generation level (as a fraction of
rated capacity) and a ramping flag. Strata with fewer than 200 daylight
observations are reported as under-sampled rather than summarised, because a metric
computed on a handful of points is noise presented as a result.

*Outputs.* `error_analysis.csv`, `fig16_intraday_error_profile.png`,
`fig02_forecast_error_over_time.png`, `fig15_error_distribution_by_model.png`,
`fig11_residual_distribution.png`.
