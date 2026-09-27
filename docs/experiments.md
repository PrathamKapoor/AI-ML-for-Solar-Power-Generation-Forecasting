# Experiments

The experiment matrix is declared in `configs/experiments.yaml` rather than
hard-coded, so the set of runs is auditable in one file. This document records
what each experiment asks, how it is implemented, and **its executed status**.

Status is read from the artefacts, not asserted:
`scripts/evaluate.py` rebuilds the registry from the per-run JSON records,
`scripts/generate_report.py` regenerates the coverage table, and
`tools/check_consistency.py` fails the build if the documentation and the
results disagree.

## Status summary

| Group | Question | Implementation | Status | Records |
| --- | --- | --- | --- | ---: |
| A | How strong are the persistence references? | training runs | executed | 3 |
| B | Classical ML vs recurrent DL vs attention | training runs | executed | 11 |
| C | Does relative performance hold across four horizons? | training runs | **executed, complete** | 44 |
| D | Does the ranking change across weather regimes? | analysis of B predictions | executed | — |
| E | What is each input group worth (leave-one-out)? | training runs | executed | 12 |
| F | What does each architectural component add? | analysis of B | executed | — |
| G | Does performance hold across seasons? | analysis of B predictions | executed | — |
| H | Which variables drive the forecasts? | refit + SHAP / permutation | executed | — |
| I | What is the accuracy-per-cost profile? | assembly from the registry | executed | — |
| J | Does a model transfer to unseen sites? | `scripts/run_cross_site.py` | executed | 60 |
| K | Can intervals be calibrated? | split conformal on stored predictions | executed | — |
| N | What is each information type worth alone? | training runs | executed | 25 |
| M | How much of a neural score is initialisation? | repeated seeds | executed | 17 |

Total training records: **103**. Canonical result table:
`results/tables/final_experiment_matrix.csv` (**454 rows**).

## Experiments in detail

### A — Baseline comparison (executed)

Persistence, smart persistence and linear regression at 1 h.

Smart persistence is the guarded clear-sky ratio
`f(t+h) = y(t) · clip(GHI_clear(t+h)/max(GHI_clear(t), 20 W/m²), 0, 10)`, clipped to
rated capacity. The three guards are each necessary: measured on this data, the
unguarded ratio reaches an RMSE of 142 kW against a 55 kW plant, because at
sunrise and sunset the reference approaches zero while the previous reading need
not. The reference is not a formality — it is the benchmark that four of the
learned models fail to beat at 15 minutes.

### B — Machine learning versus deep learning (executed, headline)

All 11 models at 1 h with the full input set, on an identical split, an identical
feature list, one seed and pinned thread counts. Every other experiment is read
relative to this one.

### C — Forecast horizons (executed, complete: 44 runs)

All 11 models at 15 min, 1 h, 6 h and 24 h. All four horizons are exact integer
multiples of the 15-minute working resolution, so the lookback window spans the
same diurnal phase at every horizon and the only thing that varies is the gap
being predicted.

**This is the experiment that changed the study's conclusion.** At 1 h, four
models fail to beat persistence; at 15 minutes the failure is far worse
(LSTM skill −0.65); at 6 hours every learned model reaches skill 0.53–0.59
because persistence itself has collapsed. A single-horizon benchmark cannot know
whether its reference is strong.

Outputs: `horizon_comparison.csv`, `fig07_horizon_comparison.png`.

### D — Weather regimes (executed)

Post-hoc stratification of the stored test predictions by the regime label at the
*target* timestamp, with the persistence reference **rescored inside each
stratum**. Stratifying a skill score without rescoring the reference in the same
stratum would be meaningless, since a regime in which persistence is nearly
perfect cannot hide behind a high absolute error.

The decisive observation: absolute error is *lowest* under cloudy conditions
(persistence nRMSE 0.104 against 0.185 in clear sky) because the signal amplitude
is small, while skill is *also* lowest there (gradient boosting +0.024 against
+0.358). Reading the error column alone inverts the answer to the question an
operator actually asks.

Outputs: `weather_regime_comparison.csv`, `fig08_weather_regime_comparison.png`.

### E — Feature-group ablation, leave-one-out (executed, 12 runs)

Each input group removed in turn, for one tree model and one recurrent model.
Percentages are relative to the same model with the full input set.

### N — Additive input regimes (executed, 25 runs)

The complementary direction: a model given *only* one kind of information. Five
regimes (PV history only, weather only, both, both plus geometry, full) × five
models spanning both access patterns.

This is where the most useful finding of the study came from, and it is one the
leave-one-out table cannot show: **for the tree models the weather variables add
nothing** (15 PV-history features reach nRMSE 0.119 against 0.121 for all 33),
**while for the recurrent models they are decisive** (LSTM goes from 0.176 with PV
history alone to 0.165 with weather, and its skill from −0.058 to +0.008). The
value of a feature group depends on how the model reads it, which is why a single
global importance ranking is not sufficient.

Outputs: `feature_regimes.csv`, `fig23_feature_regimes.png`.

### F — Model ablation (executed)

Composite architectures against the best of their recurrent backbones within
Experiment B. All share a seed, a split and an input set, so a difference is
attributable to the architecture. Not re-trained: the composites and backbones
were already trained under an identical configuration, and the group is marked
`analysis_only` for that reason.

### G — Seasonal generalisation (executed)

Post-hoc stratification of the stored predictions. The test period is a full
calendar year, so each season appears exactly once and the season mix cannot be
confounded with the split.

Outputs: `seasonal_comparison.csv`, `fig09_seasonal_comparison.png`.

### H — Explainability (executed)

* **SHAP** (`TreeExplainer`, mean |value|, `tree_path_dependent`) for XGBoost,
  collapsed over the lookback, with a beeswarm summary and per-variable
  dependence plots.
* **Grouped permutation importance** for the attention-recurrent model: a variable
  is scrambled across the whole window and the daylight MAE increase recorded.
* **Regime-specific permutation importance**, measured inside clear, partly
  cloudy, cloudy and high-variability steps separately. Regimes with fewer than
  500 windows are skipped rather than summarised. The ranking genuinely changes
  between regimes.
* **Attention weights are not treated as explanations**, and the outputs say so.
* Scope is one model per access pattern. Adding the LSTM and Transformer would
  double the cost of the stage for the same conclusion.

Outputs: `importance_xgboost.csv`, `importance_attention_lstm.csv`,
`importance_regime_attention_lstm.csv`, `feature_importance.csv`,
`dependence_attention_lstm.csv`, and the corresponding figures.

### I — Computational cost (executed)

Training duration, per-window inference latency and parameter count with a Pareto
flag. Thread counts are pinned to 1 for every run, so durations are comparable
*within* a machine; absolute values are machine-specific and the table says so.

### J — Cross-site generalisation (executed, 60 records)

Four protocols over a fixed seven-station panel, implemented in
`src/solar_forecasting/cross_site/`. Every site is processed by the same
pipeline, so only the target and the rated capacity differ.

**The most consequential result in the study.** A model fitted at one
installation is worse than persistence at every other installation tested: skill
runs from −0.10 to −1.07 across the panel. Pooling six training sites
(leave-one-site-out) repairs transfer for five of the six holdouts but not for the
primary station itself (+0.002 for the tree models).

Leakage guards, each asserted in `tests/test_cross_site.py`: scalers fitted on
training-site rows only; early stopping never sees the held-out site; target
scaler inversion returns true watts because standardisation is affine; nRMSE
normalised by each site's own rated capacity; the panel fixed in configuration and
never re-selected after seeing transfer errors.

Scope limitation: the panel uses gradient boosting, XGBoost and linear regression.
The neural models were not included, because fitting each 20 times per model is
beyond the available compute budget. This is stated as a limitation, not hidden.

Outputs: `cross_site_comparison.csv`, `fig22_cross_site_transfer.png`.

### K — Uncertainty (executed)

Split conformal prediction at α ∈ {0.1, 0.2}, calibrated on the first 31 days of
the test period.

**The calibration trade-off, stated plainly.** A conformal interval is valid under
exchangeability between calibration and evaluation residuals. The honest options
were to calibrate on the validation split (requiring a refit of every model) or on
an initial block of the test period. The second is implemented so intervals can be
derived from stored predictions, and the consequence is that the coverage
guarantee is *approximate*: the calibration block precedes the evaluation block,
so under seasonal drift the two are not exchangeable. Reported coverage is
therefore an empirical description, not a proof of validity.

Outputs: `uncertainty_coverage.csv`, `fig18_conformal_intervals.png`.

### M — Multi-seed replication (executed, 17 records)

Every neural model repeated across seeds on the identical split and input set:
LSTM 5, GRU 5, attention-LSTM 4, CNN-LSTM 3. The default seed 42 is one of them and
is already recorded under group B, so no seed is a special case.

`multi_seed_results.csv` reports the mean, standard deviation, a t-based 95%
interval and the best and worst single seed. The best-of-seeds number is never
quoted as a performance figure.

**Why three seeds at the low end.** A single transformer run costs about
19 minutes on this hardware and the study had a bounded compute budget. Three
seeds is the documented minimum, and it is enough for the question being asked —
whether seed spread is large enough to change a conclusion — because the observed
spread is 44–253 W against a 2 900 W gap between the best and worst model.

Outputs: `multi_seed_results.csv`, `fig21_multiseed_spread.png`.

## Mandatory error analysis (executed)

Daylight errors stratified by regime, season, time-of-day band, generation level
as a fraction of rated capacity, and a ramping flag. Strata with fewer than 200
daylight observations are reported as under-sampled rather than summarised.

Outputs: `error_analysis.csv`, `fig16_intraday_error_profile.png`,
`fig02_forecast_error_over_time.png`, `fig15_error_distribution_by_model.png`,
`fig11_residual_distribution.png`.

## Temporal dependence (executed)

Autocorrelation of the forecast error to 192 lags, the integrated
autocorrelation time, and the bootstrap interval width against block length from
1 to 384 steps. These are diagnostics, not adjustments: no p-value was sought by
changing a method, a block length or a sample after the fact.

Outputs: `error_autocorrelation.csv`, `block_length_sensitivity.csv`,
`fig24_error_autocorrelation.png`, `fig25_block_length_sensitivity.png`.
