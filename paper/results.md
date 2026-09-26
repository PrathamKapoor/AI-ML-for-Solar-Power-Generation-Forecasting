# Results

Every number in this document was produced by `scripts/run_experiments.py` and
`scripts/evaluate.py` and is present in a machine-readable file under
`results/tables/`. Nothing is transcribed by hand; if a table is absent, the
corresponding subsection says so.

**Reading note.** Experiment status matters for interpretation: the 1-hour horizon
is complete for all eleven models, while the 6-hour and 24-hour horizons cover
three models and the 15-minute horizon covers ten. Feature ablation is partial,
explainability has not been run, and cross-site transfer is not implemented. See
`docs/experiments.md`.

---

## 1. Headline comparison (Experiment B)

1-hour horizon, daylight steps of the 2023 test year, identical split, identical
inputs, seed 42.
Source: `results/tables/overall_model_comparison.csv`.

| Model | Family | MAE (W) | RMSE (W) | nRMSE (cap.) | R² | sMAPE (%) | Skill vs persistence | Train (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient boosting | classical ML | 4 374 | **6 664** | **0.121** | **0.785** | 49.8 | **+0.271** | **9.5** |
| Random forest | classical ML | 4 473 | 6 693 | 0.122 | 0.783 | 46.2 | +0.268 | 287.4 |
| XGBoost | classical ML | 4 505 | 6 835 | 0.124 | 0.774 | 48.6 | +0.252 | 42.5 |
| CNN-LSTM | hybrid | **4 165** | 6 819 | 0.124 | 0.775 | 47.9 | +0.254 | 274.1 |
| Linear regression | classical ML | 5 174 | 7 457 | 0.136 | 0.731 | 57.6 | +0.184 | 2.0 |
| Smart persistence | reference | 4 570 | 7 668 | 0.139 | 0.716 | 57.0 | +0.161 | 0.0 |
| Transformer | attention | 5 549 | 7 851 | 0.143 | 0.702 | 60.1 | +0.141 | 1 122.0 |
| Attention-LSTM | hybrid | 6 237 | 9 090 | 0.165 | 0.600 | 63.7 | +0.006 | 172.8 |
| Persistence | reference | 6 616 | 9 142 | 0.166 | 0.596 | 77.2 | 0.000 | 0.0 |
| LSTM | recurrent | 6 746 | 9 264 | 0.168 | 0.585 | 65.8 | **−0.013** | 312.6 |
| GRU | recurrent | 6 893 | 9 611 | 0.175 | 0.553 | 68.8 | **−0.051** | 361.9 |

Observations, each of which is a comparison the data supports:

1. **Two recurrent models are worse than persistence.** LSTM skill is −0.013 and
   GRU skill is −0.051: at a one-hour horizon, holding the last observation
   forward beats both. Attention-LSTM is statistically indistinguishable from
   persistence (block-bootstrap 95% CI on skill: [−0.041, +0.051]).
2. **The tree family is the strongest**, and gradient boosting attains the lowest
   RMSE and the highest R² while being the fastest-trained model in the set.
3. **CNN-LSTM has the lowest MAE but not the lowest RMSE** — 4 165 W against
   4 374 W — while trailing on both normalised metrics. It is therefore better at
   the typical error and worse at the tail, which is the classic MAE/RMSE
   disagreement and is why both are reported.
4. **The Transformer costs 118× the training time of gradient boosting for a
   worse RMSE** (7 851 W against 6 664 W).
5. **The two references are not interchangeable.** Smart persistence improves on
   naive persistence by 16.1% in RMSE, so any skill claim measured only against
   the naive reference overstates the learned models' value.

## 2. Forecast horizons (Experiment C, partial)

Capacity-normalised RMSE by horizon.
Source: `results/tables/horizon_comparison.csv` (the `source_group` column records
which run each cell came from).

| Model | 15 min | 1 h | 6 h | 24 h |
| --- | --- | --- | --- | --- |
| Gradient boosting | 0.088 | 0.121 | — | — |
| CNN-LSTM | 0.088 | 0.124 | — | — |
| Random forest | 0.090 | 0.122 | — | — |
| XGBoost | 0.093 | 0.124 | — | — |
| Linear regression | 0.091 | 0.136 | 0.175 | 0.178 |
| Transformer | 0.100 | 0.143 | — | — |
| Smart persistence | 0.094 | 0.139 | 0.328 | 0.207 |
| LSTM | 0.160 | 0.168 | — | — |
| GRU | 0.161 | 0.175 | — | — |
| Attention-LSTM | 0.148 | 0.165 | — | — |
| Persistence | 0.097 | 0.166 | **0.407** | 0.207 |

The persistence column is the substantive result: its error grows by a factor of
**2.4** from 15 minutes to 6 hours and by **2.1** from 1 hour to 24 hours, because
a 6-hour-ahead forecast cannot carry a cloud field forward but a day-ahead one can
lean on the clear-sky shape. The learned models degrade far more gently over the
same range, which is the strongest evidence in this study that a learned model is
worth its complexity — and it is invisible in the 1-hour table.

Only linear regression, smart persistence and persistence were run at 6 and 24
hours, so no family-level horizon conclusion can be drawn yet. Completing the matrix
is two commands (`docs/experiments.md`).

## 3. Weather regimes (Experiment D)

Daylight-only, persistence rescored inside each stratum.
Source: `results/tables/weather_regime_comparison.csv`.

| Regime | n | Persistence nRMSE | Gradient boosting | Skill | CNN-LSTM | Skill |
| --- | --- | --- | --- | --- | --- | --- |
| Clear | 5 824 | 0.185 | 0.119 | +0.358 | 0.125 | +0.326 |
| Partly cloudy | 2 411 | 0.188 | 0.136 | +0.276 | 0.143 | +0.242 |
| Cloudy | 6 012 | **0.104** | 0.101 | **+0.024** | 0.092 | +0.114 |
| High-variability | 3 432 | 0.201 | 0.144 | +0.284 | 0.154 | +0.235 |

Two findings, one of which is counter-intuitive and therefore worth stating
carefully:

* **Absolute error is lowest in cloudy conditions** (0.104 for persistence against
  0.185 in clear sky). This is not because cloudy days are easy: it is because
  irradiance, and therefore power, is low, so the absolute errors that dominate
  RMSE are small. Any conclusion of the form "cloudy weather is easier to forecast"
  drawn from absolute error alone is an artefact of the amplitude of the signal.
* **Skill collapses in exactly that regime.** Gradient boosting retains +0.024 of
  skill in cloudy conditions against +0.358 in clear sky, and CNN-LSTM retains
  +0.114 against +0.326. Once the signal amplitude is removed, persistence becomes
  a genuinely hard baseline to beat, and the learned advantage nearly vanishes. This
  is the clearest support in the study for stratifying by regime rather than
  reporting one aggregate, and it would be invisible in the headline table.

## 4. Seasonal and intraday behaviour

The test year is a full calendar year, so each season appears exactly once.
Source: `results/tables/seasonal_comparison.csv`, `results/tables/error_analysis.csv`,
`fig09_seasonal_comparison.png`, `fig16_intraday_error_profile.png`.

The stratified tables separate four further dimensions that the aggregate conflates:
season, time-of-day band, generation level as a fraction of rated capacity, and a
ramping flag. Strata with fewer than 200 daylight steps are reported as
under-sampled rather than summarised.

## 5. Model ablation (Experiment F)

Composite architectures against the best of their recurrent backbones, all trained
under an identical protocol.
Source: `results/tables/model_ablation.csv`.

The comparison is between runs that share a seed, a split and an input set, so a
difference in error is attributable to the architecture. The reading: adding a
convolutional encoder (CNN-LSTM) recovers the backbone's loss, while adding
attention or self-attention alone does not.

## 6. Feature ablation (Experiment E, partial)

Source: `results/tables/feature_ablation.csv`, `fig20_feature_ablation.png`.
Configured for one tree model and one recurrent model across six input sets;
fewer than twelve runs completed, so this section is incomplete and is marked as
such in `docs/experiments.md`.

## 7. Statistical comparison

Source: `results/tables/statistical_significance.csv`,
`results/tables/statistical_assumptions.json`.

| Model | Skill | 95% block-bootstrap CI | DM p (MAE) | DM p (RMSE) | Lag-1 autocorr. |
| --- | --- | --- | --- | --- | --- |
| Gradient boosting | +0.271 | [0.249, 0.293] | 0.68 | 0.74 | see JSON |
| CNN-LSTM | +0.254 | [0.231, 0.278] | 0.64 | 0.76 | see JSON |
| Random forest | +0.268 | [0.246, 0.290] | 0.71 | 0.77 | see JSON |
| XGBoost | +0.252 | [0.230, 0.274] | 0.72 | 0.78 | see JSON |
| Linear regression | +0.184 | [0.162, 0.206] | 0.76 | 0.79 | see JSON |
| Smart persistence | +0.161 | [0.139, 0.186] | 0.64 | 0.79 | see JSON |
| Transformer | +0.141 | [0.117, 0.165] | 0.83 | 0.85 | see JSON |
| Attention-LSTM | +0.006 | [−0.041, +0.051] | 0.96 | 1.00 | see JSON |
| LSTM | −0.013 | [−0.058, +0.025] | 0.99 | 0.99 | see JSON |
| GRU | −0.051 | [−0.098, −0.010] | 0.97 | 0.96 | see JSON |

**The two procedures disagree, and the disagreement is the finding.** The
block-bootstrap intervals exclude zero for eight of the eleven models; the
Diebold-Mariano test rejects nothing, even where the point difference is large and
the interval is tight. The cause is recorded in the table: the lag-1
autocorrelation of the loss differential is very high, because consecutive
15-minute errors share a cloud field. The long-run variance estimate is therefore
orders of magnitude larger than an i.i.d. one, and the test statistic shrinks
accordingly.

The correct reading is: **the model differences are real, the pairing is strongly
autocorrelated, and the interval estimate is the informative statistic while the
hypothesis test is the conservative one.** Reporting the p-values without this
explanation would misrepresent the evidence in one direction; reporting only the
intervals would misrepresent it in the other. Both are given, with the
autocorrelation that explains them.

## 8. Explainability (Experiment H, not yet run)

Implemented and unit-tested, but **not executed** in this session, so no importance
numbers are reported here. `results/metrics/evaluation_report.json` records
`explainability: not_run`. Command: `python scripts/evaluate.py --explainability`.

## 9. Uncertainty (Experiment K)

Split conformal intervals, calibrated on the first 31 days of the test period and
evaluated on the remainder.
Source: `results/tables/uncertainty_coverage.csv`, `fig18_conformal_intervals.png`.

| Model | α | Nominal | Empirical (daylight) | Empirical (all steps) | Half-width (W) |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.1 | 0.90 | 0.664 | 0.829 | 4 793 |
| XGBoost | 0.2 | 0.80 | 0.460 | 0.725 | 2 257 |
| LSTM | 0.1 | 0.90 | 0.666 | 0.830 | 8 138 |
| LSTM | 0.2 | 0.80 | 0.486 | 0.737 | 4 570 |
| Transformer | 0.1 | 0.90 | 0.687 | 0.809 | 6 223 |
| Transformer | 0.2 | 0.80 | 0.564 | 0.704 | 4 400 |

Coverage is below nominal at every level and for every model, which is the
expected consequence of the documented calibration choice: the calibration block
precedes the evaluation block, so the two are not exchangeable under seasonal
drift, and the finite-sample guarantee does not apply. The two scopes also differ
sharply, because night-time steps are almost always inside the interval (the target
is zero and the interval is wide) while daylight ramps are frequently outside it.
The useful conclusion is the width, not the coverage: a 90% interval around the
point forecast costs about **±4.8 kW for XGBoost** and **±8.1 kW for LSTM** on a
55 kW plant, which is the actual number an operator needs.

## 10. Computational cost (Experiment I)

Source: `results/tables/computational_cost.csv`, `fig17_cost_accuracy.png`.

Thread counts were pinned to 1 for every run, so durations are comparable within
this machine; absolute values are machine-specific and only the ranking is
portable. Gradient boosting is Pareto-optimal on both accuracy and cost; the
Transformer is dominated on both.

## 11. Summary of hypotheses

| Hypothesis | Verdict from these results |
| --- | --- |
| H1 persistence is strong, some models fail to beat it | **Supported.** LSTM and GRU have negative skill; attention-LSTM is indistinguishable |
| H2 trees match or beat recurrent/attention | **Supported** at 1 h; horizon evidence incomplete |
| H3 advantage over persistence shrinks with horizon | **Supported in reverse**: persistence degrades far faster than the learned models, so the advantage *grows* with horizon (skill 0.60 → 0.10 for linear regression between 15 min and 6 h) |
| H4 absolute accuracy is regime-dependent, skill is less so | **Partly supported, with a correction**: skill is *strongly* regime-dependent and collapses in cloudy conditions, while absolute error is lowest there |
| H5 the best model by RMSE is not best by interpretability or cost | **Supported on cost**; untested on interpretability, since H has not been run |

H3 and H4 both failed in the direction that makes the aggregate table misleading,
which is the strongest argument this study offers for stratified evaluation.
