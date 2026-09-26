# Methodology

## 1. Research problem

Given a rooftop photovoltaic plant's measured AC power output, co-located
meteorological observations and a forecast horizon of 15 minutes to 24 hours, how
accurate, robust, interpretable and cheap is a given forecasting model?

The problem is posed as **short-term** forecasting under a **strict historical
information set**: at the forecast origin every model sees only observations at or
before that origin. Numerical weather prediction (NWP) inputs, which are not
available in this dataset, are therefore excluded. This is a deliberate
restriction and it bounds what the results mean — see §13.

## 2. Research questions

| ID | Question | Answered by |
| --- | --- | --- |
| RQ1 | How do classical ML, recurrent DL and attention architectures compare under one controlled protocol? | Experiment B |
| RQ2 | Does relative performance hold across four forecast horizons? | Experiment C |
| RQ3 | How does accuracy and skill change across weather regimes? | Experiment D |
| RQ4 | Do attention-based architectures beat well-tuned gradient boosting? | Experiments B, F |
| RQ5 | Which input groups carry the forecasting signal? | Experiment E, H |
| RQ6 | Does performance hold across seasons and across sites? | Experiment G (seasons), J (sites, not run) |
| RQ7 | Can an accurate model also be interpretable and uncertainty-quantified? | Experiments H, K |

## 3. Hypotheses

Stated before reading the results, so that a null result is a finding rather than
an inconvenience.

* **H1 (baseline strength).** At a one-hour horizon, a persistence reference
  calibrated with a clear-sky ratio is a strong baseline, and at least one learned
  model fails to beat it.
* **H2 (family gap).** Gradient-boosted trees match or beat recurrent and
  attention architectures on a single-site, three-year record, because the signal
  is dominated by the diurnal and irradiance structure that trees capture with far
  less data.
* **H3 (horizon collapse).** The advantage of the best model over persistence
  decreases monotonically with horizon, and any architectural advantage shrinks
  faster than the reference's disadvantage.
* **H4 (regime dependence). *Absolute* accuracy is strongly regime-dependent,
  while *skill against persistence* is far less so, because the reference is
  itself easy in clear conditions and hard under broken cloud.
* **H5 (explainability cost).** The best model by RMSE is not the best model by
  any single measure of interpretability or computational cost, so a deployment
  choice cannot be made from the error metric alone.

## 4. Dataset

See `docs/dataset.md` for the selection rationale, the archive layout and the
quality report. In brief: a three-year, 15-minute, 60-station rooftop PV dataset
with co-located meteorology, published open access (CC0 1.0) via Dryad
(doi:10.5061/dryad.m37pvmd99) and Zenodo record 10909062. The case-study station
is **LSK North** (55.0 kW rated), chosen for the longest gap-free record and the
closest agreement between rated and measured peak power.

## 5. Preprocessing

Every stage below is implemented in `src/solar_forecasting/preprocessing/` and
recorded in `results/metrics/pipeline_report.json`.

1. **Acquisition.** The archive is downloaded and verified against a manifest
   (`data/raw/download_manifest.json`); no file is trusted without a checksum.
2. **Schema validation.** Column presence, dtype, timestamp parsing, target
   values, observed resolution and nameplate consistency, before any
   transformation, so a malformed input is a schema failure and not a NaN three
   stages later.
3. **Timestamp normalisation.** Timezone is made explicit, the index is sorted,
   and the modal spacing is compared with the declared 15-minute resolution.
4. **Duplicate detection.** Repeated timestamps are counted and removed before
   resampling.
5. **Missing-timestamp analysis.** The expected regular grid is constructed and
   the gap structure reported (count, fraction, largest gap).
6. **Missing-value handling.** Split-aware: the target is time-interpolated
   inside the training split with `limit_direction="both"`, and forward-only at
   and after `train_end` so that no value from the future leaks backwards across
   the boundary. Weather variables are time-interpolated then forward-filled with
   a bounded limit, never extrapolated.
7. **Irradiance cleaning.** A hard physical envelope (0–1200 W/m²), rolling
   median spike detection (15-step window, ratio 1.8), a second envelope pass
   after substitution, and zeroing of any positive reading below the horizon,
   which is a pyranometer zero-offset rather than dim light.
8. **Daylight handling.** Solar elevation is computed with `pvlib` at the site's
   coordinates; `daylight` is elevation > 0°. Every metric is reported for
   daylight steps and for all steps separately.
9. **Meteorological aggregation.** 1-minute observations are averaged into
   end-of-interval 15-minute bins, so a bin labelled 12:00 covers 11:45–12:00 and
   contains nothing later than its own label.

## 6. Feature engineering

Implemented in `src/solar_forecasting/features/builder.py`, declared in
`configs/data.yaml` under `features`.

* **Target history**: lags at 1, 2, 4, 8, 12, 24, 48 and 96 steps, plus trailing
  mean and standard deviation over 4, 12 and 24 steps. All are strictly backward
  looking, with `min_periods` equal to the window so partial windows never enter
  training.
* **Weather**: GHI, temperature, relative humidity, wind speed; trailing means
  over 4 and 24 steps; first and fourth differences of GHI; the calibrated
  clearness index.
* **Solar geometry**: elevation, sine of elevation, azimuth.
* **Calendar**: cyclical hour and day-of-year encodings (never raw integers, which
  would make midnight adjacent to noon).
* **Lookback window**: 24 steps (6 hours), identical for every horizon so that
  the horizon comparison varies only the horizon.

## 7. Model architectures

Eleven models in four families, all consuming the same inputs.

| Family | Models | Input handling |
| --- | --- | --- |
| Rule-based reference | persistence, smart persistence | last observation; last observation scaled by the clear-sky ratio, with a reference floor, a ratio clip and a capacity clip |
| Classical ML | linear regression, random forest, gradient boosting, XGBoost | flattened window (24 × n_features) |
| Recurrent | LSTM, GRU | ordered window |
| Hybrid / attention | CNN-LSTM, attention-LSTM, Transformer | ordered window, with a convolutional or attention encoder |

Architectures are sized for CPU reproducibility rather than to win a leaderboard;
the point is a controlled comparison, and the sizes are declared in
`configs/models.yaml`.

## 8. Training procedure

Identical for every learned model, so that the comparison is not a measure of
tuning budget:

* loss: MAE on the standardised target, early stopping on validation MAE **in
  watts** with patience, and restoration of the best weights rather than the last;
* optimiser: AdamW, fixed learning rate and weight decay family;
* classical models: a small validation-only grid search per family
  (`configs/models.yaml: tuning`), refit on the training split;
* seed 42, recorded with every run; thread counts pinned to 1 so that recorded
  durations are comparable;
* the test split is scored once, after training, and never informs any choice.

## 9. Leakage controls

Non-negotiable, and each one is either asserted in code or covered by a test:

| Control | Where |
| --- | --- |
| Chronological split with explicit dates, never a random split | `preprocessing/splitting.py`, `test_features_and_splitting.py` |
| Scalers fitted on the training split only | `fit_scalers`, `test_scalers_are_fitted_on_training_only` |
| Windows strictly causal; target read at the origin, not the label row | `training/sequences.verify_causality`, called before the first optimiser step |
| Lead shift applied exactly once | `pipeline.build_target`, verified against the raw power record |
| End-of-interval meteorological bins | `pipeline.aggregate_meteorology` |
| No hyperparameter chosen on test | `configs/models.yaml: tuning` is validation-only |
| Regime labels assigned after prediction, never fed to a model | `evaluation/regimes.py` |
| Each split builds its own windows, so no window reaches across a boundary | `training/experiment.py` |

`verify_causality` is not decoration: it re-derives each sampled window from the
source frame and fails the run on the first mismatch.

## 10. Evaluation

Metrics: MAE, RMSE, nRMSE (by rated capacity and by mean observed power), R²,
sMAPE, bias, peak error, and skill against both persistence references. Each
metric's formula, interpretation and limitations are in
`results/tables/metric_documentation.csv`.

**MAPE is deliberately not computed.** PV power is exactly zero at night, so the
percentage error is undefined on roughly half the record and unbounded near dawn
and dusk. `metrics.mape` raises rather than silently substituting an epsilon,
because a regularised MAPE produces a number that looks comparable with the
literature while being driven by near-zero night-time steps.

**Daylight-only is the headline scope**, with the all-step figures reported
beside it, because aggregating them lets 50% exact zeros dominate the statistics.

## 11. Statistical analysis

* **Moving-block bootstrap** (96-step blocks = one diurnal cycle, 2000
  resamples) for skill intervals. Block length matters: PV forecast errors are
  serially correlated, and a test that ignores that overstates the evidence. The
  suite asserts that the block interval is wider than the i.i.d. one.
* **Diebold-Mariano** with the Harvey-Leybourne-Newbold small-sample correction,
  on both MAE and RMSE differentials. Disagreement between the two losses is
  reported as disagreement, not resolved.
* **Wilcoxon signed-rank** as a secondary check, with its assumption violation
  stated in the output.
* **Holm-Bonferroni** adjusted p-values, because eleven models are compared.
* Model and reference errors are resampled with the *same* block indices, since
  they forecast the same timestamps.

## 12. Explainability and uncertainty

* **SHAP** (`TreeExplainer`, mean |value|) for the tree ensemble, collapsed over
  the lookback so the reader sees which *variable* matters rather than which lag.
* **Grouped permutation importance** for the neural models: a variable is
  scrambled across the whole window and the increase in daylight MAE is recorded.
* **Attention weights are not used as explanations.** They are internal to the
  architecture and are not validated against any perturbation; treating them as
  explanations is a claim, not a measurement.
* Both are *predictive* attributions. No causal reading is offered anywhere.
* **Split conformal intervals** give distribution-free coverage. The calibration
  block is the first 31 days of the test period, and the cost of that choice is
  stated in the output: the coverage guarantee is *approximate*, because the
  calibration block precedes the evaluation block and the two are not exchangeable
  under seasonal drift.

## 13. Limitations

* One station, one climate. A single 55 kW Hong Kong rooftop array cannot
  support a claim about PV forecasting in general; it supports a claim about the
  relative behaviour of these eleven models on this record.
* No NWP inputs. Under a strict historical information set, a one-hour forecast
  cannot anticipate a cloud front, which bounds the attainable accuracy.
* Test year follows the training period, so the test year is entirely
  out-of-sample in time but not in the sense of an unseen site.
* Cost figures are machine-specific in absolute terms; only the ranking is
  portable.
* Cross-site transfer (Experiment J) is not implemented, so RQ6's site half is
  open. See `docs/experiments.md`.

## 14. Reproducibility

See `docs/reproducibility.md`. In short: pinned environment, fixed seed, explicit
configuration, per-run JSON records with the split dates, feature list,
hyperparameters, durations and package versions, and a registry rebuildable from
those records.
