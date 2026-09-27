# AI/ML-Based Solar Photovoltaic Power Forecasting: When Does Model Complexity Pay?

A controlled, leakage-audited benchmark of solar PV power forecasting, designed
to answer one question:

> **When does additional model complexity provide meaningful forecasting value
> over strong physical and statistical baselines?**

The answer this repository supports is not "the deepest model wins". Across four
forecast horizons, four weather regimes, four seasons, five input regimes,
multiple seeds and a seven-station site panel, model capacity is the *least*
important of the three levers examined — and the model that is hardest to beat is
a persistence forecast with a clear-sky correction.

---

## Project overview

| | |
| --- | --- |
| Target | Rooftop PV AC power, 55 kW station, 15-minute resolution |
| Horizons | 15 min, 1 h, 6 h, 24 h (all exact multiples of the resolution) |
| Models | 11 in 4 families, from rule-based references to a Transformer |
| Records | 118 training runs, every one machine-readable |
| Result table | `results/tables/final_experiment_matrix.csv` — 436 rows |
| Tests | 146, all passing, no network or dataset required |
| Reproduce | `pip install -r requirements.txt` then five commands |

## Research questions

| ID | Question | Evidence |
| --- | --- | --- |
| RQ1 | How do the families compare under one protocol, and how much of a neural model's score is initialisation? | Tables 2, 9 |
| RQ2 | Does relative performance hold across four horizons? | Table 3 |
| RQ3 | How do accuracy and skill change across weather regimes and seasons? | Tables 4, 5 |
| RQ4 | Do advanced architectures and richer inputs beat strong classical baselines? | Tables 6, 7 |
| RQ5 | Which inputs and which variables carry the signal? | Tables 7, 11 |
| RQ6 | Does a model transfer to sites it has never seen? | Table 8 |
| RQ7 | What is the accuracy-per-cost profile, and can an accurate model be uncertainty-quantified? | Tables 12, 13 |

Every row of the result matrix carries the `research_question` it answers, so
this mapping is data, not prose. `tools/check_consistency.py` verifies it.

## Research gap

Derived from 30 DOI-verified papers, each gap evidenced in
`docs/research_gaps.md` and the workbook's `Research Gaps` sheet. The gaps this
study addresses: aggregate-only evaluation; horizon sensitivity asserted rather
than measured; classical baselines left untuned; persistence under-used as a
*strong* reference; cross-site transfer claimed but rarely validated; importance
reported from a single method or from attention weights; and cost never reported
alongside accuracy.

## Contributions

1. A controlled comparison of 11 model variants across 4 horizons on one split,
   with two persistence references scored alongside every learned model.
2. Stratified error analysis by regime, season, time of day, generation level and
   ramping, with the reference rescored inside each stratum.
3. An additive input-regime study, separating what each information type is worth
   alone from what it adds to everything else.
4. A cross-site study over a fixed 7-station panel with three protocols.
5. Multi-seed replication of the neural models.
6. A measured account of serial dependence in the errors and what it does to
   inference, reported without adjusting any result towards significance.

## Dataset

> Lin, Z., Zhou, Q., Wang, Z., Wang, C., Bookhart, D. B., & Leung-Shea, M. (2024).
> *A high-resolution three-year dataset supporting rooftop photovoltaics (PV)
> generation analytics.* Dryad. <https://doi.org/10.5061/dryad.m37pvmd99>

Zenodo record 10909062, **CC0 1.0** — no registration, no key, redistributable.
Case-study station **LSK North**, 55.0 kW, chosen for the longest gap-free
15-minute record and because its measured peak (55 294 W) matches nameplate to
within 0.5%. Cross-site panel: SQ1–SQ4, UG Hall6, UG Hall7 (25.0–33.3 kW).

Every dataset claim is machine-verified by `tools/audit_dataset_claims.py`
(18 checks: licence, checksum, station count, record length, resolution,
coverage dates, zero fraction, weather variables, split sizes, input finiteness).

## Methodology

Full detail in `docs/methodology.md`. The information set is strictly
historical: at the forecast origin a model sees only observations at or before
that origin, so no NWP input is used — a deliberate restriction that bounds what
the results mean.

| Component | Choice |
| --- | --- |
| Split | chronological: train 2021-06-02→2022-08-31, val 2022-09→2022-12, test 2023 (a full calendar year) |
| Scalers | fitted on the training split only; asserted per split |
| Lookback | 24 steps (6 h), identical at every horizon so only the gap varies |
| Inputs | target lags and trailing statistics, weather with trailing means and differences, calibrated clearness index, solar geometry, cyclical calendar |
| Training | MAE on the standardised target, early stopping on validation MAE in watts, best weights restored, seed recorded per run |
| Tuning | small validation-only grid per classical family |

**The forecasting formulation is validated, not assumed.** For every horizon,
tests assert the exact calendar timestamps: origin `2023-06-15 10:00` → targets
`10:15`, `11:00`, `16:00`, and `2023-06-16 10:00`; the label is checked against
the raw power record at that timestamp; every input step is asserted to be at or
before the origin; and a double-shift is guarded. The meteorological
end-of-interval convention is pinned by value, not by comment.

## Models

| Family | Models | Notes |
| --- | --- | --- |
| Rule-based reference | persistence, smart persistence | smart persistence is the guarded clear-sky ratio: reference floor, ratio clip, capacity clip |
| Classical ML | linear regression, random forest, gradient boosting, XGBoost | flattened window, validation-tuned per family |
| Recurrent | LSTM, GRU | ordered window |
| Hybrid / attention | CNN-LSTM, attention-LSTM, Transformer | sized for CPU reproducibility, not for a leaderboard |

## Experiments and status

| Group | Question | Status |
| --- | --- | --- |
| A | Baseline comparison | executed (3 runs) |
| B | ML vs DL, headline | executed (11 runs) |
| C | Four horizons | **executed, complete: 44 runs, 11 models × 4 horizons** |
| D | Weather regimes | executed (analysis of B) |
| E | Leave-one-out feature ablation | executed (12 runs) |
| F | Model ablation, composite vs backbone | executed (analysis of B) |
| G | Seasonal generalisation | executed (analysis of B) |
| H | Explainability | **executed**: SHAP for the tree ensemble, grouped permutation importance for an attention-recurrent model, plus regime-specific importance and dependence plots |
| I | Computational cost | executed (assembly from the registry) |
| J | Cross-site generalisation | **executed**: 60 records over 7 sites, 3 protocols |
| K | Conformal uncertainty | executed |
| N | Additive input regimes | **executed: 25 runs, 5 models × 5 regimes** |
| M | Multi-seed replication | **executed: 23 runs, 3-5 seeds per neural model** |

The full status with per-group run counts is regenerated by
`scripts/generate_report.py` and verified by `tools/check_consistency.py`.

## Results

All numbers below are read from `results/tables/`. No figure is typed by hand.

### 1. At one hour, several models do not beat a rule (Table 2)

| Model | MAE (W) | RMSE (W) | nRMSE (cap.) | R² | Skill vs persistence | Train (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Gradient boosting | 4 374 | 6 664 | 0.121 | 0.785 | +0.271 | 9.5 |
| Random forest | 4 473 | 6 693 | 0.122 | 0.783 | +0.268 | 287.4 |
| CNN-LSTM | 4 165 | 6 819 | 0.124 | 0.775 | +0.254 | 274.1 |
| XGBoost | 4 505 | 6 835 | 0.124 | 0.774 | +0.252 | 42.5 |
| Linear regression | 5 174 | 7 457 | 0.136 | 0.731 | +0.184 | 2.0 |
| Smart persistence | 4 570 | 7 668 | 0.139 | 0.716 | +0.161 | 0.0 |
| Transformer | 5 549 | 7 851 | 0.143 | 0.702 | +0.141 | 1 122.0 |
| Attention-LSTM | 6 237 | 9 090 | 0.165 | 0.600 | +0.006 | 172.8 |
| Persistence | 6 616 | 9 142 | 0.166 | 0.596 | 0.000 | 0.0 |
| LSTM | 6 746 | 9 264 | 0.168 | 0.585 | **−0.013** | 312.6 |
| GRU | 6 893 | 9 611 | 0.175 | 0.553 | **−0.051** | 361.9 |

LSTM and GRU are *worse than holding the last observation forward*. Attention-LSTM
is indistinguishable from it. The tree family is strongest, and gradient boosting
is the most accurate model at 9.5 s of training. It is not the cheapest trained
model - the linear fit costs 2.0 s - but it is the cheapest model within 11% of the
best RMSE, and 118x cheaper than the Transformer that is 18% worse.

### 2. The reference's competence depends entirely on horizon (Table 3)

Skill against persistence, by horizon:

| Model | 15 min | 1 h | 6 h | 24 h |
| --- | ---: | ---: | ---: | ---: |
| Gradient boosting | +0.095 | +0.271 | **+0.592** | +0.146 |
| Random forest | +0.069 | +0.268 | **+0.585** | +0.136 |
| CNN-LSTM | +0.088 | +0.254 | **+0.565** | +0.103 |
| XGBoost | +0.039 | +0.252 | **+0.577** | +0.121 |
| Linear regression | +0.056 | +0.184 | **+0.571** | +0.139 |
| Smart persistence | +0.028 | +0.161 | +0.194 | +0.000 |
| Transformer | **−0.030** | +0.141 | +0.551 | +0.119 |
| Attention-LSTM | **−0.528** | +0.006 | +0.553 | +0.021 |
| LSTM | **−0.649** | −0.013 | +0.537 | +0.029 |
| GRU | **−0.668** | −0.051 | +0.531 | +0.029 |

At 15 minutes the reference is the strongest model in the table for four of them.
At 6 hours, persistence has collapsed (nRMSE 0.407 against 0.166 at one hour) and
every learned model reaches skill 0.53–0.59. **A benchmark that reports one
horizon cannot know whether its reference is strong.** This is the single most
convincing argument in the study for the multi-horizon protocol.

### 3. PV history alone is nearly sufficient; weather rescues the neural models (Table 7)

| Regime | Inputs | Gradient boosting | XGBoost | CNN-LSTM | LSTM |
| --- | ---: | ---: | ---: | ---: | ---: |
| PV only | 15 | **0.119** | 0.120 | 0.122 | 0.176 |
| Weather only | 12 | 0.182 | 0.177 | 0.175 | 0.184 |
| PV + weather | 27 | 0.123 | 0.124 | 0.124 | 0.165 |
| PV + weather + geometry | 29 | 0.122 | 0.125 | 0.124 | 0.167 |
| Full | 33 | 0.121 | 0.124 | 0.124 | 0.168 |

(nRMSE, daylight steps, 1 h horizon.)

Two findings that the aggregate table cannot show. **For the tree models, the
weather variables add nothing**: 15 PV-history features match or beat all 33, and
adding 12 weather features makes the error slightly *worse*. **For the recurrent
models, the weather variables are decisive**: LSTM moves from 0.176 with PV
history alone to 0.165 with weather added, and its skill goes from −0.058 to
+0.008. Weather-only is worse than persistence for every family. The value of a
feature group therefore depends on the access pattern of the model reading it,
which is why a single global feature-importance ranking is not sufficient.

### 4. Cross-site transfer fails, and that is the most important result (Table 8)

Skill against persistence, training on LSK North and testing on an unseen site:

| Protocol | xgboost | gradient boosting | linear regression |
| --- | --- | --- | --- |
| Within-site (reference) | +0.24 … +0.27 | +0.24 … +0.28 | +0.09 … +0.24 |
| Cross-site (1 training site) | **−0.21 … −0.88** | **−0.10 … −0.57** | **−0.34 … −1.07** |
| Leave-one-site-out (6 training sites) | −0.21 … +0.26 | +0.00 … +0.26 | +0.21 … +0.24 |

A model fitted at one installation is **worse than persistence at every other
installation**. Pooling six training sites repairs transfer almost completely for
five of the six holdouts, but not for LSK North itself (+0.002 for the tree
models), which is the one site that differs most in capacity. The practical
consequence: published single-site PV results are *local* skill, not a property
of a model family, and a new installation must be calibrated on its own data.

### 5. Seed spread does not change the conclusions (Table 9)

| Model | Seeds | RMSE mean ± sd (W) | Skill mean ± sd |
| --- | ---: | --- | --- |
| CNN-LSTM | 5 | 6 722 ± 71 | +0.265 ± 0.008 |
| Transformer | 3 | 7 887 ± 179 | +0.137 ± 0.020 |
| Attention-LSTM | 5 | 8 792 ± 224 | +0.038 ± 0.024 |
| LSTM | 5 | 9 210 ± 194 | -0.007 ± 0.021 |
| GRU | 5 | 9 574 ± 44 | -0.047 ± 0.005 |

The across-seed spread is small relative to the between-model differences — GRU
varies by 44 W across five initialisations, against a 2 900 W gap between the
best and worst model — so the ranking in Table 2 is not an artefact of one
initialisation. The spread is also informative about which architectures are
unstable: GRU is the most reproducible (sd 44 W) and attention-LSTM the least
(sd 224 W), which is consistent with attention being the component that depends
most on where optimisation starts.

Note the honest caveat: **three seeds is the documented minimum**, and only the
Transformer was held to that floor, because a single run costs about 19 minutes on
this hardware; the other four were replicated five times. The best-of-seeds number
is never quoted, because reporting the best run of five is a report of selection.

### 6. Statistics: significance and effect size answer different questions (Table 10)

| Model | Skill | 95% block-bootstrap CI | Diebold-Mariano p (MAE) | Holm |
| --- | ---: | --- | ---: | ---: |
| Gradient boosting | +0.271 | [0.249, 0.293] | <1e-15 | <1e-15 |
| CNN-LSTM | +0.254 | [0.231, 0.278] | <1e-15 | <1e-15 |
| Attention-LSTM | +0.006 | [-0.041, +0.051] | 2.1e-11 | 6.3e-11 |
| LSTM | -0.013 | [-0.058, +0.025] | 2.2e-02 | 2.2e-02 |
| GRU | -0.051 | [-0.098, -0.010] | 3.2e-06 | 6.5e-06 |

The test rejects 18 of 20 comparisons after Holm correction, and the bootstrap
interval excludes zero for eight of the ten models compared against the reference
— the two agree once the Harvey-Leybourne-Newbold correction is applied. The
correction is what makes the agreement possible: errors at 15-minute resolution
share a cloud field, so the integrated autocorrelation time of the loss
differential is 9.2 steps, which cuts the effective sample from 17 679 daylight
steps to about 1 917, and the long-run variance is far above an i.i.d. estimate.
`error_autocorrelation.csv` and `block_length_sensitivity.csv` carry the
diagnostics.

Both statistics are reported because they answer different questions. The p-value
says whether a difference can be told from the sampling noise; the interval says
how large the difference is. Here the differences are statistically clear and
practically small — the tree models beat persistence by 2.1–2.5 kW in mean
absolute error on a 55 kW array, and the best of them, gradient boosting, is
1.1 kW ahead of the Transformer, while the LSTM and GRU are *reliably worse* than
persistence by 130 W and 277 W. A reader given only the p-value column would
conclude the second group was indistinguishable, which is the opposite of what the
evidence supports. No method, block length or sample was changed after the fact to
obtain a preferred answer.

### 7. Error, skill and interpretability (Tables 4, 11, 12)

In **cloudy** conditions absolute error is *lowest* (persistence nRMSE 0.104 vs
0.185 in clear sky) because the signal amplitude is small, while skill is
*lowest* (gradient boosting +0.024 vs +0.358). Reading the error column alone
inverts the conclusion about where forecasting is hard.

SHAP for XGBoost ranks `pv_power_w_lag_0` first, then `hour_sin`, then `ghi` — the
current power reading, the time of day, and irradiance. Grouped permutation
importance for the attention-recurrent model agrees on PV history dominating, and
the **regime-specific** analysis shows the ranking genuinely changes: under clear
sky the trailing power *variability* dominates, while under broken cloud the
current power reading does.

Conformal intervals cost about ±4.8 kW for XGBoost and ±8.1 kW for LSTM on a 55 kW
plant at 90% nominal, and empirical coverage is below nominal because the
calibration block precedes the evaluation block — an exchangeability violation
that is documented rather than glossed over.

## Reproducibility

See `docs/reproducibility.md`. Pinned environment, per-run JSON records with
split dates, feature list, hyperparameters, durations, git revision and a config
fingerprint, and a registry rebuildable from those records.

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py        # fetch + verify (CC0, no credentials)
python scripts/preprocess.py           # pipeline, split, features
python scripts/run_experiments.py      # the declared experiment matrix
python scripts/evaluate.py --report    # tables, statistics, explainability, figures
python scripts/generate_manuscript.py  # paper/manuscript.md from the tables
python -m pytest -q                    # 191 tests
python tools/check_consistency.py      # prose versus results
```

Other entry points: `scripts/train.py` (one model), `scripts/run_cross_site.py`
(transfer protocols), `tools/audit_dataset_claims.py` (dataset claims),
`tools/launch_experiments.py` (parallel workers), `tools/run_remaining.py`
(restartable queue).

## Project structure

```
configs/          data.yaml, experiments.yaml, models.yaml  (all behaviour lives here)
data/             raw / interim / processed + download manifest
src/solar_forecasting/
  data/           acquisition, loading, schema validation
  preprocessing/  cleaning, missing values, outliers, daylight, splitting
  features/       feature engineering, ablation groups, input regimes
  models/         registry, baselines, classical, neural
  training/       sequences, trainer, experiment, runner
  cross_site/     within-site, cross-site and leave-one-site-out protocols
  evaluation/     metrics, regimes, stratified, uncertainty, reporting, matrix
  explainability/ SHAP, permutation importance, dependence, analysis driver
  statistics/     block bootstrap, Diebold-Mariano, Wilcoxon, dependence diagnostics
  visualization/  26 figure functions
tests/            191 tests, no network, no dataset required
literature/       literature_review.xlsx, references.bib, verification records
docs/             methodology, dataset, experiments, reproducibility, research_gaps
results/          experiments, tables, figures, metrics, predictions
paper/            manuscript.md plus section sources and copied tables
tools/            audit, verification, launchers, consistency checker
```

## Literature

`literature/literature_review.xlsx` — 30 papers, each verified against Crossref by
`tools/literature/verify_links.py`, which resolves the DOI and compares the
registered title. All 30 resolve and all 30 titles match. Five sheets:
`Literature Review` (the 10 required columns), `Literature Analysis` (29
columns per paper), `Research Gaps`, `Provenance`, and
`Selection and Exclusions` with the 5 rejected candidates and reasons.

## Limitations

These are real and cannot be engineered away:

* **One climate.** The primary result is a single 55 kW array in Hong Kong. The
  cross-site study shows transfer fails even within one campus, so the headline
  numbers are local skill.
* **No NWP inputs.** Under a strictly historical information set, a one-hour
  forecast cannot anticipate a cloud front. This is why the reference is strong
  at 15 minutes and why the ceiling is low.
* **Three seeds, not five.** The documented minimum; a five-seed Transformer
  replication did not fit the compute budget.
* **Cross-site uses three cheap models.** The transfer experiment covers gradient
  boosting, XGBoost and linear regression, not the neural models, which were too
  expensive to fit 20 times per model.
* **One fixed protocol, not a per-family tuning budget.** The trees received a
  small validation grid; the neural models received one learning rate. This
  asymmetry is a limitation, not a strength, and it is stated in the threat to
  internal validity.
* **Conformal coverage is approximate**, because the calibration block precedes
  the evaluation block.
* **Cost figures are machine-specific** in absolute terms; only the ranking is
  portable.

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`. The dataset is redistributed by its authors under CC0 1.0
and is fetched by `scripts/download_data.py`; no rights in that data are claimed
here.

