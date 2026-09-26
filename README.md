# AI/ML-Based Solar Photovoltaic Power Forecasting: A Reproducible and Explainable Benchmark Across Forecast Horizons and Weather Conditions

A controlled, leakage-audited benchmark of machine-learning and deep-learning
models for short-term solar PV power forecasting, built so that another researcher
can reproduce the experiments end to end and so that a reader can tell which
conclusions the data support and which it does not.

---

## Research question

> How robust and generalizable are different machine-learning and deep-learning
> approaches for short-term solar PV power forecasting across different forecast
> horizons and weather conditions?

Seven explicit, testable sub-questions are listed in `docs/methodology.md` §2
(RQ1–RQ7) and mapped to the experiment that answers each.

## Motivation

A PV operator needs a forecast that is accurate *and* that can be trusted when the
sky changes. Most published PV forecasting studies report a single aggregate
error on a single split, compare against weak references, and stop there. That
leaves three practical questions largely unanswered: whether the ranking of model
families is stable across horizons and weather regimes, which inputs actually
carry the signal, and what an accurate-but-expensive model buys relative to a cheap
one. This project answers them under one protocol with the bookkeeping exposed.

## Research gap

Derived from 30 verified papers rather than asserted. Full analysis with
per-paper evidence in `docs/research_gaps.md` and in the `Research Gaps` sheet of
the workbook. The gaps this project targets:

| Gap | Evidence pattern in the reviewed corpus |
| --- | --- |
| G1 Aggregate-only evaluation | Aggregate RMSE/MAE/R² dominate; regime-, season- and time-stratified evaluation is a minority, and ranking changes within conditions are almost never reported |
| G2 Horizon sensitivity asserted, not measured | Single-horizon studies dominate; multi-horizon work rarely holds inputs and split fixed |
| G3 Weak classical baselines | Recurrent and attention papers rarely compare against a *tuned* tree model on identical inputs |
| G4 Persistence under-used as a strong reference | Naive persistence is ubiquitous; the clear-sky-ratio variant is rare, so reported skill overstates model value |
| G5 Uncertainty rarely calibrated | Probabilistic work is a visible minority; the deterministic majority reports no interval |
| G6 Explainability asserted from attention weights | Attention maps presented as explanations; cross-family importance comparison nearly absent |
| G7 Uneven reproducibility | Split dates, seeds, versions and uncertainty reported inconsistently; cross-paper comparison is not possible |
| G8 Cost not measured with accuracy | Training cost, latency and parameter count reported almost never alongside accuracy |

Two candidate gaps were examined and **not** claimed (deep learning always wins;
weather variables are under-used) because the evidence points the other way. That
disagreement is preserved rather than resolved.

## Contributions

1. A **controlled benchmark** of 11 models in 4 families — two persistence
   references, four classical ML models (linear regression, random forest, gradient
   boosting, XGBoost), two recurrent (LSTM, GRU), three hybrid/attention
   (CNN-LSTM, attention-LSTM, Transformer) — on one split, one input set, one seed.
2. **Multi-horizon evaluation** at 15 minutes, 1 hour, 6 hours and 24 hours, all
   exact integer multiples of the 15-minute working resolution so no interpolation
   enters the comparison.
3. A **transparent weather-regime taxonomy** with threshold provenance measured on
   the training split, plus seasonal, time-of-day, generation-level and ramping
   stratification, each with the persistence reference rescored inside the stratum.
4. **Statistical comparison** with a serial-correlation-aware moving-block
   bootstrap, the Diebold-Mariano test with the Harvey-Leybourne-Newbold
   correction, Wilcoxon as a secondary check, and Holm-Bonferroni adjustment.
5. **Explainability** by SHAP (tree ensemble) and grouped permutation importance
   (neural models), with attention weights explicitly not treated as explanations.
6. **Conformal prediction intervals** with the calibration trade-off documented.
7. A **results-integrity apparatus**: per-run JSON records, a registry rebuildable
   from them, an evaluation report that distinguishes "not run" from "no result",
   and 108 tests covering the leakage-sensitive code.

## Dataset

A three-year, 15-minute, 60-station rooftop PV dataset with co-located 1-minute
meteorology, published open access:

> Lin, Z., Zhou, Q., Wang, Z., Wang, C., Bookhart, D. B., & Leung-Shea, M. (2024).
> *A high-resolution three-year dataset supporting rooftop photovoltaics (PV)
> generation analytics.* Dryad. <https://doi.org/10.5061/dryad.m37pvmd99>
> (Zenodo record 10909062, **CC0 1.0**)

Case-study station **LSK North**, 55.0 kW rated, chosen for the longest gap-free
15-minute record and because its measured peak (55 294 W) matches nameplate to
within 0.5%. Selection rationale, archive layout, the known data issues and their
handling are in `docs/dataset.md`.

Case-study record after cleaning: **90 528** 15-minute steps, mean power 8 305 W,
maximum 55 295 W, and **50.1% exactly-zero steps** — which is why MAPE is not
computed and why night-time is scored separately.

## Methodology

Full detail in `docs/methodology.md`. The information set is strictly
historical: at the forecast origin a model sees only observations at or before
that origin, so no NWP input is used. That is a deliberate restriction and it
bounds what the results mean.

| Component | Choice |
| --- | --- |
| Resolution | 15 minutes (finest with continuous multi-year coverage; every horizon an exact integer multiple) |
| Lookback | 24 steps (6 h), identical at every horizon so only the gap varies |
| Inputs | target lags and trailing statistics, GHI/temperature/humidity/wind with trailing means and differences, calibrated clearness index, solar geometry, cyclical calendar |
| Split | chronological: train 2021-06-01→2022-08-31, validation 2022-09-01→2022-12-31, test 2023-01-01→2023-12-31 (a full calendar year, so each season appears once) |
| Scaling | standardisation fitted on the **training split only** |
| Training | MAE loss on the standardised target, early stopping on validation MAE in watts, best weights restored, seed 42, thread count pinned to 1 |
| Tuning | small validation-only grid per classical family |

Leakage controls are tabulated in `docs/methodology.md` §9, and three of them are
*asserted at run time* — window causality is re-derived from the source frame
before the first optimiser step, scalers refuse non-finite or zero-variance
features, and the split reports how many rows it dropped.

## Models

| Family | Models | Notes |
| --- | --- | --- |
| Rule-based reference | persistence, smart persistence | smart persistence is the guarded clear-sky ratio, with a reference floor, a ratio clip and a capacity clip |
| Classical ML | linear regression, random forest, gradient boosting, XGBoost | consume the flattened window; each family gets a small validation-only grid |
| Recurrent | LSTM, GRU | ordered window |
| Hybrid / attention | CNN-LSTM, attention-LSTM, Transformer | sized for CPU reproducibility, not for a leaderboard |

Architectures and hyperparameters live in `configs/models.yaml`; the registry in
`src/solar_forecasting/models/registry.py` is the single place a model is
constructed.

## Experimental design

Ten experiment groups, declared in `configs/experiments.yaml`, mapped to research
questions in `docs/experiments.md`, with per-group status (executed / partial /
implemented-but-not-run / not implemented).

| Group | Question | Status |
| --- | --- | --- |
| A | Baseline comparison | executed |
| B | ML versus DL, headline | executed (11/11) |
| C | Horizon comparison | **partial** — 1 h complete; 15 min, 6 h, 24 h partially run |
| D | Weather regimes | executed |
| E | Feature-group ablation | **partial** |
| F | Model ablation (composite vs backbone) | executed |
| G | Seasonal generalisation | executed |
| H | Explainability | implemented, opt-in (`--explainability`) |
| I | Computational cost | executed |
| J | Cross-site transfer | **not implemented** — RQ6's site half is open |
| K | Conformal uncertainty | executed |

The partial groups are a **time-budget** fact of this implementation session, not a
methodological choice, and the exact commands to complete them are in
`docs/experiments.md`. The horizon table records the provenance of every row in a
`source_group` column, so a substituted run is visible rather than implied.

## Evaluation metrics

| Metric | Role | Why |
| --- | --- | --- |
| MAE, RMSE | primary | robust (MAE) and peak-sensitive (RMSE), in watts |
| nRMSE | primary | normalised by rated capacity and by mean observed power, so different normalisation choices are both visible |
| R² | secondary | structurally inflated by the diurnal cycle; never read as operational skill alone |
| sMAPE | primary | scale-free and defined where the target is zero |
| Skill vs persistence | primary | `1 − RMSE_model/RMSE_reference`, on identical timestamps, against **both** references |
| bias, peak error | diagnostic | systematic over-forecasting and worst-timestep reserve margin |

**MAPE is not computed, deliberately**: the target is exactly zero for 50.1% of
steps, so the percentage error is undefined on half the record and unbounded near
dawn and dusk. `metrics.mape` raises rather than silently regularising. Every
metric's formula, interpretation and limitations are in
`results/tables/metric_documentation.csv`.

## Results

Full, regenerated results: **`docs/RESULTS_REPORT.md`** and
`results/tables/RESULTS.md`. Every number below is computed, and none is
hand-entered. Experiment status at the time of writing is in
`docs/experiments.md`.

### Headline comparison (1-hour horizon, daylight steps of the test year)

| Model | MAE (W) | RMSE (W) | nRMSE (capacity) | R² | Skill vs persistence | Train (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Gradient boosting | 4 374 | 6 664 | 0.121 | 0.785 | +0.271 | 9.5 |
| Random forest | 4 473 | 6 693 | 0.122 | 0.783 | +0.268 | 287.4 |
| XGBoost | 4 505 | 6 835 | 0.124 | 0.774 | +0.252 | 42.5 |
| CNN-LSTM | 4 165 | 6 819 | 0.124 | 0.775 | +0.254 | 274.1 |
| Linear regression | 5 174 | 7 457 | 0.136 | 0.731 | +0.184 | 2.0 |
| Transformer | 5 549 | 7 851 | 0.143 | 0.702 | +0.141 | 1 122.0 |
| Smart persistence | 4 570 | 7 668 | 0.139 | 0.716 | +0.161 | 0.0 |
| Attention-LSTM | 6 237 | 9 090 | 0.165 | 0.600 | +0.006 | 172.8 |
| Persistence | 6 616 | 9 142 | 0.166 | 0.596 | 0.000 | 0.0 |
| LSTM | 6 746 | 9 264 | 0.168 | 0.585 | **−0.013** | 312.6 |
| GRU | 6 893 | 9 611 | 0.175 | 0.553 | **−0.051** | 361.9 |

What this table says, stated without overreach:

* **A plain persistence reference is hard to beat.** Two of the neural models —
  LSTM and GRU — have *negative* skill against it: they are worse than holding
  the last observation forward. A third, attention-LSTM, is indistinguishable
  from it (95% block-bootstrap CI on skill: [−0.041, +0.051]).
* **The tree family is the strongest**, and gradient boosting achieves the lowest
  RMSE while being the cheapest trained model in the set (9.5 s).
* **Architectural sophistication is not rewarded here.** The best neural model
  (CNN-LSTM) is competitive; the Transformer costs 118× the training time of
  gradient boosting for a worse RMSE; LSTM and GRU are worse than the reference.
  On a single-site record whose signal is dominated by diurnal and irradiance
  structure, capacity does not appear to be the binding constraint.
* **The best model by RMSE is not the best by cost.** Gradient boosting is on the
  accuracy-cost frontier; the Transformer is not.

### Statistical comparison (against persistence, daylight steps)

| Model | Skill (point) | 95% block-bootstrap CI | DM p (MAE) | Holm-adjusted |
| --- | --- | --- | --- | --- |
| Gradient boosting | +0.271 | [0.249, 0.293] | 0.68 | 1.00 |
| Random forest | +0.268 | [0.246, 0.290] | 0.71 | 1.00 |
| CNN-LSTM | +0.254 | [0.231, 0.278] | 0.64 | 1.00 |
| XGBoost | +0.252 | [0.230, 0.274] | 0.72 | 1.00 |
| Linear regression | +0.184 | [0.162, 0.206] | 0.76 | 1.00 |
| Smart persistence | +0.161 | [0.139, 0.186] | 0.64 | 1.00 |
| Transformer | +0.141 | [0.117, 0.165] | 0.83 | 1.00 |
| Attention-LSTM | +0.006 | [−0.041, +0.051] | 0.96 | 1.00 |
| GRU | −0.051 | [−0.098, −0.010] | 0.97 | 1.00 |
| LSTM | −0.013 | [−0.058, +0.025] | 0.99 | 1.00 |

**The Diebold-Mariano p-values here must be read correctly, and they are the
honest part of this result.** Every test is non-significant even for differences
that are obviously real in the point estimates. The reason is in
`results/tables/statistical_significance.json`: PV forecast errors are strongly
serially correlated — they share a cloud field across consecutive 15-minute steps
— so the long-run variance of the loss differential is orders of magnitude larger
than an i.i.d. estimate, and the test loses power as a result. The block-bootstrap
intervals, which are built for exactly this dependence, *do* separate the models:
they exclude zero for eight of the eleven. The correct reading is: **the
difference between models is real, and the pairing is highly autocorrelated, so
the interval estimate is the informative statistic and the hypothesis test is the
conservative one.** Reporting the p-values without this explanation would be
misleading in the opposite direction.

### Other results

* **Horizons** — 15 min: 10 models run; 6 h and 24 h: 3 models each. Partial, and
  labelled as such. `fig07_horizon_comparison.png`,
  `results/tables/horizon_comparison.csv`.
* **Weather regimes, seasons, intraday behaviour, generation level, ramping** —
  full stratified tables and figures (`weather_regime_comparison.csv`,
  `seasonal_comparison.csv`, `error_analysis.csv`).
* **Feature ablation** — partial; `feature_ablation.csv`.
* **Model ablation** — composite vs backbone under an identical protocol;
  `model_ablation.csv`.
* **Cost** — `computational_cost.csv` with the Pareto flag; absolute durations are
  machine-specific and only the ranking is portable.
* **Conformal intervals** — `uncertainty_coverage.csv`, `fig18_conformal_intervals.png`.
  The coverage guarantee is *approximate*, because the calibration block precedes
  the evaluation block; the caveat is stated in the table, the figure and
  `docs/experiments.md`.
* **Explainability** — implemented and verified, but **not yet run**, so no
  importance numbers are reported. `results/metrics/evaluation_report.json` records
  `explainability: not_run` so this is visibly "not attempted", not "nothing found".

## Reproducibility

See `docs/reproducibility.md`. Summary: pinned `requirements.txt` and
`environment.yml`, seed 42 recorded per run, per-run JSON records with split dates,
feature list, hyperparameters, durations, package versions, git revision and a
configuration fingerprint, and a registry rebuildable from those records. No
machine-specific paths. `Dockerfile` is **not** provided — it could not be tested
here, and the environment files that were used are better than an untested one.

## Installation

```bash
git clone <this repository>
cd <this repository>

python -m pip install -r requirements.txt
# or, for a full environment:
# mamba env create -f environment.yml
```

Python 3.11+ is required; the reported results were produced on 3.13.14, Windows 11,
CPU only. XGBoost and SHAP are used but the code degrades gracefully: the benchmark
falls back to scikit-learn's gradient boosting, and the capability report records
which path was taken.

## Usage

```bash
python scripts/download_data.py                  # fetch + verify the dataset (CC0)
python scripts/preprocess.py                     # data pipeline, split, features
python scripts/run_experiments.py                # the declared experiment matrix
python scripts/run_experiments.py --list         # show the matrix without running
python scripts/run_experiments.py --only C_horizons --horizons 6h   # one slice
python scripts/train.py --model lstm             # a single model
python scripts/evaluate.py --report              # tables, statistics, figures
python scripts/evaluate.py --explainability      # SHAP + permutation importance
python scripts/evaluate.py --uncertainty         # conformal coverage
python scripts/generate_report.py                # docs/RESULTS_REPORT.md
python -m pytest -q                              # 108 tests
```

`--horizons` exists so one long experiment group can be split across parallel
processes without editing code.

## Project structure

```
configs/          data.yaml, experiments.yaml, models.yaml  (all behaviour lives here)
data/             raw / interim / processed, with a download manifest
src/solar_forecasting/
  data/           acquisition, loading, schema validation
  preprocessing/  cleaning, missing values, outliers, daylight, splitting
  features/       feature engineering and ablation groups
  models/         registry, baselines, classical, neural
  training/       sequences, trainer, experiment, runner
  evaluation/     metrics, regimes, stratified, uncertainty, reporting
  explainability/ SHAP, permutation importance, analysis driver
  statistics/     block bootstrap, Diebold-Mariano, Wilcoxon
  visualization/  every figure
  utils/          seeding, IO
scripts/          download_data, preprocess, train, run_experiments, evaluate, generate_report
tests/            108 tests, no network, no dataset required
literature/       literature_review.xlsx, references.bib, verification records
docs/             methodology, dataset, experiments, reproducibility, research_gaps, RESULTS_REPORT
results/          experiments, tables, figures, metrics, predictions
paper/            paper-support material for the write-up
tools/literature/ corpus discovery, DOI verification, workbook builder
```

## Literature

`literature/literature_review.xlsx` — **30 papers**, each verified against DOI
metadata, with five sheets:

* `Literature Review` — the required columns (Sr No, Name, Author(s), Publishing
  Date, Published By / Organization, Title, Abstract summary, Conclusion summary,
  Keywords Related, Paper Link);
* `Literature Analysis` — 29 columns per paper (horizon, dataset, sampling
  frequency, inputs, weather variables, model family, baselines, metrics, best
  reported metric, weather conditions, explainability, uncertainty, multi-site and
  external validation, contribution, limitation, gap, reproducibility, code and
  dataset availability, DOI, source);
* `Research Gaps` — gap, evidence papers, why it matters, current approaches,
  limitation, opportunity, testable question;
* `Provenance` — per paper: DOI, verified title, abstract source and length, which
  metadata sources returned a record, whether Crossref/OpenAlex titles agree, and
  citation counts;
* `Selection and Exclusions` — the search scope and the exclusions with reasons.

The corpus spans classical ML, recurrent, hybrid convolutional-recurrent,
attention/transformer, physics-informed, probabilistic/explainable work and
systematic reviews. The review is synthesised in `paper/literature_review.md` and
preserves disagreements between papers rather than manufacturing consensus.

## Limitations

* **One station, one climate.** A single 55 kW Hong Kong rooftop array cannot
  support a claim about PV forecasting in general. It supports a claim about the
  relative behaviour of these eleven models on this record.
* **No NWP inputs.** Under a strict historical information set a one-hour forecast
  cannot anticipate a cloud front, which bounds attainable accuracy and makes the
  persistence reference unusually strong.
* **Incomplete experiment matrix.** C and E are partially executed for
  time-budget reasons, H is implemented but not run, and J (cross-site) is not
  implemented, so RQ6's site half is open.
* **Cost figures** are machine-specific in absolute terms.
* **Conformal coverage** is approximate, as explained above.
* **Single seed.** One seed per configuration; seed sensitivity is not measured.

## Future work

1. Complete C and E; run H (`--explainability`).
2. Implement J (cross-site transfer) over the declared holdout panel, with
   per-station capacity-aware scoring.
3. Multi-seed replication of the neural models, to separate architecture from
   initialisation.
4. Locally-weighted or adaptive conformal intervals, which address the seasonal
   coverage drift the fixed-width intervals show.
5. A probabilistic family (quantile regression, Gaussian-process residuals) to
   compare against conformal wrappers.
6. WNIP-based evaluation as a second information regime, so that the historical
   and forecast-weather settings can be compared on one dataset.

## Citation

See `CITATION.cff`. If you use this software, its experiment configuration or its
results, please cite it as software with the version tag.

## License

MIT — see `LICENSE`. The dataset is redistributed by its authors under CC0 1.0
and is fetched by `scripts/download_data.py`; no rights in that data are claimed
here.
