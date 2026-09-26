# Reproducibility

## Principle

A result is reproducible here if a reader can identify the exact code, data,
configuration, seed and environment that produced it. Every run therefore writes a
self-describing JSON record, and the summary tables are derived from those records
rather than from anything a human typed.

## One-command reproduction

```bash
python -m pip install -r requirements.txt     # or: mamba env create -f environment.yml
python scripts/download_data.py               # fetch and verify the dataset
python scripts/preprocess.py                  # run the data pipeline
python scripts/run_experiments.py             # run the declared experiment matrix
python scripts/evaluate.py --report           # tables, stratified analysis, statistics, figures
python scripts/generate_report.py             # assemble docs/RESULTS_REPORT.md
python -m pytest -q                           # 108 tests
```

Full experiment detail: `docs/experiments.md`. Installation notes: `README.md`.

## Seeds

* Every experiment uses the seed declared in `configs/experiments.yaml`
  (`defaults.seed: 42`), recorded as `seed` in the run record.
* `solar_forecasting.utils.seeding.set_seed` seeds `random`, `numpy` and `torch`,
  and requests deterministic torch kernels where they are available.
* `run_metadata(seed)` captures the git revision, the platform, the interpreter
  version and the versions of every package that affects a number.
* The moving-block bootstrap, the conformal calibration and the permutation
  importance each draw from an explicitly seeded generator, so their reported
  intervals are reproducible too. A consequence: the spread reported for
  permutation importance is the spread *over permutations*, not over seeds. The
  model initialisation is fixed upstream by the experiment seed.

## What each run records

`results/experiments/<experiment_id>.json`:

| Field group | Contents |
| --- | --- |
| Identity | `experiment_id`, model, family, kind, horizon and horizon steps, feature specification, seed |
| Data | dataset name, station, rated capacity, split sizes, and the **target** date range of each split |
| Inputs | the exact feature-column list, lookback length, and the information set statement |
| Training | epochs run, best epoch, validation MAE in watts, early-stopping flag, tuning grid and winner |
| Cost | training seconds, inference ms per window, parameter count |
| Metrics | daylight and all-step metric dictionaries, including both persistence references and both skill scores |
| Provenance | configuration fingerprint, git revision, Python version, torch version, package versions |
| Behaviour | the epoch-by-epoch training history, and the regime distribution of the test period |

`results/experiments.csv` is a **derived index**: `scripts/evaluate.py` rebuilds it
from the JSON records at the start of every evaluation. The records are the source
of truth, which is why a run that finished just before an interruption is not lost,
and why parallel experiment groups cannot corrupt it.

## Data provenance

`data/raw/download_manifest.json` records the archive URL, the byte count, the
**SHA-256 checksum**, the licence, the working resolution and the station names the
project requires. `scripts/download_data.py` verifies the checksum and refuses to
proceed on a mismatch, and confirms that every required station is present in the
archive. Raw and processed data are not committed; they are reproduced from the
documented source.

## Environment

* `requirements.txt` pins the exact versions the reported results were produced
  in; `environment.yml` provides the equivalent conda environment; `pyproject.toml`
  declares the package metadata and the pytest configuration.
* The run environment was Python 3.13.14 on Windows 11, CPU only. The code contains
  no machine-specific path: the project root is derived from the package location
  and every output directory is created on demand.
* BLAS and OpenMP thread counts are pinned to 1 for every run
  (`set_thread_env`). Without this, an 8-thread BLAS can make a small model look
  faster than a large one purely through parallelisation, which would corrupt the
  cost comparison in Experiment I. Absolute durations remain machine-specific; only
  the ranking is portable, and the cost table says so.
* `Dockerfile` is **not** provided. It was not tested in this environment, and
  shipping an untested container file would be worse than documenting the
  environment in `requirements.txt` and `environment.yml`, both of which were used.

## Determinism: what is and is not guaranteed

Guaranteed:

* the data pipeline is deterministic given the same archive and configuration;
* the classical models are deterministic under a fixed `random_state`;
* the bootstrap, conformal and permutation procedures are seeded explicitly.

Not guaranteed:

* bit-identical neural-network training across machines. Torch CPU kernels and the
  order of floating-point reductions can differ between BLAS builds, so a rerun on
  different hardware may move a neural model's metrics in the third decimal place
  and can occasionally change which epoch early stopping selects. The seed, the
  split and the configuration are recorded precisely so that such a difference is
  detectable rather than mysterious.
* run-to-run timing, and therefore anything derived from timing.

## Leakage audit

The controls, and where each is enforced, are tabulated in
`docs/methodology.md` §9. Three of them are additionally *asserted at run time*:

1. `verify_causality` re-derives sampled windows from the source frame and fails
   the run on the first mismatch, before the first optimiser step.
2. `fit_scalers` raises on non-finite training values and on zero-variance
   features, so a scaler can never silently divide by zero.
3. `chronological_split` drops rows outside the configured window and reports how
   many, so a configuration change cannot silently include unintended data.

The test suite covers each of these, along with the case that motivated the fix in
`persistence_from_lag` (a documented lag direction that the implementation had
inverted for every lag greater than zero — a comparison that is now consistent
with the `pv_power_w_lag_k` features).

## Verification performed

```bash
python -m pytest -q                                   # 108 passed
python -m compileall -q src/solar_forecasting scripts  # no syntax errors
python scripts/run_experiments.py --list              # matrix resolves from config
python scripts/evaluate.py --report --uncertainty     # tables, statistics, figures
python scripts/generate_report.py                     # docs/RESULTS_REPORT.md
```

`results/metrics/evaluation_report.json` records, for every evaluation stage,
whether it ran, what it wrote, and — for any stage that was skipped — the reason.
This is what distinguishes "no result" from "not attempted".
