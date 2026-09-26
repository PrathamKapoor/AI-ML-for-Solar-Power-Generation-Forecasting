# Dataset

## Selection

The case study uses a single openly licensed dataset, chosen against the
assignment's criteria rather than by convenience.

| Criterion | How it is satisfied here |
| --- | --- |
| Accessibility | No registration, API key or click-through: a direct Zenodo file download, checksum-verified |
| Licensing | **CC0 1.0** — public-domain dedication, redistributable, no attribution condition |
| Data quality | Measured inverter-level AC power with a nameplate cross-check; 15-minute grid with a reported missing rate of 0.023% for the case-study station |
| Time resolution | Native 5-minute (44 files), 15-minute (37 files) and hourly (22 files) |
| Temporal coverage | 2021-06-01 to 2023-12-31, a full three years including one complete calendar year for the test split |
| Weather variables | Co-located 1-minute GHI, temperature, relative humidity, sea-level pressure, visibility, wind speed and direction, rainfall |
| Multi-site | 60 stations spanning roughly 4 kW to 55 kW rated capacity |
| Suitability | Supports all four horizons exactly, four weather regimes, four seasons and (in principle) cross-site transfer |

**Citation.** Lin, Z., Zhou, Q., Wang, Z., Wang, C., Bookhart, D. B., &
Leung-Shea, M. (2024). *A high-resolution three-year dataset supporting rooftop
photovoltaics (PV) generation analytics.* Dryad.
https://doi.org/10.5061/dryad.m37pvmd99
Distributed as Zenodo record [10909062](https://zenodo.org/records/10909062) under
CC0 1.0.

Alternatives considered and rejected: PVDAQ (registration required for several
systems, heterogeneous sampling and inconsistent station metadata) and the NREL
Solar Power Data (one site, flat terrain, no multi-station panel). Kaggle
aggregations were not used: their provenance and licensing are not verifiable.

## Archive layout

```
Dataset/
├── Metadata/PV generation system metadata.ttl          # capacities, coordinates
└── Time series dataset/
    ├── PV generation dataset/
    │   ├── PV stations with panel level optimizer/
    │   │   ├── Inverter level dataset/                  # 44 files, 5-minute
    │   │   └── Site level dataset/                      # 37 files, 15-minute
    │   └── PV stations without panel level optimizer/
    │       └── Site level dataset/                      # 22 files, hourly
    └── Meteorological dataset/
        ├── Irradiance/Irradiance_{2021,2022,2023}.csv
        ├── Temperature/Temperature_{2021,2022,2023}.csv
        ├── Relative Humidity/Relative Humidity_{2021,2022,2023}.csv
        ├── Sea Level Pressure/Sea Level Pressure_{2021,2022,2023}.csv
        ├── Visibility/Visibility_{2021,2022,2023}.csv
        ├── Wind/Wind_{2021,2022,2023}.csv
        └── Rainfall/Rainfall_{2022,2023}.{csv,xlsx}
```

126 files, 1.03 GB extracted, 296 MB compressed. The download manifest at
`data/raw/download_manifest.json` records the URL, the byte count, the SHA-256
checksum and the required station names, and the acquisition step fails if any
checksum mismatches.

## Case-study station: LSK North

Selected because it has, simultaneously:

* the **longest continuous 15-minute record** (2021-06-01 to 2023-12-31, 90 624
  steps, no absent timestamps);
* one of the **lowest missing-value rates** among the large stations (0.023%);
* a **rated capacity that matches the measured peak** — 55.0 kW nameplate against
  a 55 294 W observed maximum, a ratio of 1.005. This is an independent
  consistency check on the target series: a mis-scaled export or a unit error
  would show up here immediately.

## Working resolution and the case-study record

15 minutes is the finest resolution with continuous multi-year coverage, and it
makes every target horizon an exact integer number of steps (1, 4, 24, 96). No
horizon requires interpolation or resampling.

After cleaning, the case-study record used in the experiments contains **90 528
15-minute steps** with these measured properties:

| Statistic | Value |
| --- | --- |
| Steps | 90 528 |
| Mean power, all steps | 8 305 W |
| Standard deviation | 13 399 W |
| Median | 0 W |
| 95th percentile | 39 912 W |
| Maximum | 55 295 W |
| **Fraction of exactly-zero steps** | **50.1%** |
| Rated capacity | 55 000 W |

The zero fraction is the single most important number in this section: it is why
MAPE is undefined for this target, why night-time steps are scored separately, and
why a model that predicts zero perfectly at night still has to be judged on its
daylight behaviour.

## Split

Strictly chronological, with explicit dates so the split is human-readable in the
paper and cannot drift when the code changes:

| Split | Period | Role |
| --- | --- | --- |
| Train | 2021-06-01 → 2022-08-31 (15 months, 55%) | fitting and hyperparameter selection |
| Validation | 2022-09-01 → 2022-12-31 (4 months, 15%) | early stopping, tuning, model selection |
| Test | 2023-01-01 → 2023-12-31 (12 months, 30%) | scored once, after training |

The test period is a full calendar year, so every season appears exactly once and
seasonal generalisation is not confounded with the split. The validation period
covers a different season from the tail of the training period, which discourages
tuning to a single season.

## Weather-regime taxonomy

Defined in `configs/data.yaml` under `regimes` and implemented in
`src/solar_forecasting/evaluation/regimes.py`. From the *calibrated* clearness
index and the trailing irradiance variability, both computed from information that
exists at forecast time:

```
kt       = clip(GHI / (alpha * GHI_clear), 0, 1.2)      alpha fitted on the training split
sigma_cv = std / mean of daylight GHI over a trailing 4-hour window
```

Applied in this order of precedence:

| Condition | Label |
| --- | --- |
| solar elevation ≤ 0° | Night |
| `sigma_cv > 0.60` | High-variability (overrides the clearness bands) |
| `kt ≥ 0.72` and `sigma_cv ≤ 0.60` | Clear |
| `0.45 ≤ kt < 0.72` and `sigma_cv ≤ 0.60` | Partly cloudy |
| otherwise | Cloudy |

**Threshold provenance.** The `kt` cuts were chosen from percentiles of the
*training* distribution — `kt ≥ 0.72` selects the clearest 34% of usable daylight
steps and `kt ≥ 0.45` splits the remainder near its 31st percentile — and
`sigma_cv > 0.60` is the 80th percentile of the same training distribution. The
evaluation period plays no part in defining the classes.

**Why variability takes precedence.** A step can be bright on average yet
operationally useless; short-term ramping is what determines whether a forecast is
actionable, so a variable step is labelled as such even when its mean clearness is
high.

**Why the clearness index is calibrated.** Linke turbidity is fixed at 4.0, a
standard mid-range value for coastal humid-subtropical Hong Kong. The choice
matters: at turbidity 3.0 the raw index reaches 1.23 at its 90th percentile, which
is unphysical, whereas at 4.0 its 99th percentile is about 1.03. A single scale
factor is then estimated on the training split so that the 99th percentile of the
raw index maps to 1.0. **The measured irradiance is never rescaled**; only the
dimensionless index used for labelling is, and the estimated factor is reported in
`results/metrics/pipeline_report.json` as a data-quality diagnostic.

**Regime labels are assigned after prediction, from the realised irradiance, and
are never supplied to a model.** They are a labelling device for error
stratification: they describe the conditions the model actually had to predict.

## Known data issues handled explicitly

| Issue | Handling |
| --- | --- |
| Night-time pyranometer offset (median ≈ 5 W/m², 95th percentile ≈ 31 W/m² over three years) | zeroed below the horizon; otherwise it would propagate into `kt` and corrupt the regime labels |
| Rainfall absent for 2021 | excluded from the default feature set; retained for the missing-value analysis and available to the ablation |
| 5-minute inverter files alongside 15-minute site files | the 15-minute site-level series is used, so no resampling is applied to the target |
| Sea-level pressure and visibility near-constant over a 6-hour window | excluded from the default inputs; their value is examined directly in the ablation |
| Stations with hourly-only records | retained as a secondary resolution check, not used for the case study |
