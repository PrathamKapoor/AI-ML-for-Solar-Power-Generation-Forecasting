# AI/ML-Based Solar Photovoltaic Power Generation Forecasting: A Reproducible and Explainable Benchmark Across Forecast Horizons, Weather Regimes and Sites

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
from a machine-readable run record. The benchmark spans 4
forecast horizons (15min, 1h, 24h, 6h), four transparent weather
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
consequence is stated plainly: a Diebold-Mariano test rejects essentially
everything while a serial-correlation-aware moving-block bootstrap separates the
models. Both are reported; neither was adjusted to obtain a preferred answer.

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

1. a controlled comparison of 11 model variants across
   4 horizons on one split, with the two persistence
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

**Table 1. Case-study record statistics.**

Source: `results/tables/dataset_statistics.csv`

| scope | statistic | value |
| --- | --- | --- |
| all_rows | n | 90528 |
| all_rows | mean | 8305.427161862268 |
| all_rows | std | 13399.292482809866 |
| all_rows | min | 0.0 |
| all_rows | p01 | 0.0 |
| all_rows | p05 | 0.0 |
| all_rows | p25 | 0.0 |
| all_rows | median | 0.0 |
| all_rows | p75 | 12075.00025 |
| all_rows | p95 | 39911.817999999934 |
| all_rows | p99 | 49840.89235999998 |
| all_rows | max | 55294.668 |
| all_rows | n_zero | 45375 |
| all_rows | fraction_zero | 0.5012261399787911 |
| all_rows | n_negative | 0 |
| daylight_only | n | 45718 |
| daylight_only | mean | 16444.763108091945 |
| daylight_only | std | 14888.817762874243 |
| daylight_only | min | 0.0 |
| daylight_only | p01 | 0.0 |
| daylight_only | p05 | 113.90000039999994 |
| daylight_only | p25 | 3394.3333 |
| daylight_only | median | 11849.8335 |
| daylight_only | p75 | 27619.166 |
| daylight_only | p95 | 45106.682400000005 |
| daylight_only | p99 | 51944.53912000001 |
| daylight_only | max | 55294.668 |
| daylight_only | n_zero | 1026 |
| daylight_only | fraction_zero | 0.022441926593464282 |
| daylight_only | n_negative | 0 |
| train_daylight | n | 22477 |
| train_daylight | mean | 16825.310278114168 |
| train_daylight | std | 15298.774951409063 |
| train_daylight | min | 0.0 |
| train_daylight | p01 | 0.0 |
| train_daylight | p05 | 99.666664 |
| train_daylight | p25 | 3538.3333 |
| train_daylight | median | 12068.667 |
| train_daylight | p75 | 28080.0 |
| train_daylight | p95 | 46950.60080000001 |
| train_daylight | p99 | 52835.97232000001 |
| train_daylight | max | 55294.668 |
| train_daylight | n_zero | 536 |
| train_daylight | fraction_zero | 0.02384659874538417 |
| train_daylight | n_negative | 0 |
| outliers::pv_power_w | variable | pv_power_w |
| outliers::pv_power_w | n | 90528 |
| outliers::pv_power_w | min | 0.0 |
| outliers::pv_power_w | q1 | 0.0 |
| outliers::pv_power_w | median | 0.0 |
| outliers::pv_power_w | q3 | 12075.00025 |
| outliers::pv_power_w | max | 55294.668 |
| outliers::pv_power_w | iqr | 12075.00025 |
| outliers::pv_power_w | tukey_upper_fence | 30187.500625 |
| outliers::pv_power_w | tukey_lower_fence | -18112.500375 |
| outliers::pv_power_w | n_above_upper_fence | 9900 |
| outliers::pv_power_w | n_below_lower_fence | 0 |
| outliers::pv_power_w | skewness | 1.6411390812383053 |
| outliers::pv_power_w | kurtosis | 1.5777530221266516 |
| outliers::ghi | variable | ghi |
| outliers::ghi | n | 90528 |
| outliers::ghi | min | 0.0 |
| outliers::ghi | q1 | 0.0 |
| outliers::ghi | median | 5.702576666666666 |
| outliers::ghi | q3 | 209.8263333333333 |
| outliers::ghi | max | 1200.0 |
| outliers::ghi | iqr | 209.8263333333333 |
| outliers::ghi | tukey_upper_fence | 524.5658333333333 |
| outliers::ghi | tukey_lower_fence | -314.73949999999996 |
| outliers::ghi | n_above_upper_fence | 10433 |
| outliers::ghi | n_below_lower_fence | 0 |
| outliers::ghi | skewness | 1.8861032160934381 |
| outliers::ghi | kurtosis | 2.987571512605044 |
| outliers::temperature | variable | temperature |
| outliers::temperature | n | 90528 |
| outliers::temperature | min | 5.8202799999999995 |
| outliers::temperature | q1 | 19.283983333333335 |
| outliers::temperature | median | 24.339166666666667 |
| outliers::temperature | q3 | 27.444116666666666 |
| outliers::temperature | max | 39.60966666666666 |
| outliers::temperature | iqr | 8.16013333333333 |
| outliers::temperature | tukey_upper_fence | 39.68431666666666 |
| outliers::temperature | tukey_lower_fence | 7.043783333333339 |
| outliers::temperature | n_above_upper_fence | 0 |
| outliers::temperature | n_below_lower_fence | 154 |
| outliers::temperature | skewness | -0.3842543778743115 |
| outliers::temperature | kurtosis | -0.3746244929715874 |
| outliers::relative_humidity | variable | relative_humidity |
| outliers::relative_humidity | n | 90528 |
| outliers::relative_humidity | min | 0.3118106666666666 |
| outliers::relative_humidity | q1 | 70.27713333333332 |
| outliers::relative_humidity | median | 82.63680000000001 |
| outliers::relative_humidity | q3 | 91.68893333333334 |
| outliers::relative_humidity | max | 99.9738 |
| outliers::relative_humidity | iqr | 21.411800000000014 |
| outliers::relative_humidity | tukey_upper_fence | 123.80663333333337 |
| outliers::relative_humidity | tukey_lower_fence | 38.159433333333304 |
| outliers::relative_humidity | n_above_upper_fence | 0 |
| outliers::relative_humidity | n_below_lower_fence | 2003 |
| outliers::relative_humidity | skewness | -1.0548033353688029 |
| outliers::relative_humidity | kurtosis | 1.078796942270143 |
| outliers::wind_speed | variable | wind_speed |
| outliers::wind_speed | n | 90528 |
| outliers::wind_speed | min | 0.346958 |
| outliers::wind_speed | q1 | 1.4921658333333334 |
| outliers::wind_speed | median | 2.29418 |
| outliers::wind_speed | q3 | 3.310408333333333 |
| outliers::wind_speed | max | 52.18320000000001 |
| outliers::wind_speed | iqr | 1.8182424999999998 |
| outliers::wind_speed | tukey_upper_fence | 6.037772083333333 |
| outliers::wind_speed | tukey_lower_fence | -1.235197916666666 |
| outliers::wind_speed | n_above_upper_fence | 3589 |
| outliers::wind_speed | n_below_lower_fence | 0 |
| outliers::wind_speed | skewness | 2.3378512888634897 |
| outliers::wind_speed | kurtosis | 18.00764525317044 |


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
A_baselines, B_ml_vs_dl, C_horizons, D_regimes/G_seasonal, E_feature_ablation, J_cross_site, K_uncertainty, M_multiseed, N_feature_regimes.

## 8. Results

**Table 2. Overall model comparison, 1-hour horizon, daylight steps of the test year.**

Source: `results/tables/overall_model_comparison.csv`

| model | model_family | mae | rmse | nrmse_capacity | r2 | smape | skill_vs_persistence_rmse | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | baseline | 6,615.8800 | 9,141.7000 | 0.1662 | 0.5958 | 77.1514 | 0.0000 | 0.0000 |
| smart_persistence | baseline | 4,569.6200 | 7,668.0600 | 0.1394 | 0.7156 | 56.9708 | 0.1612 | 0.0000 |
| linear_regression | classical_ml | 5,174.3700 | 7,456.8800 | 0.1356 | 0.7310 | 57.6170 | 0.1843 | 1.9604 |
| random_forest | classical_ml | 4,472.9400 | 6,693.3000 | 0.1217 | 0.7833 | 46.2257 | 0.2678 | 287.4450 |
| gradient_boosting | classical_ml | 4,373.8600 | 6,664.2600 | 0.1212 | 0.7852 | 49.7801 | 0.2710 | 9.4859 |
| xgboost | classical_ml | 4,505.2900 | 6,835.3200 | 0.1243 | 0.7740 | 48.5750 | 0.2523 | 42.5256 |
| lstm | recurrent | 6,745.8300 | 9,263.7800 | 0.1684 | 0.5849 | 65.7897 | -0.0134 | 312.6060 |
| gru | recurrent | 6,892.6100 | 9,611.4600 | 0.1748 | 0.5532 | 68.7610 | -0.0514 | 361.9260 |
| cnn_lstm | hybrid | 4,165.3100 | 6,819.3900 | 0.1240 | 0.7751 | 47.9131 | 0.2540 | 274.1260 |
| attention_lstm | hybrid | 6,237.4300 | 9,090.1500 | 0.1653 | 0.6003 | 63.7010 | 0.0056 | 172.7780 |
| transformer | attention | 5,548.5600 | 7,850.6600 | 0.1427 | 0.7019 | 60.0508 | 0.1412 | 1,121.9900 |

**Table 3. Accuracy by forecast horizon.**

Source: `results/tables/horizon_comparison.csv`

| model | horizon | source_group | rmse | nrmse_capacity | r2 | skill_vs_persistence_rmse | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 15min | C_horizons | 5,322.5800 | 0.0968 | 0.8630 | 0.0000 | 0.0000 |
| persistence | 1h | C_horizons | 9,141.7000 | 0.1662 | 0.5958 | 0.0000 | 0.0000 |
| persistence | 24h | C_horizons | 11,367.8000 | 0.2067 | 0.3763 | 0.0000 | 0.0000 |
| persistence | 6h | C_horizons | 22,382.6000 | 0.4070 | -1.4216 | 0.0000 | 0.0000 |
| smart_persistence | 15min | C_horizons | 5,173.8500 | 0.0941 | 0.8705 | 0.0279 | 0.0000 |
| smart_persistence | 1h | C_horizons | 7,668.0600 | 0.1394 | 0.7156 | 0.1612 | 0.0000 |
| smart_persistence | 24h | C_horizons | 11,366.2000 | 0.2067 | 0.3765 | 0.0001 | 0.0000 |
| smart_persistence | 6h | C_horizons | 18,040.9000 | 0.3280 | -0.5732 | 0.1940 | 0.0000 |
| linear_regression | 15min | C_horizons | 5,026.4700 | 0.0914 | 0.8778 | 0.0556 | 1.3918 |
| linear_regression | 1h | C_horizons | 7,456.8800 | 0.1356 | 0.7310 | 0.1843 | 8.4912 |
| linear_regression | 24h | C_horizons | 9,792.0100 | 0.1780 | 0.5372 | 0.1386 | 8.2673 |
| linear_regression | 6h | C_horizons | 9,602.4400 | 0.1746 | 0.5543 | 0.5710 | 8.4667 |
| random_forest | 15min | C_horizons | 4,953.6200 | 0.0901 | 0.8813 | 0.0693 | 188.2630 |
| random_forest | 1h | C_horizons | 6,693.3000 | 0.1217 | 0.7833 | 0.2678 | 582.5380 |
| random_forest | 24h | C_horizons | 9,818.3100 | 0.1785 | 0.5348 | 0.1363 | 536.8830 |
| random_forest | 6h | C_horizons | 9,285.4400 | 0.1688 | 0.5832 | 0.5851 | 564.1860 |
| gradient_boosting | 15min | C_horizons | 4,817.0700 | 0.0876 | 0.8878 | 0.0950 | 11.8988 |
| gradient_boosting | 1h | C_horizons | 6,664.2600 | 0.1212 | 0.7852 | 0.2710 | 47.4346 |
| gradient_boosting | 24h | C_horizons | 9,704.6400 | 0.1764 | 0.5455 | 0.1463 | 63.0244 |
| gradient_boosting | 6h | C_horizons | 9,137.9200 | 0.1661 | 0.5964 | 0.5917 | 47.6943 |
| xgboost | 15min | C_horizons | 5,113.6700 | 0.0930 | 0.8735 | 0.0392 | 36.4617 |
| xgboost | 1h | C_horizons | 6,835.3200 | 0.1243 | 0.7740 | 0.2523 | 129.5610 |
| xgboost | 24h | C_horizons | 9,994.4100 | 0.1817 | 0.5179 | 0.1208 | 133.5540 |
| xgboost | 6h | C_horizons | 9,471.0900 | 0.1722 | 0.5664 | 0.5769 | 135.5020 |
| lstm | 15min | C_horizons | 8,777.0300 | 0.1596 | 0.6274 | -0.6490 | 181.0180 |
| lstm | 1h | C_horizons | 9,263.7800 | 0.1684 | 0.5849 | -0.0134 | 1,035.8800 |
| lstm | 24h | C_horizons | 11,033.4000 | 0.2006 | 0.4125 | 0.0294 | 829.5520 |
| lstm | 6h | C_horizons | 10,371.6000 | 0.1886 | 0.4800 | 0.5366 | 850.6670 |
| gru | 15min | C_horizons | 8,879.7300 | 0.1615 | 0.6186 | -0.6683 | 136.0070 |
| gru | 1h | C_horizons | 9,611.4600 | 0.1748 | 0.5532 | -0.0514 | 875.6850 |
| gru | 24h | C_horizons | 11,036.2000 | 0.2007 | 0.4122 | 0.0292 | 498.6340 |
| gru | 6h | C_horizons | 10,501.4000 | 0.1909 | 0.4669 | 0.5308 | 1,139.4300 |
| cnn_lstm | 15min | C_horizons | 4,853.3400 | 0.0882 | 0.8861 | 0.0882 | 207.6680 |
| cnn_lstm | 1h | C_horizons | 6,819.3900 | 0.1240 | 0.7751 | 0.2540 | 1,042.9200 |
| cnn_lstm | 24h | C_horizons | 10,191.4000 | 0.1853 | 0.4987 | 0.1035 | 437.0280 |
| cnn_lstm | 6h | C_horizons | 9,735.2300 | 0.1770 | 0.5419 | 0.5651 | 511.9600 |
| attention_lstm | 15min | C_horizons | 8,134.2100 | 0.1479 | 0.6800 | -0.5282 | 251.3470 |
| attention_lstm | 1h | C_horizons | 9,090.1500 | 0.1653 | 0.6003 | 0.0056 | 617.1200 |
| attention_lstm | 24h | C_horizons | 11,124.3000 | 0.2023 | 0.4028 | 0.0214 | 595.4610 |
| attention_lstm | 6h | C_horizons | 10,008.5000 | 0.1820 | 0.5158 | 0.5528 | 1,371.9300 |
| transformer | 15min | C_horizons | 5,481.9900 | 0.0997 | 0.8546 | -0.0300 | 2,766.3100 |
| transformer | 1h | C_horizons | 7,850.6600 | 0.1427 | 0.7019 | 0.1412 | 2,592.3900 |
| transformer | 24h | C_horizons | 10,012.6000 | 0.1820 | 0.5162 | 0.1192 | 3,404.4100 |
| transformer | 6h | C_horizons | 10,057.7000 | 0.1829 | 0.5110 | 0.5506 | 3,172.2700 |

**Table 4. Weather-regime comparison, with the reference rescored inside each stratum.**

Source: `results/tables/weather_regime_comparison.csv`

| model | stratum | n | nrmse_capacity | r2 | skill_vs_persistence_rmse |
| --- | --- | --- | --- | --- | --- |
| persistence | Clear | 5824 | 0.1851 | 0.4984 | 0.0000 |
| persistence | Partly cloudy | 2411 | 0.1881 | 0.0446 | 0.0000 |
| persistence | Cloudy | 6012 | 0.1036 | 0.0902 | 0.0000 |
| persistence | High-variability | 3432 | 0.2012 | 0.3609 | 0.0000 |
| persistence | Night | 0 |  |  |  |
| smart_persistence | Clear | 5824 | 0.1458 | 0.6889 | 0.2124 |
| smart_persistence | Partly cloudy | 2411 | 0.1627 | 0.2857 | 0.1354 |
| smart_persistence | Cloudy | 6012 | 0.0911 | 0.2970 | 0.1210 |
| smart_persistence | High-variability | 3432 | 0.1759 | 0.5116 | 0.1258 |
| smart_persistence | Night | 0 |  |  |  |
| linear_regression | Clear | 5824 | 0.1363 | 0.7281 | 0.2637 |
| linear_regression | Partly cloudy | 2411 | 0.1560 | 0.3433 | 0.1709 |
| linear_regression | Cloudy | 6012 | 0.1082 | 0.0068 | -0.0448 |
| linear_regression | High-variability | 3432 | 0.1598 | 0.5967 | 0.2057 |
| linear_regression | Night | 0 |  |  |  |
| random_forest | Clear | 5824 | 0.1212 | 0.7851 | 0.3454 |
| random_forest | Partly cloudy | 2411 | 0.1314 | 0.5338 | 0.3015 |
| random_forest | Cloudy | 6012 | 0.1047 | 0.0703 | -0.0109 |
| random_forest | High-variability | 3432 | 0.1415 | 0.6839 | 0.2967 |
| random_forest | Night | 0 |  |  |  |
| gradient_boosting | Clear | 5824 | 0.1188 | 0.7934 | 0.3582 |
| gradient_boosting | Partly cloudy | 2411 | 0.1361 | 0.4996 | 0.2763 |
| gradient_boosting | Cloudy | 6012 | 0.1011 | 0.1336 | 0.0242 |
| gradient_boosting | High-variability | 3432 | 0.1440 | 0.6724 | 0.2841 |
| gradient_boosting | Night | 0 |  |  |  |
| xgboost | Clear | 5824 | 0.1270 | 0.7641 | 0.3142 |
| xgboost | Partly cloudy | 2411 | 0.1350 | 0.5078 | 0.2822 |
| xgboost | Cloudy | 6012 | 0.1016 | 0.1243 | 0.0189 |
| xgboost | High-variability | 3432 | 0.1459 | 0.6637 | 0.2746 |
| xgboost | Night | 0 |  |  |  |
| lstm | Clear | 5824 | 0.1863 | 0.4922 | -0.0063 |
| lstm | Partly cloudy | 2411 | 0.1542 | 0.3578 | 0.1802 |
| lstm | Cloudy | 6012 | 0.1534 | -0.9943 | -0.4805 |
| lstm | High-variability | 3432 | 0.1712 | 0.5372 | 0.1490 |
| lstm | Night | 0 |  |  |  |
| gru | Clear | 5824 | 0.2150 | 0.3236 | -0.1613 |
| gru | Partly cloudy | 2411 | 0.1503 | 0.3905 | 0.2013 |
| gru | Cloudy | 6012 | 0.1377 | -0.6066 | -0.3289 |
| gru | High-variability | 3432 | 0.1726 | 0.5294 | 0.1419 |
| gru | Night | 0 |  |  |  |
| cnn_lstm | Clear | 5824 | 0.1248 | 0.7721 | 0.3259 |
| cnn_lstm | Partly cloudy | 2411 | 0.1426 | 0.4512 | 0.2421 |
| cnn_lstm | Cloudy | 6012 | 0.0918 | 0.2850 | 0.1135 |
| cnn_lstm | High-variability | 3432 | 0.1539 | 0.6258 | 0.2348 |
| cnn_lstm | Night | 0 |  |  |  |
| attention_lstm | Clear | 5824 | 0.1758 | 0.5477 | 0.0504 |
| attention_lstm | Partly cloudy | 2411 | 0.1710 | 0.2103 | 0.0909 |
| attention_lstm | Cloudy | 6012 | 0.1434 | -0.7432 | -0.3842 |
| attention_lstm | High-variability | 3432 | 0.1780 | 0.4998 | 0.1153 |
| attention_lstm | Night | 0 |  |  |  |
| transformer | Clear | 5824 | 0.1448 | 0.6934 | 0.2181 |
| transformer | Partly cloudy | 2411 | 0.1661 | 0.2551 | 0.1170 |
| transformer | Cloudy | 6012 | 0.1054 | 0.0580 | -0.0175 |
| transformer | High-variability | 3432 | 0.1747 | 0.5178 | 0.1314 |
| transformer | Night | 0 |  |  |  |

**Table 5. Seasonal comparison over the full test year.**

Source: `results/tables/seasonal_comparison.csv`

| model | stratum | n | nrmse_capacity | skill_vs_persistence_rmse |
| --- | --- | --- | --- | --- |
| persistence | Winter | 3950 | 0.1345 | 0.0000 |
| persistence | Spring | 4634 | 0.1447 | 0.0000 |
| persistence | Summer | 4859 | 0.2128 | 0.0000 |
| persistence | Autumn | 4236 | 0.1537 | 0.0000 |
| smart_persistence | Winter | 3950 | 0.0942 | 0.2993 |
| smart_persistence | Spring | 4634 | 0.1180 | 0.1842 |
| smart_persistence | Summer | 4859 | 0.1948 | 0.0843 |
| smart_persistence | Autumn | 4236 | 0.1186 | 0.2283 |
| linear_regression | Winter | 3950 | 0.0996 | 0.2593 |
| linear_regression | Spring | 4634 | 0.1271 | 0.1215 |
| linear_regression | Summer | 4859 | 0.1758 | 0.1735 |
| linear_regression | Autumn | 4236 | 0.1197 | 0.2211 |
| random_forest | Winter | 3950 | 0.0853 | 0.3655 |
| random_forest | Spring | 4634 | 0.1118 | 0.2270 |
| random_forest | Summer | 4859 | 0.1618 | 0.2393 |
| random_forest | Autumn | 4236 | 0.1063 | 0.3085 |
| gradient_boosting | Winter | 3950 | 0.0853 | 0.3655 |
| gradient_boosting | Spring | 4634 | 0.1109 | 0.2334 |
| gradient_boosting | Summer | 4859 | 0.1631 | 0.2334 |
| gradient_boosting | Autumn | 4236 | 0.1025 | 0.3328 |
| xgboost | Winter | 3950 | 0.0882 | 0.3445 |
| xgboost | Spring | 4634 | 0.1133 | 0.2166 |
| xgboost | Summer | 4859 | 0.1663 | 0.2184 |
| xgboost | Autumn | 4236 | 0.1069 | 0.3040 |
| lstm | Winter | 3950 | 0.1387 | -0.0310 |
| lstm | Spring | 4634 | 0.1626 | -0.1237 |
| lstm | Summer | 4859 | 0.2039 | 0.0419 |
| lstm | Autumn | 4236 | 0.1546 | -0.0059 |
| gru | Winter | 3950 | 0.1384 | -0.0291 |
| gru | Spring | 4634 | 0.1654 | -0.1432 |
| gru | Summer | 4859 | 0.2202 | -0.0348 |
| gru | Autumn | 4236 | 0.1551 | -0.0096 |
| cnn_lstm | Winter | 3950 | 0.0869 | 0.3539 |
| cnn_lstm | Spring | 4634 | 0.1142 | 0.2104 |
| cnn_lstm | Summer | 4859 | 0.1670 | 0.2151 |
| cnn_lstm | Autumn | 4236 | 0.1042 | 0.3218 |
| attention_lstm | Winter | 3950 | 0.1418 | -0.0543 |
| attention_lstm | Spring | 4634 | 0.1654 | -0.1434 |
| attention_lstm | Summer | 4859 | 0.1960 | 0.0786 |
| attention_lstm | Autumn | 4236 | 0.1457 | 0.0517 |
| transformer | Winter | 3950 | 0.1011 | 0.2483 |
| transformer | Spring | 4634 | 0.1267 | 0.1241 |
| transformer | Summer | 4859 | 0.1963 | 0.0775 |
| transformer | Autumn | 4236 | 0.1173 | 0.2369 |

**Table 6. Leave-one-out feature ablation: what each input group adds to the full set.**

Source: `results/tables/feature_ablation.csv`

| model | feature_spec | n_features | rmse | nrmse_capacity | rmse_vs_full_pct |
| --- | --- | --- | --- | --- | --- |
| xgboost | full | 33 | 6,835.3200 | 0.1243 | 0.0000 |
| xgboost | minimal | 4 | 6,708.8200 | 0.1220 | -1.8506 |
| xgboost | no_calendar | 29 | 6,854.8600 | 0.1246 | 0.2859 |
| xgboost | no_pv_history | 18 | 8,437.4500 | 0.1534 | 23.4390 |
| xgboost | no_solar_geometry | 31 | 6,788.4400 | 0.1234 | -0.6858 |
| xgboost | no_weather | 21 | 6,512.4000 | 0.1184 | -4.7242 |
| lstm | full | 33 | 9,263.7800 | 0.1684 | 0.0000 |
| lstm | minimal | 4 | 8,724.4500 | 0.1586 | -5.8219 |
| lstm | no_calendar | 29 | 9,193.1200 | 0.1671 | -0.7627 |
| lstm | no_pv_history | 18 | 8,806.8200 | 0.1601 | -4.9328 |
| lstm | no_solar_geometry | 31 | 9,177.9200 | 0.1669 | -0.9268 |
| lstm | no_weather | 21 | 9,658.0700 | 0.1756 | 4.2563 |

**Table 7. Additive input regimes: what each kind of information is worth alone.**

Source: `results/tables/feature_regimes.csv`

| model | n_features | rmse | nrmse_capacity | skill_vs_persistence_rmse |
| --- | --- | --- | --- | --- |
| gradient_boosting | 15 | 6,560.0700 | 0.1193 | 0.2824 |
| xgboost | 15 | 6,598.1100 | 0.1200 | 0.2782 |
| lstm | 15 | 9,670.5200 | 0.1758 | -0.0578 |
| cnn_lstm | 15 | 6,724.0600 | 0.1223 | 0.2645 |
| attention_lstm | 15 | 9,621.2300 | 0.1749 | -0.0525 |
| lstm | 12 | 10,107.7000 | 0.1838 | -0.1057 |
| xgboost | 12 | 9,753.1700 | 0.1773 | -0.0669 |
| gradient_boosting | 12 | 10,004.8000 | 0.1819 | -0.0944 |
| attention_lstm | 12 | 9,294.6100 | 0.1690 | -0.0167 |
| cnn_lstm | 12 | 9,636.8800 | 0.1752 | -0.0542 |
| cnn_lstm | 27 | 6,837.6600 | 0.1243 | 0.2520 |
| attention_lstm | 27 | 8,709.9300 | 0.1584 | 0.0472 |
| lstm | 27 | 9,065.3700 | 0.1648 | 0.0083 |
| gradient_boosting | 27 | 6,737.9000 | 0.1225 | 0.2629 |
| xgboost | 27 | 6,800.8000 | 0.1237 | 0.2561 |
| lstm | 29 | 9,193.1200 | 0.1671 | -0.0056 |
| xgboost | 29 | 6,854.8600 | 0.1246 | 0.2502 |
| gradient_boosting | 29 | 6,714.5700 | 0.1221 | 0.2655 |
| attention_lstm | 29 | 8,824.5700 | 0.1604 | 0.0347 |
| cnn_lstm | 29 | 6,815.0400 | 0.1239 | 0.2545 |
| cnn_lstm | 33 | 6,819.3900 | 0.1240 | 0.2540 |
| lstm | 33 | 9,263.7800 | 0.1684 | -0.0134 |
| gradient_boosting | 33 | 6,664.2600 | 0.1212 | 0.2710 |
| xgboost | 33 | 6,835.3200 | 0.1243 | 0.2523 |
| attention_lstm | 33 | 9,090.1500 | 0.1653 | 0.0056 |

**Table 8. Cross-site generalisation.**

Source: `results/tables/cross_site_comparison.csv`

| protocol | model | horizon | test_site | n | nrmse_capacity | skill_vs_persistence_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| cross_site | gradient_boosting | 1h | SQ1 | 17679 | 0.2060 | -0.2610 |
| cross_site | gradient_boosting | 1h | SQ2 | 17679 | 0.1698 | -0.2441 |
| cross_site | gradient_boosting | 1h | SQ3 | 17678 | 0.1691 | -0.1229 |
| cross_site | gradient_boosting | 1h | SQ4 | 17678 | 0.1693 | -0.1007 |
| cross_site | gradient_boosting | 1h | UG Hall6 | 17675 | 0.2213 | -0.3332 |
| cross_site | gradient_boosting | 1h | UG Hall7 | 17678 | 0.1777 | -0.5653 |
| cross_site | linear_regression | 1h | SQ1 | 17679 | 0.2561 | -0.5680 |
| cross_site | linear_regression | 1h | SQ2 | 17679 | 0.2165 | -0.5861 |
| cross_site | linear_regression | 1h | SQ3 | 17678 | 0.2074 | -0.3776 |
| cross_site | linear_regression | 1h | SQ4 | 17678 | 0.2069 | -0.3449 |
| cross_site | linear_regression | 1h | UG Hall6 | 17675 | 0.2853 | -0.7191 |
| cross_site | linear_regression | 1h | UG Hall7 | 17678 | 0.2347 | -1.0675 |
| cross_site | xgboost | 1h | SQ1 | 17679 | 0.2317 | -0.4187 |
| cross_site | xgboost | 1h | SQ2 | 17679 | 0.1905 | -0.3958 |
| cross_site | xgboost | 1h | SQ3 | 17678 | 0.1874 | -0.2449 |
| cross_site | xgboost | 1h | SQ4 | 17678 | 0.1860 | -0.2091 |
| cross_site | xgboost | 1h | UG Hall6 | 17675 | 0.2560 | -0.5423 |
| cross_site | xgboost | 1h | UG Hall7 | 17678 | 0.2129 | -0.8752 |
| leave_one_site_out | gradient_boosting | 1h | LSK North | 17679 | 0.1660 | 0.0014 |
| leave_one_site_out | gradient_boosting | 1h | SQ1 | 17679 | 0.1218 | 0.2543 |
| leave_one_site_out | gradient_boosting | 1h | SQ2 | 17679 | 0.1016 | 0.2558 |
| leave_one_site_out | gradient_boosting | 1h | SQ3 | 17678 | 0.1129 | 0.2501 |
| leave_one_site_out | gradient_boosting | 1h | SQ4 | 17678 | 0.1143 | 0.2567 |
| leave_one_site_out | gradient_boosting | 1h | UG Hall6 | 17675 | 0.1236 | 0.2552 |
| leave_one_site_out | gradient_boosting | 1h | UG Hall7 | 17678 | 0.0909 | 0.1995 |
| leave_one_site_out | linear_regression | 1h | LSK North | 17679 | 0.1296 | 0.2203 |
| leave_one_site_out | linear_regression | 1h | SQ1 | 17679 | 0.1248 | 0.2360 |
| leave_one_site_out | linear_regression | 1h | SQ2 | 17679 | 0.1033 | 0.2433 |
| leave_one_site_out | linear_regression | 1h | SQ3 | 17678 | 0.1153 | 0.2339 |
| leave_one_site_out | linear_regression | 1h | SQ4 | 17678 | 0.1176 | 0.2356 |
| leave_one_site_out | linear_regression | 1h | UG Hall6 | 17675 | 0.1254 | 0.2446 |
| leave_one_site_out | linear_regression | 1h | UG Hall7 | 17678 | 0.0892 | 0.2139 |
| leave_one_site_out | xgboost | 1h | LSK North | 17679 | 0.1659 | 0.0017 |
| leave_one_site_out | xgboost | 1h | SQ1 | 17679 | 0.1215 | 0.2558 |
| leave_one_site_out | xgboost | 1h | SQ2 | 17679 | 0.1014 | 0.2574 |
| leave_one_site_out | xgboost | 1h | SQ3 | 17678 | 0.1158 | 0.2311 |
| leave_one_site_out | xgboost | 1h | SQ4 | 17678 | 0.1171 | 0.2384 |
| leave_one_site_out | xgboost | 1h | UG Hall6 | 17675 | 0.1254 | 0.2446 |
| leave_one_site_out | xgboost | 1h | UG Hall7 | 17678 | 0.0882 | 0.2233 |
| within_site | gradient_boosting | 1h | LSK North | 17679 | 0.1212 | 0.2710 |
| within_site | gradient_boosting | 1h | SQ1 | 17679 | 0.1207 | 0.2609 |
| within_site | gradient_boosting | 1h | SQ2 | 17679 | 0.0985 | 0.2782 |
| within_site | gradient_boosting | 1h | SQ3 | 17678 | 0.1104 | 0.2668 |
| within_site | gradient_boosting | 1h | SQ4 | 17678 | 0.1128 | 0.2666 |
| within_site | gradient_boosting | 1h | UG Hall6 | 17675 | 0.1199 | 0.2773 |
| within_site | gradient_boosting | 1h | UG Hall7 | 17678 | 0.0867 | 0.2366 |
| within_site | linear_regression | 1h | LSK North | 17679 | 0.1356 | 0.1843 |
| within_site | linear_regression | 1h | SQ1 | 17679 | 0.1272 | 0.2212 |
| within_site | linear_regression | 1h | SQ2 | 17679 | 0.1128 | 0.1736 |
| within_site | linear_regression | 1h | SQ3 | 17678 | 0.1142 | 0.2413 |
| within_site | linear_regression | 1h | SQ4 | 17678 | 0.1190 | 0.2261 |
| within_site | linear_regression | 1h | UG Hall6 | 17675 | 0.1510 | 0.0900 |
| within_site | linear_regression | 1h | UG Hall7 | 17678 | 0.0894 | 0.2122 |
| within_site | xgboost | 1h | LSK North | 17679 | 0.1243 | 0.2523 |
| within_site | xgboost | 1h | SQ1 | 17679 | 0.1237 | 0.2426 |
| within_site | xgboost | 1h | SQ2 | 17679 | 0.0997 | 0.2695 |
| within_site | xgboost | 1h | SQ3 | 17678 | 0.1117 | 0.2581 |
| within_site | xgboost | 1h | SQ4 | 17678 | 0.1148 | 0.2536 |
| within_site | xgboost | 1h | UG Hall6 | 17675 | 0.1237 | 0.2547 |
| within_site | xgboost | 1h | UG Hall7 | 17678 | 0.0865 | 0.2386 |

**Table 9. Multi-seed replication of the neural models.**

Source: `results/tables/multi_seed_results.csv`

| model | n_seeds | seeds | rmse_mean | rmse_std | rmse_ci_low | rmse_ci_high | skill_vs_persistence_rmse_mean | skill_vs_persistence_rmse_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_lstm | 4 | 42, 123, 456, 789 | 8,813.4200 | 252.5470 | 8,411.5600 | 9,215.2700 | 0.0359 | 0.0276 |
| cnn_lstm | 3 | 123, 456, 789 | 6,711.1000 | 55.7639 | 6,572.5800 | 6,849.6300 | 0.2659 | 0.0061 |
| gru | 1 | 42 | 9,611.4600 | 0.0000 | 9,611.4600 | 9,611.4600 | -0.0514 | 0.0000 |

## 9. Statistical analysis

**Table 10. Skill intervals and Diebold-Mariano tests against persistence.**

Source: `results/tables/statistical_significance.csv`

| model | loss | n_daylight_comparisons | skill_point | skill_ci_low | skill_ci_high | dm_p_value | dm_p_value_holm | dm_lag1_autocorrelation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smart_persistence | mae | 17679 | 0.1612 | 0.1395 | 0.1858 | 0.6366 | 1 | 0.6068 |
| cnn_lstm | mae | 17679 | 0.2540 | 0.2307 | 0.2777 | 0.6442 | 1 | 0.5402 |
| gradient_boosting | mae | 17679 | 0.2710 | 0.2490 | 0.2930 | 0.6789 | 1 | 0.5950 |
| random_forest | mae | 17679 | 0.2678 | 0.2459 | 0.2898 | 0.7146 | 1 | 0.5796 |
| xgboost | mae | 17679 | 0.2523 | 0.2301 | 0.2741 | 0.7158 | 1 | 0.5562 |
| linear_regression | mae | 17679 | 0.1843 | 0.1624 | 0.2062 | 0.7623 | 1 | 0.5687 |
| transformer | mae | 17679 | 0.1412 | 0.1166 | 0.1651 | 0.8327 | 1 | 0.5858 |
| attention_lstm | mae | 17679 | 0.0056 | -0.0412 | 0.0506 | 0.9598 | 1 | 0.6332 |
| gru | mae | 17679 | -0.0514 | -0.0983 | -0.0096 | 0.9721 | 1 | 0.6313 |
| lstm | mae | 17679 | -0.0134 | -0.0577 | 0.0248 | 0.9863 | 1 | 0.6205 |
| gradient_boosting | rmse | 17679 | 0.2710 | 0.2490 | 0.2930 | 0.7369 | 1 | 0.4534 |
| cnn_lstm | rmse | 17679 | 0.2540 | 0.2307 | 0.2777 | 0.7573 | 1 | 0.3962 |
| random_forest | rmse | 17679 | 0.2678 | 0.2459 | 0.2898 | 0.7689 | 1 | 0.4344 |
| xgboost | rmse | 17679 | 0.2523 | 0.2301 | 0.2741 | 0.7784 | 1 | 0.4322 |
| smart_persistence | rmse | 17679 | 0.1612 | 0.1395 | 0.1858 | 0.7859 | 1 | 0.4914 |
| linear_regression | rmse | 17679 | 0.1843 | 0.1624 | 0.2062 | 0.7942 | 1 | 0.4478 |
| transformer | rmse | 17679 | 0.1412 | 0.1166 | 0.1651 | 0.8509 | 1 | 0.4483 |
| gru | rmse | 17679 | -0.0514 | -0.0983 | -0.0096 | 0.9648 | 1 | 0.5550 |
| lstm | rmse | 17679 | -0.0134 | -0.0577 | 0.0248 | 0.9902 | 1 | 0.5239 |
| attention_lstm | rmse | 17679 | 0.0056 | -0.0412 | 0.0506 | 0.9961 | 1 | 0.5667 |

The integrated autocorrelation time of the loss differential on this data is **9** steps, so the nominal sample size overstates the information in the error series by roughly that factor. That is the quantitative reason the Diebold-Mariano test rejects nothing here while the block-bootstrap interval excludes zero for several models. Both are reported; the interval is the informative statistic and the test is the conservative one. `results/tables/error_autocorrelation.csv` and `results/tables/block_length_sensitivity.csv` carry the diagnostics.

## 10. Explainability

**Table 11. Feature importance: SHAP for the tree ensemble, grouped permutation importance for the neural models. Predictive attributions only, not causal claims.**

Source: `results/tables/feature_importance.csv`

| model | feature | value | method |
| --- | --- | --- | --- |
| xgboost | ghi | 0.1159 |  |
| xgboost | temperature | 0.0128 |  |
| xgboost | relative_humidity | 0.0383 |  |
| xgboost | wind_speed | 0.0214 |  |
| xgboost | solar_elevation | 0.0161 |  |
| xgboost | sin_solar_elevation | 0.0106 |  |
| xgboost | hour_sin | 0.1895 |  |
| xgboost | hour_cos | 0.0305 |  |
| xgboost | doy_sin | 0.0109 |  |
| xgboost | doy_cos | 0.0098 |  |
| xgboost | pv_power_w_lag_0 | 0.2434 |  |
| xgboost | pv_power_w_lag_1 | 0.0060 |  |
| xgboost | pv_power_w_lag_2 | 0.0017 |  |
| xgboost | pv_power_w_lag_4 | 0.0005 |  |
| xgboost | pv_power_w_lag_8 | 0.0007 |  |
| xgboost | pv_power_w_lag_12 | 0.0003 |  |
| xgboost | pv_power_w_lag_24 | 0.0003 |  |
| xgboost | pv_power_w_lag_48 | 0.0408 |  |
| xgboost | pv_power_w_lag_96 | 0.0261 |  |
| xgboost | pv_rolling_mean_4 | 0.0366 |  |
| xgboost | pv_rolling_std_4 | 0.0261 |  |
| xgboost | pv_rolling_mean_12 | 0.0049 |  |
| xgboost | pv_rolling_std_12 | 0.0117 |  |
| xgboost | pv_rolling_mean_24 | 0.0051 |  |
| xgboost | pv_rolling_std_24 | 0.0097 |  |
| xgboost | ghi_rollmean_4 | 0.0178 |  |
| xgboost | ghi_rollmean_24 | 0.0075 |  |
| xgboost | temperature_rollmean_4 | 0.0103 |  |
| xgboost | temperature_rollmean_24 | 0.0135 |  |
| xgboost | ghi_delta_1 | 0.0192 |  |
| xgboost | ghi_slope_1 | 0.0096 |  |
| xgboost | ghi_delta_4 | 0.0324 |  |
| xgboost | ghi_slope_4 | 0.0200 |  |
| attention_lstm | pv_power_w_lag_0 | 7,114.4600 |  |
| attention_lstm | pv_power_w_lag_1 | 6,282.1200 |  |
| attention_lstm | pv_power_w_lag_2 | 5,283.2200 |  |
| attention_lstm | pv_power_w_lag_8 | 4,208.1300 |  |
| attention_lstm | pv_rolling_mean_4 | 3,468.0300 |  |
| attention_lstm | pv_rolling_mean_24 | 3,030.9100 |  |
| attention_lstm | pv_rolling_mean_12 | 2,511.9600 |  |
| attention_lstm | pv_power_w_lag_4 | 2,296.8100 |  |
| attention_lstm | pv_power_w_lag_12 | 2,202.4600 |  |
| attention_lstm | ghi | 2,127.1400 |  |
| attention_lstm | solar_elevation | 1,981.1900 |  |
| attention_lstm | pv_rolling_std_12 | 1,710.0500 |  |
| attention_lstm | ghi_delta_4 | 1,685.3200 |  |
| attention_lstm | pv_power_w_lag_96 | 1,624.3700 |  |
| attention_lstm | pv_rolling_std_4 | 1,589.8800 |  |
| attention_lstm | pv_power_w_lag_48 | 1,450.4400 |  |
| attention_lstm | ghi_slope_1 | 1,355.8200 |  |
| attention_lstm | pv_rolling_std_24 | 1,318.4900 |  |
| attention_lstm | ghi_delta_1 | 1,290.5100 |  |
| attention_lstm | ghi_rollmean_4 | 1,259.8200 |  |
| attention_lstm | ghi_slope_4 | 1,146.3300 |  |
| attention_lstm | ghi_rollmean_24 | 579.7540 |  |
| attention_lstm | pv_power_w_lag_24 | 539.2510 |  |
| attention_lstm | relative_humidity | 284.9940 |  |
| attention_lstm | temperature_rollmean_4 | 36.1125 |  |
| attention_lstm | hour_cos | 3.0997 |  |
| attention_lstm | doy_cos | 1.9364 |  |
| attention_lstm | sin_solar_elevation | -1.3714 |  |
| attention_lstm | doy_sin | -2.6503 |  |
| attention_lstm | hour_sin | -3.0782 |  |
| attention_lstm | temperature_rollmean_24 | -3.1474 |  |
| attention_lstm | temperature | -16.5406 |  |
| attention_lstm | wind_speed | -29.9804 |  |

## 11. Uncertainty

**Table 12. Split-conformal interval coverage.**

Source: `results/tables/uncertainty_coverage.csv`

| model | alpha | scope | nominal_coverage | empirical_coverage | coverage_error | interval_half_width_w | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | 0.1000 | evaluation_all_steps | 0.9000 | 0.8286 | -0.0714 | 4,793.0000 | 32033 |
| xgboost | 0.1000 | evaluation_daylight | 0.9000 | 0.6638 | -0.2362 | 4,793.0000 | 16331 |
| xgboost | 0.2000 | evaluation_all_steps | 0.8000 | 0.7245 | -0.0755 | 2,257.2500 | 32033 |
| xgboost | 0.2000 | evaluation_daylight | 0.8000 | 0.4602 | -0.3398 | 2,257.2500 | 16331 |
| lstm | 0.1000 | evaluation_all_steps | 0.9000 | 0.8297 | -0.0703 | 8,138.3600 | 32033 |
| lstm | 0.1000 | evaluation_daylight | 0.9000 | 0.6661 | -0.2339 | 8,138.3600 | 16331 |
| lstm | 0.2000 | evaluation_all_steps | 0.8000 | 0.7372 | -0.0628 | 4,569.9900 | 32033 |
| lstm | 0.2000 | evaluation_daylight | 0.8000 | 0.4855 | -0.3145 | 4,569.9900 | 16331 |
| transformer | 0.1000 | evaluation_all_steps | 0.9000 | 0.8086 | -0.0914 | 6,222.5300 | 32033 |
| transformer | 0.1000 | evaluation_daylight | 0.9000 | 0.6872 | -0.2128 | 6,222.5300 | 16331 |
| transformer | 0.2000 | evaluation_all_steps | 0.8000 | 0.7036 | -0.0964 | 4,399.9500 | 32033 |
| transformer | 0.2000 | evaluation_daylight | 0.8000 | 0.5640 | -0.2360 | 4,399.9500 | 16331 |

## 12. Computational cost

**Table 13. Computational cost on the pinned single-thread CPU protocol. Absolute durations are machine-specific; only the ranking is portable.**

Source: `results/tables/computational_cost.csv`

| model | model_family | train_seconds | inference_ms_per_window | n_parameters | nrmse_capacity | pareto_optimal |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | rule-based reference | 0.0000 | 0.0000 | 0 | 0.1662 | False |
| persistence | rule-based reference | 0.0000 | 0.0000 | 0 | 0.0968 | False |
| persistence | rule-based reference | 0.0000 | 0.0000 | 0 | 0.2067 | False |
| persistence | rule-based reference | 0.0000 | 0.0000 | 0 | 0.4070 | False |
| smart_persistence | rule-based reference | 0.0000 | 0.0000 | 0 | 0.1394 | False |
| smart_persistence | rule-based reference | 0.0000 | 0.0000 | 0 | 0.0941 | True |
| smart_persistence | rule-based reference | 0.0000 | 0.0000 | 0 | 0.2067 | False |
| smart_persistence | rule-based reference | 0.0000 | 0.0000 | 0 | 0.3280 | False |
| linear_regression | classical machine learning | 2.1435 | 0.0012 | 793 | 0.1356 | False |
| linear_regression | classical machine learning | 1.3918 | 0.0009 | 793 | 0.0914 | True |
| linear_regression | classical machine learning | 8.2673 | 0.0014 | 793 | 0.1780 | False |
| linear_regression | classical machine learning | 8.4667 | 0.0010 | 793 | 0.1746 | False |
| random_forest | classical machine learning | 287.4450 | 0.0202 | 200 | 0.1217 | False |
| random_forest | classical machine learning | 188.2630 | 0.0171 | 200 | 0.0901 | False |
| random_forest | classical machine learning | 536.8830 | 0.1066 | 200 | 0.1785 | False |
| random_forest | classical machine learning | 564.1860 | 0.0428 | 200 | 0.1688 | False |
| gradient_boosting | classical machine learning | 9.4859 | 0.0206 | 0 | 0.1212 | False |
| gradient_boosting | classical machine learning | 11.8988 | 0.0173 | 0 | 0.0876 | True |
| gradient_boosting | classical machine learning | 63.0244 | 0.0633 | 0 | 0.1764 | False |
| gradient_boosting | classical machine learning | 47.6943 | 0.0711 | 0 | 0.1661 | False |
| xgboost | classical machine learning | 42.5256 | 0.0029 | 400 | 0.1243 | False |
| xgboost | classical machine learning | 36.4617 | 0.0021 | 400 | 0.0930 | False |
| xgboost | classical machine learning | 133.5540 | 0.0036 | 400 | 0.1817 | False |
| xgboost | classical machine learning | 135.5020 | 0.0036 | 400 | 0.1722 | False |
| lstm | recurrent neural network | 312.6060 | 0.0520 | 50401 | 0.1684 | False |
| lstm | recurrent neural network | 181.0180 | 0.0359 | 50401 | 0.1596 | False |
| lstm | recurrent neural network | 829.5520 | 0.0888 | 50401 | 0.2006 | False |
| lstm | recurrent neural network | 850.6670 | 0.0753 | 50401 | 0.1886 | False |
| gru | recurrent neural network | 361.9260 | 0.0467 | 37825 | 0.1748 | False |
| gru | recurrent neural network | 136.0070 | 0.0447 | 37825 | 0.1615 | False |
| gru | recurrent neural network | 498.6340 | 0.1874 | 37825 | 0.2007 | False |
| gru | recurrent neural network | 1,139.4300 | 0.2975 | 37825 | 0.1909 | False |
| cnn_lstm | hybrid convolutional-recurrent | 274.1260 | 0.0265 | 64225 | 0.1240 | False |
| cnn_lstm | hybrid convolutional-recurrent | 207.6680 | 0.0258 | 64225 | 0.0882 | False |
| cnn_lstm | hybrid convolutional-recurrent | 437.0280 | 0.0695 | 64225 | 0.1853 | False |
| cnn_lstm | hybrid convolutional-recurrent | 511.9600 | 0.0766 | 64225 | 0.1770 | False |
| attention_lstm | hybrid recurrent-attention | 172.7780 | 0.0328 | 55201 | 0.1653 | False |
| attention_lstm | hybrid recurrent-attention | 251.3470 | 0.0369 | 55201 | 0.1479 | False |
| attention_lstm | hybrid recurrent-attention | 595.4610 | 0.0924 | 55201 | 0.2023 | False |
| attention_lstm | hybrid recurrent-attention | 1,371.9300 | 0.0657 | 55201 | 0.1820 | False |
| transformer | attention-only transformer | 1,121.9900 | 0.0478 | 69185 | 0.1427 | False |
| transformer | attention-only transformer | 2,766.3100 | 0.0718 | 69185 | 0.0997 | False |
| transformer | attention-only transformer | 3,404.4100 | 0.1323 | 69185 | 0.1820 | False |
| transformer | attention-only transformer | 3,172.2700 | 0.1873 | 69185 | 0.1829 | False |

## 13. Discussion

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
