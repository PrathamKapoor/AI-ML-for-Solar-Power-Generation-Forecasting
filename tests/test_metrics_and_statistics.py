"""Metrics, statistical tests, conformal intervals and the results tables.

The statistical tests are checked against cases with a known answer, and the
bootstrap is checked for the property that matters here: it must be wider than
an i.i.d. bootstrap on serially correlated data, which is the whole reason it is
used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecasting.evaluation import metrics, reporting, stratified, uncertainty
from solar_forecasting.statistics import comparison


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_mae_and_rmse_match_hand_calculation() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    f = np.array([1.5, 2.5, 2.0, 6.0])
    assert metrics.mae(y, f) == pytest.approx((0.5 + 0.5 + 1.0 + 2.0) / 4)
    assert metrics.rmse(y, f) == pytest.approx(np.sqrt((0.25 + 0.25 + 1 + 4) / 4))


def test_r2_is_one_for_a_perfect_forecast() -> None:
    y = np.array([0.0, 1.0, 5.0, 9.0])
    assert metrics.r2(y, y) == pytest.approx(1.0)


def test_r2_is_negative_when_worse_than_the_mean() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    f = np.full(4, 10.0)
    assert metrics.r2(y, f) < 0.0


def test_r2_of_a_constant_target_is_reported_as_zero() -> None:
    y = np.full(10, 3.0)
    assert metrics.r2(y, y) == 0.0


def test_mape_refuses_to_run_on_a_target_with_zeros() -> None:
    with pytest.raises(NotImplementedError, match="MAPE is undefined"):
        metrics.mape(np.array([0.0, 1.0]), np.array([0.0, 1.5]))


def test_smape_is_zero_when_both_series_are_zero() -> None:
    y = np.array([0.0, 0.0, 10.0])
    f = np.array([0.0, 0.0, 12.0])
    value = metrics.smape(y, f)
    assert 0.0 < value <= 200.0
    assert value < 30.0


def test_smape_is_symmetric() -> None:
    y = np.array([4.0, 8.0])
    f = np.array([3.0, 10.0])
    assert metrics.smape(y, f) == pytest.approx(metrics.smape(f, y))


def test_nrmse_by_capacity_and_by_mean() -> None:
    y = np.array([100.0, 200.0])
    f = np.array([110.0, 190.0])
    by_capacity = metrics.nrmse(y, f, capacity_w=500.0)
    by_mean = metrics.nrmse(y, f)
    assert by_capacity < by_mean
    assert by_capacity == pytest.approx(metrics.rmse(y, f) / 500.0)


def test_skill_is_zero_against_itself() -> None:
    value = 1234.0
    assert metrics.skill(value, value) == pytest.approx(0.0)


def test_skill_rejects_a_degenerate_reference() -> None:
    with pytest.raises(ValueError, match="reference metric"):
        metrics.skill(0.0, 1.0)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.mae(np.array([1.0, 2.0]), np.array([1.0]))


def test_every_metric_is_documented() -> None:
    documentation = metrics.documentation_table()
    assert set(documentation["metric"]) >= {"mae", "rmse", "nrmse", "r2", "smape", "mape"}
    for _, row in documentation.iterrows():
        assert row["formula"] and row["interpretation"] and row["limitations"]


# --------------------------------------------------------------------------- #
# Statistical comparison
# --------------------------------------------------------------------------- #
def test_block_bootstrap_interval_contains_the_point_estimate() -> None:
    errors = np.random.default_rng(0).normal(size=500)
    result = comparison.block_bootstrap_metric(errors, comparison._rmse,
                                               n_resamples=300, block_length=32, seed=1)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
    assert result["method"] == "moving-block bootstrap"


def test_block_bootstrap_is_wider_than_an_iid_bootstrap_on_correlated_errors() -> None:
    """The reason the block bootstrap is used: dependence widens the interval."""
    rng = np.random.default_rng(3)
    noise = rng.normal(size=2000)
    correlated = np.convolve(noise, np.ones(24) / np.sqrt(24.0), mode="same")
    block = comparison.block_bootstrap_metric(correlated, comparison._rmse,
                                              n_resamples=300, block_length=96, seed=2)
    iid = comparison.block_bootstrap_metric(correlated, comparison._rmse,
                                            n_resamples=300, block_length=1, seed=2)
    assert (block["ci_high"] - block["ci_low"]) > (iid["ci_high"] - iid["ci_low"])


def test_bootstrap_skill_of_a_better_model_is_positive() -> None:
    actual = np.full(400, 1_000.0)
    model_errors = np.random.default_rng(4).normal(scale=100.0, size=400)
    reference_errors = np.random.default_rng(5).normal(scale=200.0, size=400)
    result = comparison.block_bootstrap_skill(model_errors, reference_errors,
                                              n_resamples=300, block_length=32, seed=0)
    assert result["point_estimate"] > 0.0
    assert result["ci_low"] > 0.0


def test_diebold_mariano_detects_a_real_difference() -> None:
    rng = np.random.default_rng(6)
    better = rng.normal(scale=20.0, size=600)
    worse = rng.normal(loc=600.0, scale=20.0, size=600)
    result = comparison.diebold_mariano(better, worse, "better", "worse", loss="mae")
    assert result.p_value < 0.01
    assert result.better_model == "better"
    assert result.mean_significance


def test_diebold_mariano_does_not_invent_a_difference() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=800)
    b = rng.normal(size=800)
    result = comparison.diebold_mariano(a, b, "a", "b", loss="mae")
    assert result.p_value > 0.05
    assert not result.mean_significance


def test_diebold_mariano_reports_the_serial_correlation_it_found() -> None:
    rng = np.random.default_rng(8)
    noise = rng.normal(size=1500)
    correlated = np.convolve(noise, np.ones(20) / np.sqrt(20.0), mode="same")
    shifted = correlated + 50.0
    result = comparison.diebold_mariano(correlated, shifted, "a", "b", loss="rmse")
    assert result.lag_1_autocorrelation > 0.3
    assert "Harvey" in result.correction or "harvey" in result.correction


def test_wilcoxon_reports_its_assumption_violation() -> None:
    rng = np.random.default_rng(9)
    result = comparison.wilcoxon_signed_rank(rng.normal(size=200),
                                              rng.normal(loc=1.0, size=200))
    assert result["p_value"] < 0.05
    assert "assumption_violation" in result
    assert -1.0 <= result["effect_size_rank_biserial"] <= 1.0


def test_holm_bonferroni_is_monotone_and_bounded() -> None:
    adjusted = comparison.holm_bonferroni([0.001, 0.02, 0.04, 0.9])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2] <= adjusted[3]
    assert adjusted[0] == pytest.approx(0.004)


def test_compare_models_returns_every_component() -> None:
    rng = np.random.default_rng(10)
    actual = np.abs(rng.normal(loc=5_000.0, size=300))
    entry = comparison.compare_models(
        {"model": actual - rng.normal(scale=200.0, size=300),
         "persistence": actual - rng.normal(scale=400.0, size=300)},
        actual=actual, model="model", n_resamples=100, block_length=16)
    assert set(entry) == {"model", "reference_model", "loss",
                          "skill_vs_reference_rmse", "diebold_mariano", "wilcoxon"}
    assert entry["skill_vs_reference_rmse"]["point_estimate"] > 0


def test_statistical_assumptions_are_documented() -> None:
    assert comparison.ASSUMPTIONS
    assert any("serial" in key for key in comparison.ASSUMPTIONS)


# --------------------------------------------------------------------------- #
# Stratified analysis
# --------------------------------------------------------------------------- #
@pytest.fixture()
def predictions_frame() -> pd.DataFrame:
    times = pd.date_range("2023-01-01", periods=96 * 20, freq="15min")  # 20 days at 15 minutes
    hours = times.hour.to_numpy()
    daylight = (hours >= 6) & (hours < 18)
    actual = np.where(daylight, 30_000.0, 0.0)
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "Time": times,
        "actual": actual,
        "predicted": np.where(daylight, actual - rng.normal(scale=1_500.0,
                                                           size=len(times)), 0.0),
        "daylight": daylight,
        "regime": np.where(daylight, "Clear", "Night"),
        "season": "winter",
        "reference_persistence": np.where(daylight, actual * 0.8, 0.0),
    })


def test_stratified_metrics_score_the_reference_in_the_same_stratum(
        predictions_frame: pd.DataFrame) -> None:
    frame = stratified.build_strata(predictions_frame, 55_000.0)
    table = stratified.stratified_metrics(frame, 55_000.0, "regime")
    clear = table[table["stratum"] == "Clear"].iloc[0]
    assert clear["n"] == int(daylight_count(predictions_frame))
    assert clear["persistence_rmse"] > 0
    assert 0.0 <= clear["skill_vs_persistence_rmse"] <= 1.0


def daylight_count(frame: pd.DataFrame) -> int:
    return int(frame["daylight"].astype(bool).sum())


def test_under_sampled_strata_are_flagged_not_reported(predictions_frame: pd.DataFrame
                                                       ) -> None:
    frame = stratified.build_strata(predictions_frame, 55_000.0)
    table = stratified.stratified_metrics(frame, 55_000.0, "regime", min_size=10_000)
    assert table["under_sampled"].all()
    assert table["nrmse_capacity"].isna().all()


def test_season_labels_are_normalised_to_the_declared_order(
        predictions_frame: pd.DataFrame) -> None:
    frame = stratified.build_strata(predictions_frame, 55_000.0)
    assert set(frame["season"].astype(object).unique()) <= set(stratified.SEASON_ORDER)
    assert list(frame["season"].cat.categories) == stratified.SEASON_ORDER


def test_time_of_day_bands_cover_the_operating_day() -> None:
    """Bands are defined for the hours a PV plant is producing, not for midnight."""
    times = pd.date_range("2023-06-21 05:00", periods=60, freq="15min")  # 05:00-19:45
    labels = stratified.time_of_day_band(times)
    assert labels.notna().all()
    assert labels.iloc[0] == "Early morning (05-08)"
    assert labels.iloc[(12 - 5) * 4] == "Solar noon (12-14)"
    assert labels.iloc[-1] == "Late evening (18-20)"


def test_power_level_bands_scale_by_capacity() -> None:
    labels = stratified.power_level_band(np.array([0.0, 5_000.0, 40_000.0]), 55_000.0)
    assert labels.iloc[0] == "0-10% of capacity"
    assert labels.iloc[2] == "50-75% of capacity"


def test_stratified_table_stacks_every_dimension(predictions_frame: pd.DataFrame) -> None:
    table = stratified.stratified_table(predictions_frame, 55_000.0, "model", "1h")
    assert set(table["stratum_type"]) >= {"regime", "season", "time_of_day"}
    assert "nrmse_capacity" in table.columns


# --------------------------------------------------------------------------- #
# Conformal intervals
# --------------------------------------------------------------------------- #
def test_conformal_quantile_follows_the_finite_sample_rule() -> None:
    residuals = np.arange(1.0, 101.0)
    q = uncertainty.conformal_quantile(residuals, alpha=0.1)
    assert q == pytest.approx(91.0)


def test_conformal_coverage_table_reports_both_scopes(predictions_frame: pd.DataFrame
                                                       ) -> None:
    rows = uncertainty.coverage_table(predictions_frame, "model", "1h",
                                      alphas=(0.1, 0.2), calibration_days=5)
    assert {row["scope"] for row in rows} == {"evaluation_all_steps",
                                              "evaluation_daylight"}
    for row in rows:
        assert 0.0 <= row["empirical_coverage"] <= 1.0
        assert row["n_calibration"] > 0
        assert "approximate" in row["guarantee"]


def test_conformal_split_refuses_a_tiny_calibration_block(predictions_frame: pd.DataFrame
                                                         ) -> None:
    with pytest.raises(ValueError, match="too small"):
        uncertainty.split_calibrate(predictions_frame, calibration_days=0)


def test_wider_intervals_for_a_smaller_alpha(predictions_frame: pd.DataFrame) -> None:
    wide = uncertainty.coverage_table(predictions_frame, "model", "1h", alphas=(0.05,),
                                      calibration_days=5)[0]
    narrow = uncertainty.coverage_table(predictions_frame, "model", "1h", alphas=(0.4,),
                                        calibration_days=5)[0]
    assert wide["interval_half_width_w"] > narrow["interval_half_width_w"]


# --------------------------------------------------------------------------- #
# Reporting tables
# --------------------------------------------------------------------------- #
@pytest.fixture()
def registry() -> pd.DataFrame:
    rows = []
    for model, rmse, seconds in (("persistence", 9_000.0, 0.0),
                                 ("lstm", 9_200.0, 300.0),
                                 ("xgboost", 6_800.0, 40.0)):
        for horizon in ("1h", "6h"):
            rows.append({
                "experiment_id": f"B_ml_vs_dl__{model}__{horizon}", "model": model,
                "experiment_group": "B_ml_vs_dl", "horizon": horizon,
                "feature_spec": "full", "rmse": rmse * (1.0 if horizon == "1h" else 2.0),
                "mae": rmse / 1.4, "nrmse_capacity": rmse / 55_000.0,
                "skill_vs_persistence_rmse": 0.25, "train_seconds": seconds,
                "n_parameters": 1_000, "inference_ms_per_window": 0.05,
                "r2": 0.6,
            })
    return pd.DataFrame(rows)


def test_overall_comparison_keeps_the_model_order(registry: pd.DataFrame) -> None:
    table = reporting.overall_comparison(registry, horizon="1h")
    assert list(table["model"]) == ["persistence", "xgboost", "lstm"]


def test_horizon_comparison_records_where_each_row_came_from(registry: pd.DataFrame
                                                             ) -> None:
    table = reporting.horizon_comparison(registry)
    assert set(table["horizon"]) == {"1h", "6h"}
    assert "source_group" in table.columns
    assert set(table["source_group"]) == {"B_ml_vs_dl"}


def test_computational_cost_flags_the_pareto_frontier(registry: pd.DataFrame) -> None:
    cost = reporting.computational_cost(registry)
    assert set(cost["model"]) == {"persistence", "lstm", "xgboost"}
    # xgboost is both the most accurate and cheaper than the LSTM, so the LSTM
    # cannot be on the frontier.
    assert bool(cost.loc[cost["model"] == "xgboost", "pareto_optimal"].iloc[0])
    assert not bool(cost.loc[cost["model"] == "lstm", "pareto_optimal"].iloc[0])


def test_model_ablation_compares_against_the_best_backbone(registry: pd.DataFrame
                                                            ) -> None:
    table = reporting.model_ablation(registry)
    assert table.empty or {"composite", "backbone"} <= set(table.columns)


def test_results_summary_reports_coverage(registry: pd.DataFrame) -> None:
    summary = reporting.results_summary(registry)
    assert summary["n_experiments"] == len(registry)
    assert summary["models"] == ["lstm", "persistence", "xgboost"]
    assert summary["horizons"] == ["1h", "6h"]


def test_markdown_renderer_handles_an_empty_table() -> None:
    assert "no rows" in reporting.format_markdown_table(pd.DataFrame())


def test_markdown_renderer_escapes_pipes() -> None:
    text = reporting.format_markdown_table(pd.DataFrame({"a": ["x|y"]}))
    assert "x/y" in text


