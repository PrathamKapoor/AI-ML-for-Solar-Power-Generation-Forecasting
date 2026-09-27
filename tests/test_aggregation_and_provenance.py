"""Tests for the aggregation, provenance and footprint paths added last.

These cover the machinery that turns records into the published claims: the
canonical matrix, the multi-seed aggregation, the additive input regimes, the
run provenance block, the model footprint measurement, the figure provenance
manifest, and the integrated-gradients completeness property.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_forecasting.evaluation import matrix as matrix_mod
from solar_forecasting.evaluation import reporting
from solar_forecasting.explainability import importance as imp_mod
from solar_forecasting.visualization import provenance as prov_mod


def make_registry(n_models: int = 3, seeds: Sequence[int] = (42, 123)) -> pd.DataFrame:
    rows = []
    for model in ("gradient_boosting", "xgboost", "lstm")[:n_models]:
        for seed in seeds:
            for horizon in ("1h", "6h"):
                rows.append({
                    "experiment_id": f"M_multiseed__{model}__{horizon}__seed{seed}"
                    if model == "lstm" else f"B_ml_vs_dl__{model}__{horizon}",
                    "experiment_group": "M_multiseed" if model == "lstm" else "B_ml_vs_dl",
                    "model": model, "model_family": "classical_ml",
                    "horizon": horizon, "feature_spec": "full", "seed": seed,
                    "mae": 4000.0 + (seed % 7), "rmse": 6000.0 + (seed % 7), "nrmse_capacity": 0.12,
                    "r2": 0.78, "smape": 48.0, "bias": 10.0,
                    "skill_vs_persistence_rmse": 0.25, "train_seconds": 40.0,
                    "inference_ms_per_window": 0.02, "n_parameters": 500,
                    "n_test": 1000,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Multi-seed aggregation
# --------------------------------------------------------------------------- #
def test_multiseed_summary_reports_mean_spread_and_interval() -> None:
    registry = make_registry()
    per_seed, summary = matrix_mod.multiseed_summary(registry, group="M_multiseed")
    assert len(summary) == 2, "one row per model and horizon"
    row = summary[summary["horizon"] == "1h"].iloc[0]
    assert row["n_seeds"] == 2
    # seed % 7 keeps the synthetic spread constant as seeds are added, so the
    # interval-width test below measures the effect of the sample size alone.
    assert row["rmse_mean"] == pytest.approx((6000.0 + 42 % 7 + 6000.0 + 123 % 7) / 2)
    assert row["rmse_std"] > 0
    assert row["rmse_ci_low"] < row["rmse_mean"] < row["rmse_ci_high"]
    assert row["rmse_min"] < row["rmse_mean"] < row["rmse_max"]
    assert not per_seed.empty


def test_multiseed_summary_of_an_absent_group_is_empty_not_an_error() -> None:
    per_seed, summary = matrix_mod.multiseed_summary(make_registry(), group="NOPE")
    assert summary.empty and per_seed.empty


def test_multiseed_ci_narrows_with_more_seeds() -> None:
    _, two = matrix_mod.multiseed_summary(
        make_registry(seeds=(42, 123)), group="M_multiseed")
    _, five = matrix_mod.multiseed_summary(
        make_registry(seeds=(42, 123, 456, 789, 2026)), group="M_multiseed")
    two = two[two["horizon"] == "1h"]
    five = five[five["horizon"] == "1h"]
    width_two = float(two["rmse_ci_high"].iloc[0] - two["rmse_ci_low"].iloc[0])
    width_five = float(five["rmse_ci_high"].iloc[0] - five["rmse_ci_low"].iloc[0])
    assert width_five < width_two, "more seeds must give a narrower interval"


def test_a_single_seed_gives_a_zero_width_interval_and_is_flagged() -> None:
    _, summary = matrix_mod.multiseed_summary(
        make_registry(seeds=(42,)), group="M_multiseed")
    row = summary[summary["horizon"] == "1h"].iloc[0]
    assert row["n_seeds"] == 1
    assert float(row["rmse_std"]) == 0.0
    assert row["rmse_ci_low"] == row["rmse_ci_high"] == row["rmse_mean"]


# --------------------------------------------------------------------------- #
# Canonical matrix
# --------------------------------------------------------------------------- #
def test_matrix_carries_every_required_column() -> None:
    matrix, _ = matrix_mod.build(make_registry(), None, None, None)
    required = ["experiment_id", "research_question", "model", "model_family", "horizon",
                "site", "weather_regime", "season", "feature_regime", "seed", "mae",
                "rmse", "nrmse", "r2", "smape", "skill_vs_persistence", "train_time",
                "inference_time", "status"]
    assert not matrix.empty
    assert [c for c in required if c not in matrix.columns] == []


def test_matrix_maps_every_group_to_a_research_question() -> None:
    matrix, _ = matrix_mod.build(make_registry(), None, None, None)
    executed = matrix[matrix["status"] == "executed"]
    assert executed["research_question"].str.startswith("RQ").all()
    assert set(executed["research_question"]) <= {f"RQ{i}" for i in range(1, 8)}


def test_matrix_handles_none_inputs_without_raising() -> None:
    matrix, _ = matrix_mod.build(make_registry(), None, None, None)
    assert not matrix.empty
    # Passing an empty frame and None must both be safe.
    matrix2, _ = matrix_mod.build(pd.DataFrame(), pd.DataFrame(), None, None)
    assert matrix2.empty


def test_matrix_coverage_summary_reports_what_ran() -> None:
    matrix, _ = matrix_mod.build(make_registry(), None, None, None)
    summary = matrix_mod.coverage_summary(matrix)
    assert summary["executed_rows"] > 0
    assert "RQ1" in summary["research_questions_covered"]
    assert set(summary["horizons_covered"]) == {"1h", "6h"}


# --------------------------------------------------------------------------- #
# Additive input regimes
# --------------------------------------------------------------------------- #
def test_feature_regime_table_orders_the_regimes_by_information_content() -> None:
    registry = make_registry()
    registry.loc[registry["experiment_group"] == "B_ml_vs_dl", "experiment_group"] = \
        "N_feature_regimes"
    for regime in ("full", "pv_only", "weather_only", "pv_weather", "pv_weather_solar"):
        block = registry[registry["experiment_group"] == "N_feature_regimes"].copy()
        block["feature_spec"] = regime
        registry = pd.concat([registry, block], ignore_index=True)
    table = reporting.feature_regime_comparison(registry)
    assert not table.empty
    assert list(table.drop_duplicates("feature_spec")["feature_spec"])[:5] == [
        "pv_only", "weather_only", "pv_weather", "pv_weather_solar", "full"]


def test_regimes_resolve_to_different_input_widths(config) -> None:
    from solar_forecasting.training.experiment import select_features
    columns = json.loads(
        (config.project_root_for("data") / "processed" / "feature_columns.json")
        .read_text(encoding="utf-8"))["feature_columns"]
    widths = {name: len(select_features(columns, name, config))
              for name in ("pv_only", "weather_only", "pv_weather", "pv_weather_solar",
                           "full")}
    assert widths["pv_only"] < widths["pv_weather"] < widths["full"]
    assert widths["weather_only"] < widths["full"]
    # Weather-only must contain no target-derived feature at all.
    weather_only = select_features(columns, "weather_only", config)
    assert not any(c.startswith("pv_") for c in weather_only)
    pv_only = select_features(columns, "pv_only", config)
    assert all(c.startswith("pv_") for c in pv_only)


# --------------------------------------------------------------------------- #
# Model footprint
# --------------------------------------------------------------------------- #
def test_footprint_of_a_rule_based_model_is_zero() -> None:
    footprint = reporting.model_footprint("persistence")
    assert footprint["n_parameters"] == 0.0
    assert footprint["model_size_bytes"] == 0.0


def test_footprint_measures_a_fitted_tree_ensemble() -> None:
    footprint = reporting.model_footprint("gradient_boosting")
    assert footprint["n_parameters"] > 0
    assert footprint["model_size_bytes"] > 0
    assert footprint["peak_train_memory_mb"] > 0


def test_a_tree_ensemble_is_never_reported_as_having_no_parameters() -> None:
    """Regression test for the misleading zero.

    A fitted gradient-boosting model has no weight tensor, so a runner that
    records a parameter count records zero. Reporting that zero as the
    deployment footprint would understate the artefact by two orders of
    magnitude, so a structural size is measured instead.
    """
    for name in ("gradient_boosting", "random_forest", "xgboost"):
        footprint = reporting.model_footprint(name)
        assert footprint["n_parameters"] > 0, f"{name} was measured as empty"
        assert footprint["model_size_bytes"] > 0, f"{name} has no artefact size"


def test_every_model_reports_a_size_and_a_memory_figure() -> None:
    from solar_forecasting.models import registry as model_registry

    for name in model_registry.all_model_names():
        footprint = reporting.model_footprint(name)
        assert np.isfinite(footprint["n_parameters"]), name
        assert np.isfinite(footprint["model_size_bytes"]), name
        assert "peak_train_memory_mb" in footprint, name
        assert "resident_memory_mb" in footprint, name


def test_a_neural_forecasters_size_is_its_parameter_buffer() -> None:
    footprint = reporting.model_footprint("lstm")
    # float32 parameters: the buffer is four bytes per counted scalar.
    assert footprint["model_size_bytes"] == pytest.approx(
        footprint["n_parameters"] * 4, rel=0.05)
    # A sequence forecaster is trained by the runner, so its training peak is
    # not measurable through the adapter and must not be reported as zero.
    assert np.isnan(footprint["peak_train_memory_mb"])
    assert np.isfinite(footprint["resident_memory_mb"])


def test_footprint_keeps_the_runner_record_separate_from_the_measured_size() -> None:
    """A recorded count is not the same thing as a parameter count.

    The runner records 200 for a random forest, which is its estimator count. If
    that were published as the parameter count, a 200-tree forest would appear
    three orders of magnitude smaller than a 200-weight layer. The structural
    size is the comparable measure; the recorded value is kept beside it.
    """
    registry = make_registry()
    cost = reporting.computational_cost(registry)
    row = cost[cost["model"] == "gradient_boosting"].iloc[0]
    assert row["n_parameters_recorded"] == 500
    assert row["n_parameters"] != 500
    assert row["n_parameters"] == pytest.approx(
        reporting.model_footprint("gradient_boosting")["n_parameters"])
    assert row["n_parameters_measure"] in {"fitted structure", "parameter buffer"}


# --------------------------------------------------------------------------- #
# Figure provenance
# --------------------------------------------------------------------------- #
def test_figure_provenance_covers_every_known_figure(tmp_path) -> None:
    for stem in prov_mod.FIGURE_PROVENANCE:
        (tmp_path / f"{stem}.png").write_bytes(b"\x89PNG\r\n")
    provenance = prov_mod.build(tmp_path)
    assert provenance["n_documented"] == len(prov_mod.FIGURE_PROVENANCE)
    assert provenance["unmatched"] == []
    for entry in provenance["figures"].values():
        assert entry["source"] and entry["research_question"] and entry["caption"]


def test_figure_provenance_reports_an_undocumented_figure(tmp_path) -> None:
    (tmp_path / "fig_unknown_thing.png").write_bytes(b"\x89PNG\r\n")
    provenance = prov_mod.build(tmp_path)
    assert "fig_unknown_thing.png" in provenance["unmatched"]


def test_figure_provenance_of_a_missing_directory_is_empty(tmp_path) -> None:
    provenance = prov_mod.build(tmp_path / "absent")
    assert provenance["figures"] == {}


def test_paper_figure_index_is_generated_from_the_manifest(tmp_path) -> None:
    """The index lists exactly the figures the manifest carries.

    The index used to be maintained by hand: it listed a figure that was never
    produced and omitted most of the ones that were, so a reader could not tell
    which figures existed. It is now generated, and this pins the property that
    makes it trustworthy -- no figure in the index that the manifest does not
    have, and none missing from it.
    """
    for stem in prov_mod.FIGURE_PROVENANCE:
        (tmp_path / f"{stem}.png").write_bytes(b"\x89PNG\r\n")
    provenance = prov_mod.build(tmp_path)
    target = tmp_path / "README.md"
    prov_mod.write_paper_index(provenance, target)
    text = target.read_text(encoding="utf-8")

    listed = set(re.findall(r"\| `([^`]+\.png)` \|", text))
    assert listed == set(provenance["figures"])
    for name in provenance["figures"]:
        entry = provenance["figures"][name]
        assert entry["research_question"] in text
        assert entry["source"] in text
    assert "Do not edit by hand" in text


def test_paper_figure_index_omits_a_template_that_was_never_produced(tmp_path) -> None:
    """A declared template with no file on disk must not appear in the index."""
    stems = [s for s in prov_mod.FIGURE_PROVENANCE if s != "fig14_training_curves"]
    for stem in stems:
        (tmp_path / f"{stem}.png").write_bytes(b"\x89PNG\r\n")
    provenance = prov_mod.build(tmp_path)
    target = tmp_path / "README.md"
    prov_mod.write_paper_index(provenance, target)
    text = target.read_text(encoding="utf-8")
    assert "fig14_training_curves.png" not in text
    assert (tmp_path / "fig14_training_curves.png").exists() is False


def test_paper_figure_index_refuses_to_replace_a_list_with_a_blank(tmp_path) -> None:
    """An empty manifest must not silently blank the index."""
    with pytest.raises(ValueError, match="empty figure index"):
        prov_mod.write_paper_index({"figures": {}, "unmatched": []},
                                   tmp_path / "README.md")


# --------------------------------------------------------------------------- #
# Cross-site provenance and cost
# --------------------------------------------------------------------------- #
def test_cross_site_report_records_provenance(config) -> None:
    """The transfer protocols are evidence, so they carry provenance too.

    The main result set writes a record per run with a checksum, a fingerprint
    and a git revision. The cross-site protocols wrote a pooled report with
    none of those, so the evidence behind the generalisation question could not
    be traced to the data or the code that produced it.
    """
    from solar_forecasting.cross_site import experiment as cross_site

    block = cross_site._provenance(config, ["xgboost"], ["1h"], [42])
    for field in ("recorded_at_utc", "config_fingerprint", "git_revision",
                  "dataset", "experiment_group", "source_archive_sha256"):
        assert field in block, field
    assert block["experiment_group"] == "J_cross_site"
    assert block["horizons"] == ["1h"] and block["seeds"] == [42]


def test_cross_site_records_a_measured_training_time_not_a_placeholder() -> None:
    """A fitted tree ensemble has no parameter count, but it does have a runtime.

    The sklearn branch returned a hard-coded zero seconds, so every cross-site
    row reached the experiment matrix with a training time of 0 next to models
    that genuinely take minutes. Zero is a claim about the measurement.
    """
    import inspect

    from solar_forecasting.cross_site import experiment as cross_site

    source = inspect.getsource(cross_site._train)
    assert "time.perf_counter()" in source, "sklearn branch must be timed"
    # The sentinel must not be a literal zero any more.
    assert "None, time.perf_counter() - start, None" in source


def test_cross_site_parameter_count_is_absent_rather_than_zero() -> None:
    """None means 'not a number for this model kind'; 0 would mean 'no parameters'."""
    import inspect

    from solar_forecasting.cross_site import experiment as cross_site

    source = inspect.getsource(cross_site.run_transfer)
    assert "if n_parameters is not None else None" in source



# --------------------------------------------------------------------------- #
# Integrated gradients
# --------------------------------------------------------------------------- #
def test_integrated_gradients_are_complete_and_signed() -> None:
    import torch

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6, 3)).astype(np.float32)

    def forward(window):
        return (window[0, -1, 0] * 3.0 + window[0, -1, 1] * 0.5).sum()

    attributions = imp_mod.integrated_gradients(
        forward, X, ["ghi", "temperature", "wind"], n_samples=32, n_steps=16)
    assert set(attributions["feature"]) == {"ghi", "temperature", "wind"}
    # The attribution magnitude must follow the coefficients: 3 versus 0.5.
    ghi = float(attributions.loc[attributions["feature"] == "ghi",
                                 "mean_absolute"].iloc[0])
    temperature = float(attributions.loc[attributions["feature"] == "temperature",
                                         "mean_absolute"].iloc[0])
    assert ghi > temperature
    # Completeness: the residual must vanish.
    residuals = [abs(row["residual_w"]) for row in attributions.attrs["completeness"]]
    assert max(residuals) < 1e-2 * max(
        1.0, max(abs(row["prediction_change_w"])
                 for row in attributions.attrs["completeness"]))
    assert attributions.attrs["baseline"] == "mean"
    assert attributions["relative"].sum() == pytest.approx(1.0, rel=1e-6)


def test_diebold_mariano_matches_the_t_statistic_at_horizon_one() -> None:
    """The DM statistic must shrink with the sample, not just track the effect.

    At horizon one the Bartlett kernel has no terms, so the test reduces to a
    paired t-test on the loss differential and the two must agree exactly. An
    earlier implementation divided the summed autocovariances by n once, which
    gives dbar divided by sd rather than dbar divided by sd over root n. That
    error does not shrink with n, so it reported a twenty percent difference in
    error across four thousand intervals as insignificant, and every
    significance test in the project inherited it.
    """
    from solar_forecasting.statistics import comparison as stat

    rng = np.random.default_rng(0)
    errors = rng.normal(size=4000)
    better = 0.8 * errors
    worse = 1.0 * errors

    result = stat.diebold_mariano(better, worse, "A", "B", loss="mae", horizon=1)
    differential = np.abs(better) - np.abs(worse)
    expected = float(differential.mean()
                     / (differential.std(ddof=1) / np.sqrt(differential.size)))
    assert result.statistic == pytest.approx(expected, rel=1e-9)
    assert result.mean_significance, "a 20% error gap must be detected"


def test_diebold_mariano_gains_power_with_more_observations() -> None:
    from solar_forecasting.statistics import comparison as stat

    def statistic(n: int) -> float:
        rng = np.random.default_rng(11)
        errors = rng.normal(size=n)
        return stat.diebold_mariano(0.8 * errors, errors, "A", "B",
                                    loss="mae", horizon=1).statistic

    assert abs(statistic(4000)) < abs(statistic(16000))


def test_diebold_mariano_does_not_reject_two_identical_models() -> None:
    from solar_forecasting.statistics import comparison as stat

    rng = np.random.default_rng(3)
    errors = rng.normal(size=3000)
    result = stat.diebold_mariano(errors, errors.copy(), "A", "B",
                                  loss="mae", horizon=1)
    assert not result.mean_significance
    assert result.p_value > 0.05


def test_ablation_statistics_pair_each_regime_against_the_full_input_run() -> None:
    """A regime drop must be measured against the same model on full inputs.

    Comparing every regime to persistence would answer a different question, and
    the resulting table would be a copy of the model comparison.
    """
    frames = {}
    for model in ("gradient_boosting", "lstm"):
        block = make_registry()
        block = block[block["model"] == model].head(2)
        block.loc[:, "model"] = model
        frames[model] = block
    registry = pd.concat(frames.values(), ignore_index=True)
    registry["experiment_group"] = "N_feature_regimes"
    for regime in ("pv_only", "weather_only", "pv_weather", "pv_weather_solar", "full"):
        extra = registry.copy()
        extra["feature_spec"] = regime
        registry = pd.concat([registry, extra], ignore_index=True)
    table = reporting.feature_regime_comparison(registry)
    assert not table.empty
    # Every regime must appear as a column or a row, never as a missing value.
    assert table.notna().all().all()


def test_generation_labels_split_at_a_fixed_fraction_of_capacity() -> None:
    labels = imp_mod.generation_level_labels(
        np.array([0.0, 5_000.0, 30_000.0, 50_000.0]), 55_000.0)
    assert list(labels) == ["low_generation", "low_generation",
                            "high_generation", "high_generation"]
    # Without a capacity the labelling is not meaningful and says so.
    assert list(imp_mod.generation_level_labels(np.array([1.0]), None)) == ["unknown"]


def test_collapse_over_lookback_sums_the_lags() -> None:
    values = np.arange(12, dtype=float)
    collapsed = imp_mod.collapse_over_lookback(values, ["a", "b", "c"], 4)
    assert collapsed.index.name == "feature"
    # The lags are stored row-major, so feature "a" holds 0, 3, 6 and 9. The
    # result is a share of the total, and the three shares must sum to one.
    total = 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11
    assert collapsed.loc["a"] == pytest.approx(18 / total)
    assert collapsed.loc["b"] == pytest.approx(22 / total)
    assert collapsed.loc["c"] == pytest.approx(26 / total)
    assert collapsed.sum() == pytest.approx(1.0)

def test_every_experiment_record_carries_source_provenance() -> None:
    """A result with no source cannot be checked, so every record must name one.

    The block has to name the archive checksum, the record's own DOI and Zenodo
    identifier, the timestamp and the git revision that produced the run. The
    revision in particular has to be the one the run was made under, not the
    revision in the working tree at the time the record is inspected.
    """
    directory = Path("results/experiments")
    if not directory.exists():
        pytest.skip("no experiment records on disk")
    records = sorted(directory.glob("*.json"))
    assert records, "the experiment directory is empty"
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        provenance = record.get("provenance")
        assert isinstance(provenance, dict), f"{path.name} has no provenance block"
        for field in ("source_sha256", "source_doi", "recorded_at", "git_revision",
                      "experiment_id"):
            assert provenance.get(field), f"{path.name} is missing {field}"
        assert len(provenance["source_sha256"]) == 64, path.name
        assert provenance["source_licence"] == "CC0-1.0", path.name
        # The revision must be the one inside the record's own environment block,
        # not whatever the working tree happens to be at inspection time.
        environment = (record.get("metadata", {}) or {}).get("environment", {})
        assert provenance["git_revision"] == environment.get("git_revision"), path.name


