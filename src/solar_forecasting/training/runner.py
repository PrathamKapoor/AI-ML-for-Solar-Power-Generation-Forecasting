"""Experiment driver: builds the experiment matrix and runs it.

This module owns the mapping from ``configs/experiments.yaml`` to concrete
:class:`~solar_forecasting.training.experiment.ExperimentSpec` objects, and
persists a tidy long-format record of every run. Keeping the matrix
construction separate from the running keeps the declared experiment set
auditable in one place.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import Config, results_dir
from ..utils.io import write_json, write_table
from .experiment import (ExperimentResult, ExperimentSpec, horizon_lookup,
                         run_experiment, save_experiment)

#: Long-format column order for results/experiments.csv
EXPERIMENT_COLUMNS = [
    "experiment_id", "experiment_group", "model", "model_family", "model_kind",
    "horizon", "horizon_steps", "features", "feature_spec", "seed",
    "n_features", "n_train", "n_val", "n_test",
    "train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
    "rmse", "mae", "nrmse_capacity", "nrmse_mean", "r2", "smape", "bias", "peak_error",
    "persistence_rmse", "smart_persistence_rmse",
    "skill_vs_persistence_rmse", "skill_vs_smart_persistence_rmse",
    "rmse_all_steps", "mae_all_steps", "r2_all_steps",
    "train_seconds", "inference_ms_per_window", "n_parameters",
    "config_fingerprint", "git_revision", "python_version", "torch_version",
    "notes",
]


@dataclass
class ExperimentPlan:
    """One declared experiment: a group of specs to run."""

    group: str
    description: str
    specs: list[ExperimentSpec]


def _horizons(config: Config, names: Iterable[str] | None) -> list[dict[str, Any]]:
    table = horizon_lookup(config)
    wanted = list(names) if names else [config.section("default_horizon")]
    out = []
    for name in wanted:
        if name not in table:
            raise KeyError(f"unknown horizon {name!r}. Available: {sorted(table)}")
        out.append({"name": name, "steps": int(table[name]["steps"]),
                    "hours": float(table[name]["hours"])})
    return out


def build_plans(config: Config, only: list[str] | None = None,
                skip: list[str] | None = None,
                horizons: list[str] | None = None,
                seeds: list[int] | None = None) -> list[ExperimentPlan]:
    """Turn the experiment configuration into runnable plans.

    Experiments marked ``analysis_only`` in the configuration (model ablation,
    explainability, cost, uncertainty) and those marked
    ``reuse_predictions_from`` are post-hoc analyses of other experiments'
    predictions rather than independent training runs, so they are not included
    here; ``scripts/evaluate.py`` performs them against the stored predictions.

    ``horizons`` restricts every plan to a subset of its declared horizons, so
    that one long experiment group can be split across parallel processes.

    ``seeds`` replaces the single declared seed with one plan entry per seed,
    which is what the multi-seed experiment needs: a repeated run of the *same*
    configuration, distinguished by seed so that the spread across initialisations
    can be reported.
    """
    declared = config.section("experiments")
    defaults = declared.get("defaults", {})
    seed_list = [int(s) for s in seeds] if seeds else [int(defaults.get("seed", 42))]

    plans: list[ExperimentPlan] = []
    for group, spec in declared.items():
        if not isinstance(spec, dict) or group == "defaults":
            continue
        # An entry without a model list is configuration for the analysis stage
        # (for example the shared bootstrap and conformal settings), not a
        # training experiment.
        if not (spec.get("models") or spec.get("feature_specs")):
            continue
        if spec.get("holdout_stations"):
            # Cross-site transfer is a separate procedure: it needs a holdout
            # panel that is never seen during fitting, so it is run by
            # scripts/run_experiments.py --cross-site rather than here.
            continue
        if spec.get("analysis_only"):
            # Analytical experiment: assembled by evaluate.py from other runs.
            continue
        if only and group not in only:
            continue
        if skip and group in skip:
            continue
        if spec.get("reuse_predictions_from") and not only:
            # Post-hoc analysis of stored predictions, handled by evaluate.py.
            continue

        models = spec.get("models") or defaults.get("models", [])
        horizon_names = spec.get("horizons") or defaults.get("horizons")
        if horizons:
            horizon_names = [h for h in horizon_names if h in set(horizons)]
            if not horizon_names:
                continue
        hz = _horizons(config, horizon_names)
        feature_specs = spec.get("feature_specs") or ["full"]

        specs: list[ExperimentSpec] = []
        for model in models:
            for horizon in hz:
                for feature_spec in feature_specs:
                    for seed in seed_list:
                        suffix = "" if feature_spec == "full" else f"__{feature_spec}"
                        seed_suffix = "" if seed_list == [int(defaults.get("seed", 42))] \
                            else f"__seed{seed}"
                        specs.append(ExperimentSpec(
                            experiment_id=(f"{group}__{model}__{horizon['name']}"
                                           f"{suffix}{seed_suffix}"),
                            model=model,
                            horizon=horizon["name"],
                            horizon_steps=horizon["steps"],
                            features=feature_spec,
                            seed=seed,
                            ablation=(spec["description"] if feature_spec != "full" else None),
                            label=(f"{group} / {model} / {horizon['name']} / "
                                   f"{feature_spec} / seed {seed}"),
                            notes=spec.get("description", ""),
                        ))
        plans.append(ExperimentPlan(group=group,
                                    description=spec.get("description", ""),
                                    specs=specs))
    return plans


def result_row(result: ExperimentResult, group: str, config: Config) -> dict[str, Any]:
    """Flatten one result into the long-format record."""
    spec = result.spec
    meta = result.metadata
    day = result.metrics_daylight
    allm = result.metrics_all_steps
    env = meta.get("environment", {})
    ranges = meta.get("split_date_ranges", {})
    targets = meta.get("target_split_dates", {})
    sizes = meta.get("split_sizes", {})
    cost = meta.get("cost", {})

    def first(values: Iterable[Any]) -> Any:
        for value in values:
            if value:
                return value
        return None

    return {
        "experiment_id": spec.experiment_id,
        "experiment_group": group,
        "model": spec.model,
        "model_family": meta.get("model_family"),
        "model_kind": meta.get("model_kind"),
        "horizon": spec.horizon,
        "horizon_steps": spec.horizon_steps,
        "features": spec.features,
        "feature_spec": spec.features,
        "seed": spec.seed,
        "n_features": meta.get("n_features"),
        "n_train": sizes.get("train"),
        "n_val": sizes.get("val"),
        "n_test": sizes.get("test"),
        "train_start": first([targets.get("train", [None])[0]]),
        "train_end": first([targets.get("train", [None, None])[-1]]),
        "val_start": first([targets.get("val", [None])[0]]),
        "val_end": first([targets.get("val", [None, None])[-1]]),
        "test_start": first([targets.get("test", [None])[0]]),
        "test_end": first([targets.get("test", [None, None])[-1]]),
        "rmse": day.get("rmse"),
        "mae": day.get("mae"),
        "nrmse_capacity": day.get("nrmse_capacity"),
        "nrmse_mean": day.get("nrmse_mean"),
        "r2": day.get("r2"),
        "smape": day.get("smape"),
        "bias": day.get("bias"),
        "peak_error": day.get("peak_error"),
        "persistence_rmse": day.get("persistence_rmse"),
        "smart_persistence_rmse": day.get("smart_persistence_rmse"),
        "skill_vs_persistence_rmse": day.get("skill_vs_persistence_rmse"),
        "skill_vs_smart_persistence_rmse": day.get("skill_vs_smart_persistence_rmse"),
        "rmse_all_steps": allm.get("rmse"),
        "mae_all_steps": allm.get("mae"),
        "r2_all_steps": allm.get("r2"),
        "train_seconds": cost.get("train_seconds"),
        "inference_ms_per_window": cost.get("inference_ms_per_window"),
        "n_parameters": cost.get("n_parameters"),
        "config_fingerprint": config.fingerprint(),
        "git_revision": env.get("git_revision"),
        "python_version": env.get("platform", {}).get("python"),
        "torch_version": env.get("torch", {}).get("version"),
        "notes": spec.notes,
    }


def append_to_registry(rows: list[dict[str, Any]], path=None) -> pd.DataFrame:
    """Merge new rows into the long-format registry, replacing duplicate ids."""
    target = path or (results_dir() / "experiments.csv")
    if target.is_dir():
        # A stray directory at this path (an earlier bug created one) would
        # otherwise surface as a bare PermissionError from pandas. Fail with the
        # actual cause instead.
        raise IsADirectoryError(
            f"the experiment registry path {target} is a directory; remove it and "
            f"re-run so the CSV registry can be written")
    if target.exists():
        existing = pd.read_csv(target)
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(rows)
    combined = combined.drop_duplicates(subset=["experiment_id"], keep="last")
    ordered = [c for c in EXPERIMENT_COLUMNS if c in combined.columns]
    extra = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + extra]
    write_table(combined, target)
    return combined


def run_plans(config: Config, plans: list[ExperimentPlan], pipeline_result: dict[str, Any],
              only_models: list[str] | None = None, progress: bool = True
              ) -> tuple[pd.DataFrame, dict[str, ExperimentResult]]:
    """Execute every plan and persist records, the registry and the predictions."""
    featured = pipeline_result["featured"]
    split = pipeline_result["split"]
    columns = pipeline_result["columns"]
    scalers = pipeline_result["scalers"]
    rated_w = pipeline_result["station"].get("rated_w")

    rows: list[dict[str, Any]] = []
    results: dict[str, ExperimentResult] = {}
    predictions_dir = results_dir("predictions")

    for plan in plans:
        if progress:
            print(f"\n  {plan.group}: {plan.description}")
        for spec in plan.specs:
            if only_models and spec.model not in only_models:
                continue
            key = (spec.model, spec.horizon, spec.features, spec.seed)
            if any((r.spec.model, r.spec.horizon, r.spec.features, r.spec.seed) == key
                   for r in results.values()):
                continue
            started = time.perf_counter()
            result = run_experiment(
                config=config, spec=spec, featured=featured,
                split_train=split.train, split_val=split.val, split_test=split.test,
                columns=columns, target_scaler=scalers.target_scaler,
                rated_w=rated_w, progress=progress)
            wall = time.perf_counter() - started
            result.metadata["wall_seconds_including_setup"] = wall
            results[spec.experiment_id] = result
            save_experiment(result)
            if result.predictions is not None:
                result.predictions.to_parquet(
                    predictions_dir / f"{spec.experiment_id}.parquet", index=False)
            rows.append(result_row(result, plan.group, config))

    registry = append_to_registry(rows) if rows else pd.DataFrame()
    return registry, results


class _RecordView:
    """Attribute view over a persisted experiment record.

    The JSON record written by :func:`save_experiment` contains exactly the four
    fields :func:`result_row` reads, so a record on disk can be turned back into
    a registry row without refitting anything.
    """

    __slots__ = ("spec", "metrics_daylight", "metrics_all_steps", "metadata")

    def __init__(self, record: dict[str, Any]) -> None:
        self.spec = _SpecView(record["spec"])
        self.metrics_daylight = record.get("metrics_daylight", {}) or {}
        self.metrics_all_steps = record.get("metrics_all_steps", {}) or {}
        self.metadata = record.get("metadata", {}) or {}


class _SpecView:
    """Minimal attribute container for a persisted experiment specification."""

    __slots__ = ("experiment_id", "model", "horizon", "horizon_steps", "features", "seed",
                 "ablation", "label", "notes")

    def __init__(self, spec: dict[str, Any]) -> None:
        for name in self.__slots__:
            setattr(self, name, spec.get(name))


def rebuild_registry_from_disk(config: Config, directory: Path | None = None) -> pd.DataFrame:
    """Rebuild ``results/experiments.csv`` from the per-run JSON records.

    The JSON files are the authoritative record: each is written the moment a run
    finishes, whereas the CSV is a derived index that several processes may append
    to. Rebuilding from the records therefore removes any ordering dependency, and
    it means a run that finished just before an interruption is not lost.
    """
    source = directory or results_dir("experiments")
    if not source.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A truncated file from a killed process: skip it rather than fail the
            # whole rebuild, and let the missing run show up as an absent row.
            continue
        if "spec" not in record:
            continue
        view = _RecordView(record)
        group = str(view.spec.experiment_id).split("__")[0]
        rows.append(result_row(view, group, config))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    ordered = [c for c in EXPERIMENT_COLUMNS if c in frame.columns]
    extras = [c for c in frame.columns if c not in ordered]
    frame = frame[ordered + extras]
    frame = frame.sort_values("experiment_id").reset_index(drop=True)
    write_table(frame, results_dir() / "experiments.csv")
    return frame


def summarise_registry(registry: pd.DataFrame) -> dict[str, Any]:
    """Counts and headline numbers, written next to the registry."""
    if registry.empty:
        return {"n_experiments": 0}
    return {
        "n_experiments": int(len(registry)),
        "groups": sorted(registry["experiment_group"].unique().tolist()),
        "models": sorted(registry["model"].unique().tolist()),
        "horizons": sorted(registry["horizon"].unique().tolist()),
        "feature_specs": sorted(registry["feature_spec"].unique().tolist()),
        "n_errors_recorded": int(registry["rmse"].isna().sum()),
        "best_rmse_run": (
            registry.loc[registry["rmse"].idxmin(), ["experiment_id", "model", "horizon",
                                                      "feature_spec", "rmse"]].to_dict()
            if registry["rmse"].notna().any() else None),
    }


def write_summary(registry: pd.DataFrame) -> Path:
    summary = summarise_registry(registry)
    path = results_dir("metrics") / "experiment_summary.json"
    write_json(summary, path)
    return path
