#!/usr/bin/env python
"""Train a single model and record the run.

    python scripts/train.py --model lstm
    python scripts/train.py --model xgboost --horizon 6h
    python scripts/train.py --model cnn_lstm --features no_weather --seed 7

The split, the input set, the seed and the hyperparameters come from
``configs/``; the command line only chooses what to run. The result is written
through the same path as ``scripts/run_experiments.py`` -- one JSON record per
run, the CSV registry and the per-step prediction file -- so a single-model run
and a matrix run are indistinguishable in the results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from solar_forecasting.config import load_config, set_thread_env  # noqa: E402
from solar_forecasting.models import registry as model_registry  # noqa: E402
from solar_forecasting.preprocessing.pipeline import run_pipeline  # noqa: E402
from solar_forecasting.training.experiment import ExperimentSpec, run_experiment, save_experiment  # noqa: E402
from solar_forecasting.training.runner import append_to_registry, result_row  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model name from the registry")
    parser.add_argument("--horizon", default=None, help="forecast horizon, e.g. 15min, 1h, 6h, 24h")
    parser.add_argument("--features", default="full",
                        help="'full' or an ablation group from configs/data.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--group", default="single_run",
                        help="experiment group label recorded with the run")
    parser.add_argument("--rerun-pipeline", action="store_true",
                        help="rebuild the processed data before training")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    set_thread_env(args.threads)
    config = load_config()
    horizons = {h["name"]: int(h["steps"]) for h in config.section("horizons")}
    horizon = args.horizon or str(config.section("default_horizon"))
    if horizon not in horizons:
        parser.error(f"unknown horizon {horizon!r}; available: {sorted(horizons)}")
    try:
        model_registry.model_info(args.model)
    except KeyError:
        parser.error(f"unknown model {args.model!r}; available: "
                     f"{sorted(model_registry.all_model_names())}")
    seed = args.seed if args.seed is not None else int(
        (config.section("experiments").get("defaults", {}) or {}).get("seed", 42))

    processed = config.project_root_for("data") / "processed" / "split_test.parquet"
    if args.rerun_pipeline or not processed.exists():
        print("Running the data pipeline...")
        state = run_pipeline(config, quiet=True)
    else:
        print("Reusing cached processed artefacts (pass --rerun-pipeline to rebuild).")
        from _pipeline_state import load_pipeline_state
        state = load_pipeline_state(config)

    suffix = "" if args.features == "full" else f"__{args.features}"
    spec = ExperimentSpec(
        experiment_id=f"{args.group}__{args.model}__{horizon}{suffix}",
        model=args.model, horizon=horizon, horizon_steps=horizons[horizon],
        features=args.features, seed=seed,
        label=f"{args.group} / {args.model} / {horizon} / {args.features}",
        notes="single-model run from scripts/train.py")

    print(f"\nRunning {spec.experiment_id}")
    started = time.perf_counter()
    result = run_experiment(
        config=config, spec=spec, featured=state["featured"],
        split_train=state["split"].train, split_val=state["split"].val,
        split_test=state["split"].test, columns=state["columns"],
        target_scaler=state["scalers"].target_scaler,
        rated_w=state["station"].get("rated_w"), progress=True)
    result.metadata["wall_seconds_including_setup"] = time.perf_counter() - started

    path = save_experiment(result)
    if result.predictions is not None:
        prediction_path = (config.project_root_for("results") / "predictions"
                           / f"{spec.experiment_id}.parquet")
        result.predictions.to_parquet(prediction_path, index=False)
        print(f"  predictions: {prediction_path.relative_to(config.project_root_for('results'))}")
    row = result_row(result, args.group, config)
    append_to_registry([row])
    print(f"  record     : {path.relative_to(config.project_root_for('results'))}")
    print(f"  registry   : results/experiments.csv")
    daylight = result.metrics_daylight
    print(f"\nDaylight metrics: MAE {daylight.get('mae', float('nan')):,.1f} W  "
          f"RMSE {daylight.get('rmse', float('nan')):,.1f} W  "
          f"nRMSE {daylight.get('nrmse_capacity', float('nan')):.4f}  "
          f"R2 {daylight.get('r2', float('nan')):.4f}  "
          f"skill vs persistence {daylight.get('skill_vs_persistence_rmse', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
