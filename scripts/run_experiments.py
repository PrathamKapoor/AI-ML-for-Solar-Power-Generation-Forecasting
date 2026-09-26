#!/usr/bin/env python
"""Run the declared experiment matrix.

    python scripts/run_experiments.py                       # all training experiments
    python scripts/run_experiments.py --list                # show the planned matrix
    python scripts/run_experiments.py --only A_baselines    # one group
    python scripts/run_experiments.py --models lstm xgboost # subset of models
    python scripts/run_experiments.py --skip C_horizons     # skip a group

Runs the data pipeline first if the processed artefacts are missing, then trains
and scores every declared combination and writes:

    results/experiments/<experiment_id>.json     one record per run
    results/experiments.csv                      long-format registry
    results/predictions/<experiment_id>.parquet  per-step predictions
    results/metrics/experiment_summary.json      counts and headline numbers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

import pandas as pd  # noqa: E402

from solar_forecasting.config import load_config, set_thread_env  # noqa: E402
from solar_forecasting.preprocessing.pipeline import run_pipeline  # noqa: E402
from solar_forecasting.training.runner import (build_plans, run_plans,  # noqa: E402
                                              write_summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None,
                        help="experiment groups to run (default: all training experiments)")
    parser.add_argument("--skip", nargs="*", default=None,
                        help="experiment groups to skip")
    parser.add_argument("--models", nargs="*", default=None,
                        help="restrict to these model names")
    parser.add_argument("--horizons", nargs="*", default=None,
                        help="restrict to these horizons (splits one group "
                             "across parallel processes)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="repeat every plan once per seed, for the "
                             "multi-seed experiment")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="print the planned experiment matrix and exit")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="reuse cached processed artefacts")
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS/OMP threads; pinned so runtimes are comparable")
    args = parser.parse_args()

    set_thread_env(args.threads)
    config = load_config()
    plans = build_plans(config, only=args.only, skip=args.skip,
                        horizons=args.horizons, seeds=args.seeds)

    if args.list_only:
        total = 0
        for plan in plans:
            print(f"\n{plan.group}  ({len(plan.specs)} runs)")
            print(f"  {plan.description}")
            for spec in plan.specs:
                print(f"    {spec.experiment_id}")
            total += len(plan.specs)
        print(f"\ntotal runs planned: {total}")
        return 0

    processed = config.project_root_for("data") / "processed" / "split_test.parquet"
    if args.skip_pipeline and processed.exists():
        print("Reusing cached processed artefacts.")
        from _pipeline_state import load_pipeline_state
        pipeline_result = load_pipeline_state(config)
    else:
        print("Running the data pipeline...")
        started = time.perf_counter()
        pipeline_result = run_pipeline(config, quiet=True)
        print(f"  pipeline finished in {time.perf_counter() - started:.1f}s")

    print(f"\nRunning {sum(len(p.specs) for p in plans)} experiments "
          f"across {len(plans)} groups")
    started = time.perf_counter()
    registry, results = run_plans(config, plans, pipeline_result,
                                  only_models=args.models)
    elapsed = time.perf_counter() - started

    print(f"\nCompleted {len(results)} runs in {elapsed / 60:.1f} min")
    path = write_summary(registry)
    print(f"  registry : results/experiments.csv ({len(registry)} rows)")
    print(f"  summary  : {path.relative_to(config.project_root_for('results'))}")

    if not registry.empty:
        print("\nDaylight-horizon results (RMSE, W):")
        view = registry[["model", "horizon", "feature_spec", "mae", "rmse",
                         "nrmse_capacity", "r2", "skill_vs_persistence_rmse"]].copy()
        view = view.sort_values(["horizon", "rmse"])
        with pd.option_context("display.width", 160, "display.max_rows", 200):
            print(view.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

