"""Cross-site generalisation experiments.

    python scripts/run_cross_site.py --list
    python scripts/run_cross_site.py --models xgboost lstm --horizons 1h
    python scripts/run_cross_site.py --protocols loso

Four protocols are available and all four are reported:

``within``   train and test inside one site; the reference every transfer
             number is compared against
``cross``    train on the primary site, test on each held-out site
``multi``    train on several pooled sites, hold one out
``loso``     ``multi`` repeated for every site, one transfer number per site

Site selection is fixed in ``configs/data.yaml`` (``dataset.holdout_stations``)
and is based on capacity spread and record coverage. It is deliberately not
re-selected here, because choosing sites after seeing transfer errors would
invalidate the experiment.
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
from solar_forecasting.cross_site.experiment import (cross_site_table,  # noqa: E402
                                                     run_cross_site)
from solar_forecasting.utils.io import write_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=["xgboost", "lstm"],
                        help="models to fit")
    parser.add_argument("--horizons", nargs="*", default=["1h"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--sites", nargs="*", default=None,
                        help="override the configured holdout panel")
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    set_thread_env(args.threads)
    config = load_config()
    panel = args.sites or config.section("dataset").get("holdout_stations", [])
    primary = config.section("dataset")["primary_station"]

    if args.list_only:
        print(f"train site     : {primary} "
              f"({config.section('dataset')['primary_station']})")
        print(f"holdout panel  : {', '.join(panel)}")
        print(f"models         : {', '.join(args.models)}")
        print(f"horizons       : {', '.join(args.horizons)}")
        print(f"seeds          : {args.seeds}")
        print("\nprotocols: within_site, cross_site, leave_one_site_out "
              "(multi-site pooling is the training side of leave_one_site_out)")
        cycles = len(args.models) * len(args.horizons) * len(args.seeds)
        sites = 1 + len(panel)
        print(f"\nfit/predict cycles: {cycles * sites * sites}")
        return 0

    started = time.perf_counter()
    print(f"cross-site panel: {primary} -> {', '.join(panel)}")
    report = run_cross_site(config, args.models, args.horizons, args.seeds,
                            holdout=panel, quiet=True)
    table = cross_site_table(report)
    if not table.empty:
        path = write_table(table, config.project_root_for("results") / "tables" /
                           "cross_site_comparison.csv")
        print(f"\nwrote {path.relative_to(config.project_root_for('results'))} "
              f"({len(table)} rows)")

    view = table[[c for c in ("protocol", "model", "horizon", "test_site", "n",
                              "nrmse_capacity", "rmse", "skill_vs_persistence_rmse")
                  if c in table.columns]]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(view.round(4).to_string(index=False))
    print(f"\ncompleted in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
