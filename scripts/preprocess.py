#!/usr/bin/env python
"""Run the seventeen-stage data pipeline.

    python scripts/preprocess.py
    python scripts/preprocess.py --quiet

Writes:
    data/processed/featured_primary_station.parquet
    data/processed/split_{train,val,test}.parquet
    data/processed/feature_columns.json
    results/metrics/pipeline_report.json
    results/tables/dataset_statistics.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from solar_forecasting.config import load_config  # noqa: E402
from solar_forecasting.preprocessing.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="suppress stage logging")
    args = parser.parse_args()

    config = load_config()
    result = run_pipeline(config, quiet=args.quiet)
    report = result["report"]

    print("\nPipeline complete.")
    print(f"  config fingerprint : {report['config_fingerprint']}")
    print(f"  validation         : {report['validation']['n_errors']} errors, "
          f"{report['validation']['n_warnings']} warnings")
    print(f"  split sizes        : {report['stages']['14_split']['sizes']}")
    print(f"  features           : {report['stages']['15_scaling']['n_features']}")
    cleaning = report["stages"]["08_outliers"]["irradiance"]
    print(f"  irradiance cleaned : {cleaning['n_outside_physical_envelope']} outside envelope, "
          f"{cleaning['n_local_spikes']} local spikes, "
          f"{cleaning['n_night_offset_zeroed']} night-offset values zeroed")
    print(f"  night bias         : {cleaning['night_bias_before_wm2']:.2f} -> "
          f"{cleaning['night_bias_after_wm2']:.2f} W/m2")
    print("\n  report: results/metrics/pipeline_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
