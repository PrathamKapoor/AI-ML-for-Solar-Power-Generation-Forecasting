"""Print the executed result set in one place, for writing the final prose.

    python tools/summarise_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def show(name: str, columns: list[str], sort: list[str] | None = None) -> None:
    path = TABLES / name
    if not path.exists():
        print(f"\n### {name}: NOT PRODUCED")
        return
    frame = pd.read_csv(path)
    if sort:
        frame = frame.sort_values(sort)
    subset = [c for c in columns if c in frame.columns]
    print(f"\n### {name}  ({len(frame)} rows)")
    with pd.option_context("display.width", 220, "display.max_rows", 80,
                           "display.max_colwidth", 34):
        print(frame[subset].round(4).to_string(index=False))


def main() -> int:
    print("=" * 78)
    print("EXECUTED RESULT SET")
    print("=" * 78)
    show("overall_model_comparison.csv",
         ["model", "model_family", "mae", "rmse", "nrmse_capacity", "r2",
          "skill_vs_persistence_rmse", "train_seconds"], ["rmse"])
    show("horizon_comparison.csv",
         ["model", "horizon", "rmse", "nrmse_capacity", "skill_vs_persistence_rmse"])
    show("feature_regimes.csv",
         ["model", "feature_regime", "n_features", "nrmse_capacity",
          "skill_vs_persistence_rmse"])
    show("multi_seed_results.csv",
         ["model", "n_seeds", "rmse_mean", "rmse_std", "skill_vs_persistence_rmse_mean",
          "skill_vs_persistence_rmse_std"])
    show("cross_site_comparison.csv",
         ["protocol", "model", "test_site", "nrmse_capacity",
          "skill_vs_persistence_rmse"])
    show("weather_regime_comparison.csv",
         ["model", "stratum", "n", "nrmse_capacity", "skill_vs_persistence_rmse"])
    show("statistical_significance.csv",
         ["model", "loss", "skill_point", "skill_ci_low", "skill_ci_high",
          "dm_p_value", "dm_lag1_autocorrelation"])
    show("uncertainty_coverage.csv",
         ["model", "alpha", "scope", "nominal_coverage", "empirical_coverage",
          "interval_half_width_w"])
    matrix = pd.read_csv(TABLES / "final_experiment_matrix.csv")
    print(f"\n### final_experiment_matrix.csv ({len(matrix)} rows)")
    print("  by research question:", matrix["research_question"].value_counts().to_dict())
    print("  by group           :", matrix["experiment_group"].value_counts().to_dict())
    print("  horizons           :", sorted(set(matrix["horizon"])))
    print("  models             :", len(set(matrix["model"])))
    print("  sites              :", sorted(set(matrix["site"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
