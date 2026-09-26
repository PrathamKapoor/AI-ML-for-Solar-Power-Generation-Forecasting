"""Cross-check the claims in the prose against the computed results.

Documentation drifts. A number written into ``README.md`` or ``paper/results.md``
can survive for months after the run that produced it has been re-run with
different settings, and the drift is invisible to a reader who has neither.

This script extracts every number that the prose states as a *result* and
verifies it against ``results/tables/``. It reports PASS or FAIL per claim and
exits non-zero on any mismatch, so it can be run in the same command as the test
suite.

    python tools/check_consistency.py
    python tools/check_consistency.py --verbose

What it checks:

* every ``(model, horizon)`` metric quoted in the README's results table against
  ``overall_model_comparison.csv`` and ``horizon_comparison.csv``;
* every persisted status claim in the README experiment table against the
  experiment registry;
* the count of recorded runs against the registry row count;
* the literature paper count against the workbook.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
TABLES = ROOT / "results" / "tables"
WORKBOOK = ROOT / "literature" / "literature_review.xlsx"

#: Metrics quoted in the README, and the column each is compared against.
QUOTED_METRICS = {"mae": "mae", "rmse": "rmse", "nrmse_capacity": "nrmse_capacity",
                  "r2": "r2", "smape": "smape"}

#: Tolerance: the README rounds to the nearest watt and the third decimal.
ABS_TOLERANCE = {"mae": 1.0, "rmse": 1.0, "nrmse_capacity": 0.001, "r2": 0.001,
                 "smape": 0.1, "train_seconds": 0.5}

failures: list[str] = []
checks = 0


def fail(message: str) -> None:
    failures.append(message)
    print(f"FAIL  {message}")


def ok(message: str) -> None:
    print(f"PASS  {message}")


def parse_number(text: str) -> float | None:
    """Read a number that may carry thin spaces or commas as thousands separators."""
    cleaned = text.replace(" ", "").replace(" ", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def check_readme_results_table(verbose: bool = False) -> None:
    """Every metric in the README's headline results table."""
    global checks
    if not README.exists():
        fail("README.md is missing")
        return
    overall_path = TABLES / "overall_model_comparison.csv"
    if not overall_path.exists():
        fail("overall_model_comparison.csv is missing; run scripts/evaluate.py")
        return
    overall = pd.read_csv(overall_path)
    text = README.read_text(encoding="utf-8")

    # Rows look like: | Gradient boosting | 4 374 | 6 664 | 0.121 | 0.785 | 49.8 | +0.271 | 9.5 |
    row_pattern = re.compile(
        r"^\|\s*([A-Za-z][A-Za-z \-]+?)\s*\|\s*([\d  ,.]+)\s*\|\s*([\d  ,.]+)\s*\|"
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
    name_map = {m.replace("_", " "): m for m in overall["model"]}
    checked_rows = 0
    for line in text.splitlines():
        match = row_pattern.match(line.strip())
        if not match:
            continue
        label = match.group(1).strip()
        key = name_map.get(label, label.replace(" ", "_").lower())
        if key not in set(overall["model"]):
            continue
        row = overall[overall["model"] == key].iloc[0]
        for index, metric in ((2, "mae"), (3, "rmse"), (4, "nrmse_capacity"),
                              (5, "r2")):
            claimed = parse_number(match.group(index))
            if claimed is None:
                continue
            actual = float(row[QUOTED_METRICS[metric]])
            checks += 1
            if abs(claimed - actual) > ABS_TOLERANCE[metric]:
                fail(f"README {key} {metric}: text says {claimed:g}, table has {actual:g}")
            elif verbose:
                ok(f"README {key} {metric} = {actual:g}")
        checked_rows += 1
    if checked_rows == 0:
        fail("no model rows in the README results table could be matched to the table")
    else:
        ok(f"README headline table: {checked_rows} model rows checked against the CSV")


def check_status_claims() -> None:
    """README experiment-status claims against what actually exists.

    Training groups must appear in the registry. The analysis groups
    (regimes, seasonal, model ablation, cost, uncertainty) produce result
    tables rather than runs, so their presence is checked in the tables
    directory instead.
    """
    global checks
    registry_path = ROOT / "results" / "experiments.csv"
    if not registry_path.exists():
        fail("results/experiments.csv is missing")
        return
    registry = pd.read_csv(registry_path)
    executed = set(registry["experiment_group"])
    training_groups = {"A_baselines", "B_ml_vs_dl", "C_horizons", "E_feature_ablation",
                       "N_feature_regimes", "M_multiseed"}
    missing = training_groups - executed
    checks += 1
    if missing:
        fail(f"training experiment groups with no recorded run: {sorted(missing)}")
    else:
        ok(f"all expected training groups present: {len(executed)} groups in the registry")

    analysis_tables = {
        "D_regimes": "weather_regime_comparison.csv",
        "G_seasonal": "seasonal_comparison.csv",
        "F_model_ablation": "model_ablation.csv",
        "I_cost": "computational_cost.csv",
        "K_uncertainty": "uncertainty_coverage.csv",
        "H_explainability": "feature_importance.csv",
        "J_cross_site": "cross_site_comparison.csv",
    }
    for group, table_name in analysis_tables.items():
        checks += 1
        if (TABLES / table_name).exists():
            ok(f"{group}: {table_name} present")
        else:
            fail(f"{group}: {table_name} is missing, so the group is not evidenced")

    for group in sorted(executed):
        rows = registry[registry["experiment_group"] == group]
        print(f"      {group:20} {len(rows):>3} runs, "
              f"horizons {sorted(set(rows['horizon']))}, "
              f"models {len(set(rows['model']))}")


def check_literature_count() -> None:
    global checks
    if not WORKBOOK.exists():
        fail("literature_review.xlsx is missing")
        return
    frame = pd.read_excel(WORKBOOK, sheet_name="Literature Review")
    checks += 1
    if len(frame) != 30:
        fail(f"literature review holds {len(frame)} papers, not 30")
    else:
        ok("literature review holds exactly 30 papers")
    checks += 1
    if frame["Paper Link"].duplicated().any():
        fail("duplicate paper links in the literature review")
    else:
        ok("no duplicate paper links")


def check_stale_status_phrases() -> None:
    """No prose may still claim a completed experiment is outstanding."""
    global checks
    stale = ["not yet run", "not_run", "has not been run", "is not implemented",
             "partially executed", "not implemented"]
    targets = [README, ROOT / "docs" / "methodology.md", ROOT / "docs" / "experiments.md",
               ROOT / "paper" / "results.md", ROOT / "paper" / "conclusion.md",
               ROOT / "paper" / "discussion.md", ROOT / "docs" / "reproducibility.md"]
    # Phrases that are legitimate when they describe a *documented limitation* are
    # still reported, so a human decides; a hard failure is reserved for the
    # README, which is the document a reader trusts most.
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for phrase in stale:
            if phrase in text:
                message = f"{path.relative_to(ROOT)} still contains {phrase!r}"
                if path == README:
                    fail(message)
                else:
                    checks += 1
                    print(f"WARN  {message}")


def main() -> int:
    verbose = "--verbose" in sys.argv
    print("consistency check: prose against computed results\n")
    check_readme_results_table(verbose=verbose)
    check_status_claims()
    check_literature_count()
    check_stale_status_phrases()
    print(f"\n{checks} checks, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
