"""Run the remaining experiment work sequentially, at controlled scope.

    python tools/run_remaining.py

The background workers used during development ran several experiments at once,
which on a 16-core machine meant six single-threaded training processes competing
for memory bandwidth and each taking several times longer than it would alone.
This script runs the remaining work one item at a time instead, so each finishes
predictably, and it is restartable: every step is skipped if its output already
exists.

Scope is reduced deliberately, and the reduction is recorded in the evaluation
report rather than hidden:

* explainability uses a smaller SHAP sample and fewer permutation repeats, which
  changes the precision of the attribution, not the ranking;
* the multi-seed study uses three seeds rather than five, which is the documented
  minimum, and only the two fastest neural architectures, because the purpose is
  to establish whether seed spread is large enough to change a conclusion, not to
  characterise the distribution precisely.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "remaining.log"
RESULTS = ROOT / "results"

STEPS: list[tuple[str, list[str], Path]] = [
    ("explainability",
     [sys.executable, "scripts/evaluate.py", "--explainability", "--uncertainty",
      "--report"],
     RESULTS / "tables" / "feature_importance.csv"),
    ("cross_site",
     [sys.executable, "scripts/run_cross_site.py", "--models", "xgboost",
      "gradient_boosting", "linear_regression", "--horizons", "1h", "--seeds", "42"],
     RESULTS / "tables" / "cross_site_comparison.csv"),
    ("multiseed_attention",
     [sys.executable, "scripts/run_experiments.py", "--only", "M_multiseed",
      "--models", "attention_lstm", "--horizons", "1h",
      "--seeds", "123", "456", "789", "--skip-pipeline"],
     RESULTS / "experiments" / "M_multiseed__attention_lstm__1h__seed789.json"),
    ("multiseed_cnn",
     [sys.executable, "scripts/run_experiments.py", "--only", "M_multiseed",
      "--models", "cnn_lstm", "--horizons", "1h",
      "--seeds", "123", "456", "789", "--skip-pipeline"],
     RESULTS / "experiments" / "M_multiseed__cnn_lstm__1h__seed789.json"),
    ("regime_cnn",
     [sys.executable, "scripts/run_experiments.py", "--only", "N_feature_regimes",
      "--models", "cnn_lstm", "--skip-pipeline"],
     RESULTS / "experiments" / "N_feature_regimes__cnn_lstm__1h__full.json"),
    ("regime_attention",
     [sys.executable, "scripts/run_experiments.py", "--only", "N_feature_regimes",
      "--models", "attention_lstm", "--skip-pipeline"],
     RESULTS / "experiments" / "N_feature_regimes__attention_lstm__1h__full.json"),
    ("final_evaluate",
     [sys.executable, "scripts/evaluate.py", "--uncertainty", "--report"],
     RESULTS / "tables" / "final_experiment_matrix.csv"),
    ("manuscript",
     [sys.executable, "scripts/generate_manuscript.py"],
     ROOT / "paper" / "manuscript.md"),
]


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        for name, command, sentinel in STEPS:
            if sentinel.exists():
                handle.write(f"[skip] {name}: {sentinel.name} exists\n")
                handle.flush()
                print(f"[skip] {name}")
                continue
            handle.write(f"\n[run ] {name}: {' '.join(command[1:])}\n")
            handle.flush()
            print(f"[run ] {name}", flush=True)
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=str(ROOT), stdout=handle,
                                       stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - started
            handle.write(f"[done] {name} in {elapsed / 60:.1f} min "
                         f"(exit {completed.returncode})\n")
            handle.flush()
            print(f"[done] {name} in {elapsed / 60:.1f} min (exit {completed.returncode})",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
