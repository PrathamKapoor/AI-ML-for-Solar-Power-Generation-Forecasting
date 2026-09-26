#!/usr/bin/env python
"""Launch the remaining training work as parallel background processes.

    python tools/launch_experiments.py horizons 1h 6h 24h
    python tools/launch_experiments.py ablation
    python tools/launch_experiments.py seeds 123 456 789 2026
    python tools/launch_experiments.py --status

Each worker writes to its own log under ``logs/`` (git-ignored) and records its
process id in ``logs/workers.json``, so a crashed run can be identified and
restarted without disturbing the others. The per-run JSON records are the source
of truth, so a worker's only shared write is the CSV registry, which
``scripts/evaluate.py`` rebuilds from those records anyway.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
STATE = LOG_DIR / "workers.json"


def log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def launch(name: str, args: list[str]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = log_path(name).open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "scripts/run_experiments.py", *args, "--skip-pipeline"],
        cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    state[name] = {"pid": process.pid, "args": args, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"launched {name:12} pid={process.pid:<7} {' '.join(args)}")
    return process.pid


def running(pid: int) -> bool:
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(pid)
    except Exception:
        pass
    # Fall back to a platform query that needs no extra dependency.
    if sys.platform == "win32":
        import subprocess as sp
        result = sp.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                        capture_output=True, text=True)
        return str(pid) in result.stdout
    import os
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def show_status() -> None:
    if not STATE.exists():
        print("no workers recorded")
        return
    state = json.loads(STATE.read_text(encoding="utf-8"))
    print(f"{'worker':14} {'pid':>7}  status      tail")
    for name, info in state.items():
        alive = running(int(info["pid"]))
        tail = ""
        path = log_path(name)
        if path.exists():
            lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            tail = lines[-1][:60] if lines else ""
        print(f"{name:14} {info['pid']:>7}  {'RUNNING' if alive else 'finished':<10}  {tail}")


def launch_raw(name: str, args: list[str]) -> int:
    """Launch an arbitrary script as a tracked worker."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = log_path(name).open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, *args], cwd=str(ROOT), stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    state[name] = {"pid": process.pid, "args": args,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"launched {name:14} pid={process.pid:<7} {' '.join(args)}")
    return process.pid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group", choices=["horizons", "ablation", "seeds", "regimes",
                                          "cross_site", "explain"])
    parser.add_argument("items", nargs="*")
    parser.add_argument("--status", action="store_true", dest="show")
    args = parser.parse_args()

    if args.show:
        show_status()
        return 0

    if args.group == "horizons":
        for horizon in args.items:
            launch(f"h_{horizon}", ["--only", "C_horizons", "--horizons", horizon])
    elif args.group == "ablation":
        launch("ablation", ["--only", "E_feature_ablation"])
    elif args.group == "regimes":
        # Additive input regimes, split by model so several can run at once.
        for model in args.items:
            launch(f"reg_{model}", ["--only", "N_feature_regimes", "--models", model,
                                    "--skip-pipeline"])
    elif args.group == "cross_site":
        launch("cross_site", ["scripts/run_cross_site.py",
                              "--models", *(args.items or ["xgboost"]),
                              "--horizons", "1h", "--seeds", "42"])
    elif args.group == "explain":
        launch("explain", ["scripts/evaluate.py", "--explainability", "--uncertainty",
                           "--report"])
    else:
        for seed in args.items:
            launch(f"seed_{seed}", ["--only", "M_multiseed", "--horizons", "1h",
                                    "--seeds", seed])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
