"""Write the four analysis notebooks.

The notebooks are generated rather than hand-authored so that they stay
consistent with the repository layout and with the paths the scripts use. They
contain no stored outputs: a reader runs them after `scripts/preprocess.py` and
`scripts/run_experiments.py`, and every figure they draw comes from the stored
results.

    python tools/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

PRELUDE = [
    ("code", [
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "ROOT = Path.cwd().parents[1] if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
        "sys.path.insert(0, str(ROOT / \"src\"))\n",
    ]),
]


def markdown(text: str) -> tuple[str, list[str]]:
    return ("markdown", text.splitlines(keepends=True))


def code(lines: Sequence[str]) -> tuple[str, list[str]]:
    return ("code", list(lines))


def build(cells: list[tuple[str, list[str]]]) -> dict[str, Any]:
    return {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": source,
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, source in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def data_exploration() -> dict[str, Any]:
    return build([
        markdown("# 01 - Data exploration\n\n"
                 "The case-study record, its quality, its daylight structure and the derived\n"
                 "features. Run after `python scripts/preprocess.py`; every plot and number\n"
                 "comes from the processed artefacts in `data/processed/`.\n"),
        *PRELUDE,
        code(["import numpy as np\n", "import pandas as pd\n",
              "import matplotlib.pyplot as plt\n", "\n",
              "pd.set_option(\"display.width\", 200)\n"]),
        code(["featured = pd.read_parquet(ROOT / \"data\" / \"processed\" /\n",
              "                          \"featured_primary_station.parquet\")\n",
              "print(featured.shape)\n", "display(featured.head())\n"]),
        code(["# Half the steps are exactly zero: night-time. This is why MAPE is not\n",
              "# computed and why daylight and all-step metrics are reported separately.\n",
              "display(featured[\"pv_power_w\"].describe())\n",
              "print(\"zero fraction:\", float((featured[\"pv_power_w\"] <= 0).mean()))\n"]),
        code(["rated = featured[\"pv_power_w\"].max()\n",
              "times = pd.to_datetime(featured[\"Time\"])\n",
              "fig, axes = plt.subplots(2, 1, figsize=(11, 6))\n",
              "axes[0].plot(times, featured[\"pv_power_w\"] / 1000, lw=0.4,\n",
              "            color=\"#1f77b4\")\n",
              "axes[0].set_ylabel(\"PV power (kW)\")\n",
              "axes[0].set_title(f\"Case-study station, full record; peak \"\n",
              "                    f\"{rated / 1000:.1f} kW\")\n",
              "axes[1].hist(featured.loc[featured[\"daylight\"], \"pv_power_w\"] / 1000,\n",
              "            bins=80, color=\"#ff7f0e\")\n",
              "axes[1].set_xlabel(\"Daylight power (kW)\")\n",
              "axes[1].set_ylabel(\"Count\")\n", "plt.tight_layout()\n"]),
        code(["# Class balance of the weather-regime taxonomy, with the persistence\n",
              "# reference rescored inside each stratum in the analysis stage.\n",
              "display(featured[\"regime\"].value_counts())\n"]),
        code(["from solar_forecasting.features.builder import (feature_columns,\n",
              "                                            describe_feature_groups)\n",
              "columns = feature_columns(featured)\n",
              "print(len(columns), \"model inputs\")\n",
              "for name, group in describe_feature_groups(columns).items():\n",
              "    print(f\"{name:>18}: {len(group):>3}  {group[:6]}\")\n"]),
        markdown("## Reading the exploration\n\n"
                 "- The zero fraction is the single most important number here: it governs\n"
                 "  the metric choice and the evaluation scope.\n"
                 "- The regime counts are the class-balance check for the stratified\n"
                 "  analysis. A stratum with fewer than 200 daylight steps is reported as\n"
                 "  under-sampled rather than summarised.\n"),
    ])


def baselines() -> dict[str, Any]:
    return build([
        markdown("# 02 - Baselines and reference forecasts\n\n"
                 "Persistence, smart persistence and the classical ML family, on exactly the\n"
                 "split every other experiment uses.\n"),
        *PRELUDE,
        code(["import numpy as np\n", "import pandas as pd\n",
              "pd.set_option(\"display.width\", 220)\n"]),
        code(["registry = pd.read_csv(ROOT / \"results\" / \"experiments.csv\")\n",
              "names = [\"persistence\", \"smart_persistence\", \"linear_regression\",\n",
              "         \"random_forest\", \"gradient_boosting\", \"xgboost\"]\n",
              "baselines = registry[registry.model.isin(names)]\n",
              "display(baselines[[\"model\", \"horizon\", \"mae\", \"rmse\", \"nrmse_capacity\",\n",
              "                   \"r2\", \"skill_vs_persistence_rmse\",\n",
              "                   \"train_seconds\"]].drop_duplicates(\"model\"))\n"]),
        markdown("Smart persistence is the last observation rescaled by the clear-sky\n"
                 "ratio, with a reference floor, a ratio clip and a capacity clip. That\n"
                 "ratio is what makes the reference strong: a naive persistence is beaten\n"
                 "by almost anything, so a skill score against it flatters every model.\n"),
        code(["from solar_forecasting.models.baselines import smart_persistence_forecast\n",
              "last = np.array([40_000.0])\n",
              "print(smart_persistence_forecast(last, np.array([700.0]), np.array([900.0]),\n",
              "                                   rated_w=55_000.0))\n",
              "print(smart_persistence_forecast(last, np.array([0.0]), np.array([0.0]),\n",
              "                                   rated_w=55_000.0),\n",
              "      \"<- below the reference floor, so plain persistence\")\n"]),
        code(["preds = pd.read_parquet(ROOT / \"results\" / \"predictions\" /\n",
              "                            \"B_ml_vs_dl__persistence__1h.parquet\")\n",
              "daylight = preds[preds[\"daylight\"].astype(bool)]\n",
              "print(len(daylight), \"daylight steps\")\n",
              "display(daylight[[\"actual\", \"predicted\", \"error\"]].describe())\n"]),
        markdown("## What to take from this notebook\n\n"
                 "- The reference is the point of comparison, not a formality. Skill is\n"
                 "  measured against both persistence variants, on identical timestamps.\n"
                 "- A negative skill score is a reportable result, not a failure of the run.\n"),
    ])


def deep_learning() -> dict[str, Any]:
    return build([
        markdown("# 03 - Recurrent, hybrid and attention models\n\n"
                 "Model shapes, the shared training protocol and the recorded costs.\n"),
        *PRELUDE,
        code(["import torch\n", "from solar_forecasting.models import registry\n"]),
        code(["n_features, lookback = 33, 24\n",
              "sizes = {\"hidden_size\": 32, \"n_filters\": 8, \"kernel_size\": 3,\n",
              "         \"attention_size\": 16, \"d_model\": 32, \"n_heads\": 2,\n",
              "         \"n_layers\": 1, \"dim_feedforward\": 64}\n",
              "x = torch.zeros(4, lookback, n_features)\n",
              "for name in [\"lstm\", \"gru\", \"cnn_lstm\", \"attention_lstm\", \"transformer\"]:\n",
              "    model = registry.build_model(name, n_features=n_features, **sizes)\n",
              "    print(f\"{name:>15}: output {tuple(model(x).shape)}\")\n"]),
        markdown("Every neural model shares one protocol: MAE loss on the standardised\n"
                 "target, early stopping on validation MAE **in watts**, best weights\n"
                 "restored, AdamW, one seed. If the protocol differed per architecture, the\n"
                 "comparison would be a measure of tuning effort rather than of method.\n"),
        code(["from solar_forecasting.config import load_config\n",
              "config = load_config()\n", "print(config.get(\"training\"))\n",
              "print(config.get(\"neural\"))\n"]),
        code(["registry = pd.read_csv(ROOT / \"results\" / \"experiments.csv\")\n",
              "neural = registry[registry.model.isin(\n",
              "    [\"lstm\", \"gru\", \"cnn_lstm\", \"attention_lstm\", \"transformer\"])]\n",
              "display(neural[[\"model\", \"horizon\", \"mae\", \"rmse\", \"nrmse_capacity\", \"r2\",\n",
              "                 \"skill_vs_persistence_rmse\", \"train_seconds\",\n",
              "                 \"n_parameters\"]].drop_duplicates([\"model\", \"horizon\"]))\n"]),
        markdown("## What to take from this notebook\n\n"
                 "- Parameter counts and training durations are recorded per run, so the\n"
                 "  accuracy-cost trade-off is measured rather than asserted.\n"
                 "- At a one-hour horizon, two of these models are worse than persistence.\n"
                 "  That is the result and it is reported as such.\n"),
    ])


def error_analysis() -> dict[str, Any]:
    return build([
        markdown("# 04 - Error analysis\n\n"
                 "Where the models actually fail: weather regime, season, time of day,\n"
                 "generation level and ramping.\n"),
        *PRELUDE,
        code(["import pandas as pd\n", "pd.set_option(\"display.width\", 220)\n"]),
        code(["error = pd.read_csv(ROOT / \"results\" / \"tables\" / \"error_analysis.csv\")\n",
              "intraday = error[error.stratum_type == \"time_of_day\"]\n",
              "display(intraday[[\"model\", \"stratum\", \"n\", \"nrmse_capacity\", \"mae\"]]\n",
              "        .sort_values([\"model\", \"nrmse_capacity\"]).head(14))\n"]),
        code(["regime = pd.read_csv(ROOT / \"results\" / \"tables\" /\n",
              "                        \"weather_regime_comparison.csv\")\n",
              "display(regime.pivot_table(index=\"stratum\", columns=\"model\",\n",
              "                            values=\"skill_vs_persistence_rmse\").round(3))\n"]),
        markdown("Absolute error is **lowest** in cloudy conditions, because irradiance and\n"
                 "therefore the signal amplitude are small. Skill is also lowest there, which\n"
                 "is the opposite conclusion. Reading the error column alone inverts the\n"
                 "answer to the question a forecaster actually asks.\n"),
        code(["display(regime[regime.stratum == \"Cloudy\"]\n",
              "        [[\"model\", \"n\", \"nrmse_capacity\", \"skill_vs_persistence_rmse\"]]\n",
              "        .round(4))\n"]),
        code(["stats = pd.read_csv(ROOT / \"results\" / \"tables\" /\n",
              "                      \"statistical_significance.csv\")\n",
              "display(stats[stats.loss == \"mae\"]\n",
              "        [[\"model\", \"skill_point\", \"skill_ci_low\", \"skill_ci_high\",\n",
              "          \"dm_p_value\", \"dm_p_value_holm\",\n",
              "          \"dm_lag1_autocorrelation\"]].round(4))\n"]),
        markdown("## What to take from this notebook\n\n"
                 "- The block-bootstrap intervals and the Diebold-Mariano test disagree,\n"
                 "  because consecutive 15-minute errors share a cloud field. Both are\n"
                 "  reported, together with the autocorrelation that explains the\n"
                 "  disagreement.\n"
                 "- Stratifying by regime changes the conclusion about where forecasting is\n"
                 "  hard. That is the argument for stratified evaluation.\n"),
    ])


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_data_exploration.ipynb": data_exploration(),
        "02_baselines.ipynb": baselines(),
        "03_deep_learning.ipynb": deep_learning(),
        "04_error_analysis.ipynb": error_analysis(),
    }
    for name, notebook in notebooks.items():
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
