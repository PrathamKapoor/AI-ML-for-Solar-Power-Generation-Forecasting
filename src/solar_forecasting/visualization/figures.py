"""Publication-quality figures.

Every figure is written at 200 dpi to ``results/figures`` with a non-interactive
backend, so a headless run on a server produces the same files as a laptop.
Each function takes prepared data rather than reaching into the results
directory, which keeps the plotting testable and lets the report regenerate a
single figure without rerunning an experiment.

Conventions applied throughout:

* a publication-style matplotlib setup applied once at import;
* the model colour map is defined once, so a model keeps the same colour in every
  figure and across the paper;
* daylight-only analysis is drawn with daylight data, and night-time steps are
  never silently averaged into a daytime statistic;
* error bars that are drawn are computed by the block bootstrap in
  :mod:`solar_forecasting.statistics.comparison`, never assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 200
FIGURE_FORMAT = "png"

#: One colour per model, reused in every figure and in the paper.
MODEL_COLORS: dict[str, str] = {
    "persistence": "#7f7f7f",
    "smart_persistence": "#4c4c4c",
    "linear_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "gradient_boosting": "#8bc34a",
    "xgboost": "#ff7f0e",
    "lstm": "#d62728",
    "gru": "#e377c2",
    "cnn_lstm": "#17becf",
    "attention_lstm": "#bcbd22",
    "transformer": "#9467bd",
}

PRETTY_NAMES: dict[str, str] = {
    "persistence": "Persistence",
    "smart_persistence": "Smart persistence",
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN-LSTM",
    "attention_lstm": "Attention-LSTM",
    "transformer": "Transformer",
}


def apply_style() -> None:
    """One consistent, print-safe matplotlib configuration."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.autolayout": False,
    })


apply_style()


def pretty(model: str) -> str:
    return PRETTY_NAMES.get(model, model.replace("_", " ").capitalize())


def color(model: str) -> str:
    return MODEL_COLORS.get(model, "#333333")


def figures_dir(config=None) -> Path:
    if config is not None:
        path = config.project_root_for("results") / "figures"
    else:
        path = Path(__file__).resolve().parents[3] / "results" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(fig: plt.Figure, name: str, directory: Path | None = None) -> Path:
    """Write one figure and close it, returning the path."""
    target = (directory or figures_dir()) / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(target)
    plt.close(fig)
    return target


def _tick_formatter(ax, unit: str = "") -> None:
    """Thousands separators on large watt axes."""
    try:
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:,.0f}{unit}")
    except Exception:  # pragma: no cover - defensive, formatter support varies
        pass


# ---------------------------------------------------------------------------
# 1. Actual versus predicted time series
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(predictions: Mapping[str, pd.DataFrame], model: str,
                             rated_w: float | None, name: str = "fig01_actual_vs_predicted",
                             window_hours: float = 168.0,
                             directory: Path | None = None) -> Path:
    """One week of daylight forecasts, with the observed series overlaid."""
    frame = predictions[model].copy()
    frame["Time"] = pd.to_datetime(frame["Time"])
    frame = frame[frame["daylight"].astype(bool)].sort_values("Time")
    start = frame["Time"].min()
    end = start + pd.Timedelta(hours=window_hours)
    window = frame[(frame["Time"] >= start) & (frame["Time"] < end)]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(window["Time"], window["actual"] / 1000.0, color="#111111", lw=1.6,
            label="Observed")
    ax.plot(window["Time"], window["predicted"] / 1000.0, color=color(model), lw=1.3,
            ls="--", label=f"{pretty(model)} forecast")
    if "reference_persistence" in window.columns:
        ax.plot(window["Time"], window["reference_persistence"] / 1000.0,
                color="#aaaaaa", lw=1.0, ls=":", label="Persistence")
    if rated_w:
        ax.axhline(rated_w / 1000.0, color="#cc4444", lw=0.8, ls="-.", label="Rated capacity")
    ax.set_xlabel("Target time")
    ax.set_ylabel("AC power (kW)")
    ax.set_title(f"Observed versus {pretty(model).lower()} forecasts, first "
                 f"{window_hours / 24:.0f} days of the test year (daylight steps only)")
    ax.legend(ncol=4, loc="upper right")
    fig.autofmt_xdate()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 2. Forecast error over time
# ---------------------------------------------------------------------------
def plot_error_over_time(predictions: Mapping[str, pd.DataFrame], models: Sequence[str],
                         rated_w: float | None, name: str = "fig02_forecast_error_over_time",
                         window_days: int = 21,
                         directory: Path | None = None) -> Path:
    """Daily mean absolute error, to show when the models fail rather than how much."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    for model in models:
        frame = predictions[model].copy()
        frame["Time"] = pd.to_datetime(frame["Time"])
        frame = frame[frame["daylight"].astype(bool)].sort_values("Time")
        start = frame["Time"].min()
        window = frame[frame["Time"] < start + pd.Timedelta(days=window_days)]
        daily = window.set_index("Time")["abs_error"].resample("D").mean()
        axes[0].plot(daily.index, daily.to_numpy() / 1000.0, color=color(model), lw=1.0,
                     label=pretty(model))
        axes[1].plot(window["Time"], window["error"] / 1000.0, color=color(model), lw=0.6,
                     alpha=0.85)
    axes[0].set_ylabel("Daily MAE (kW)")
    axes[0].set_title(f"Forecast error over the first {window_days} days of the test year")
    axes[0].legend(ncol=3)
    axes[1].set_ylabel("Signed error (kW)")
    axes[1].set_xlabel("Target time")
    axes[1].axhline(0.0, color="#000000", lw=0.6)
    fig.autofmt_xdate()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 3-6. Aggregate model comparison
# ---------------------------------------------------------------------------
def plot_metric_comparison(table: pd.DataFrame, metric: str, title: str,
                           name: str, annotate: bool = True,
                           ci_columns: tuple[str | None, str | None] = (None, None),
                           directory: Path | None = None) -> Path:
    """Bar chart of one aggregate metric across models, ordered best first.

    For error metrics the bars are sorted ascending and capped, so a shorter bar
    is better and the axis is labelled accordingly; for R^2 they are sorted
    descending.
    """
    frame = table.dropna(subset=[metric]).copy()
    if frame.empty:
        raise ValueError(f"no rows with a finite {metric!r} to plot")
    higher_is_better = metric.startswith("r2")
    frame = frame.sort_values(metric, ascending=higher_is_better)
    labels = [pretty(m) for m in frame["model"]]
    values = frame[metric].to_numpy(dtype=float)
    colours = [color(m) for m in frame["model"]]

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(frame) + 1.4))
    positions = np.arange(len(frame))
    if ci_columns[0] and ci_columns[0] in frame.columns:
        low = frame[ci_columns[0]].to_numpy(dtype=float)
        high = frame[ci_columns[1]].to_numpy(dtype=float)
        errors = np.vstack([np.clip(values - low, 0, None), np.clip(high - values, 0, None)])
        ax.barh(positions, values, color=colours, xerr=errors, capsize=2.5,
                error_kw={"lw": 0.8, "ecolor": "#333333"})
    else:
        ax.barh(positions, values, color=colours)
    ax.set_yticks(positions, labels)
    ax.set_xlabel(_metric_label(metric))
    ax.set_title(title)
    ax.axvline(0.0, color="#000000", lw=0.7)
    if annotate:
        for position, value in zip(positions, values):
            fmt = "{:,.3f}" if abs(value) < 10 else "{:,.0f}"
            offset = 0.01 * (np.nanmax(np.abs(values)) or 1.0)
            ax.text(value + (offset if value >= 0 else -offset), position, fmt.format(value),
                    va="center", ha="left" if value >= 0 else "right", fontsize=7.5)
    return save(fig, name, directory)


def _metric_label(metric: str) -> str:
    return {
        "mae": "MAE (W)",
        "rmse": "RMSE (W)",
        "r2": "R$^2$",
        "nrmse_capacity": "nRMSE (fraction of rated capacity)",
        "smape": "sMAPE (%)",
        "skill_vs_persistence_rmse": "Skill against persistence (1 - RMSE ratio)",
        "skill_vs_smart_persistence_rmse": "Skill against smart persistence",
    }.get(metric, metric)


# ---------------------------------------------------------------------------
# 7. Horizon comparison
# ---------------------------------------------------------------------------
def plot_horizon_comparison(table: pd.DataFrame, name: str = "fig07_horizon_comparison",
                            directory: Path | None = None) -> Path:
    """Accuracy and skill against horizon, one line per model."""
    frame = table.dropna(subset=["rmse"]).copy()
    horizons = sorted(frame["horizon"].unique(),
                      key=lambda h: float(str(h).rstrip("hmin").rstrip("m") or 0))
    order = ["15min", "1h", "6h", "24h"]
    horizons = [h for h in order if h in set(frame["horizon"])] + \
               [h for h in horizons if h not in order]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for model, group in frame.groupby("model"):
        group = group.set_index("horizon").reindex(horizons).dropna(subset=["rmse"])
        positions = [horizons.index(h) for h in group.index]
        axes[0].plot(positions, group["rmse"], marker="o", ms=4, lw=1.2,
                     color=color(model), label=pretty(model))
        axes[1].plot(positions, group["nrmse_capacity"], marker="o", ms=4, lw=1.2,
                     color=color(model))
        if "skill_vs_persistence_rmse" in group.columns:
            axes[2].plot(positions, group["skill_vs_persistence_rmse"], marker="o", ms=4,
                         lw=1.2, color=color(model))
    for ax, title, ylabel in ((axes[0], "Error growth with horizon", "RMSE (W)"),
                             (axes[1], "Capacity-normalised error", "nRMSE"),
                             (axes[2], "Skill against persistence", "1 - RMSE ratio")):
        ax.set_xticks(range(len(horizons)), horizons, rotation=30)
        ax.set_title(title)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(ylabel)
    axes[0].legend(ncol=2)
    _tick_formatter(axes[0])
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 8. Weather-regime comparison
# ---------------------------------------------------------------------------
def plot_regime_comparison(regime_table: pd.DataFrame, models: Sequence[str] | None = None,
                           name: str = "fig08_weather_regime_comparison",
                           directory: Path | None = None) -> Path:
    """Model performance inside each weather regime, as grouped bars."""
    frame = regime_table.dropna(subset=["nrmse_capacity"]).copy()
    if models:
        frame = frame[frame["model"].isin(list(models))]
    regimes = [r for r in ["Clear", "Partly cloudy", "Cloudy", "High-variability"]
               if r in set(frame["stratum"])]
    models = list(dict.fromkeys(frame["model"]))
    if not regimes or not models:
        raise ValueError("regime table has no plottable daylight strata")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(regimes))
    for i, model in enumerate(models):
        subset = frame[frame["model"] == model].set_index("stratum").reindex(regimes)
        offset = positions - 0.4 + width * (i + 0.5)
        axes[0].bar(offset, subset["nrmse_capacity"], width=width, color=color(model),
                    label=pretty(model))
        axes[1].bar(offset, subset["skill_vs_persistence_rmse"], width=width,
                    color=color(model))
    axes[0].set_xticks(positions, [r.replace(" ", "\n") for r in regimes])
    axes[0].set_ylabel("nRMSE (fraction of capacity)")
    axes[0].set_title("Error by weather regime")
    axes[1].set_xticks(positions, [r.replace(" ", "\n") for r in regimes])
    axes[1].set_ylabel("Skill against persistence")
    axes[1].set_title("Skill by weather regime")
    axes[1].axhline(0.0, color="#000000", lw=0.7)
    axes[0].legend(ncol=2)
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 9. Seasonal comparison
# ---------------------------------------------------------------------------
def plot_seasonal_comparison(season_table: pd.DataFrame,
                             name: str = "fig09_seasonal_comparison",
                             directory: Path | None = None) -> Path:
    """Seasonal error and skill, one panel per season."""
    frame = season_table.dropna(subset=["nrmse_capacity"]).copy()
    models = list(dict.fromkeys(frame["model"]))
    seasons = [s for s in ["Winter", "Spring", "Summer", "Autumn"] if s in set(frame["stratum"])]
    if not seasons or not models:
        raise ValueError("season table has no plottable daylight strata")

    fig, axes = plt.subplots(1, len(seasons), figsize=(2.6 * len(seasons), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    positions = np.arange(len(models))
    for ax, season in zip(axes, seasons):
        subset = frame[frame["stratum"] == season].set_index("model").reindex(models)
        ax.barh(positions, subset["nrmse_capacity"].to_numpy(dtype=float),
                color=[color(m) for m in models])
        ax.set_yticks(positions, [pretty(m) for m in models] if ax is axes[0] else [])
        ax.set_title(season)
        ax.set_xlabel("nRMSE")
    axes[0].set_ylabel("")
    fig.suptitle("Generalisation across seasons (test year, daylight steps)", y=1.02)
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 10. Feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(importance: pd.DataFrame, top_n: int = 15,
                            value_column: str = "importance",
                            error_column: str | None = None,
                            title: str = "Feature importance",
                            name: str = "fig10_feature_importance",
                            directory: Path | None = None) -> Path:
    """Horizontal importance ranking with an optional uncertainty bar."""
    frame = importance.dropna(subset=[value_column]).head(int(top_n)).iloc[::-1]
    if frame.empty:
        raise ValueError("no finite importances to plot")
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(frame) + 1.2))
    positions = np.arange(len(frame))
    values = frame[value_column].to_numpy(dtype=float)
    if error_column and error_column in frame.columns:
        ax.barh(positions, values, xerr=frame[error_column].to_numpy(dtype=float),
                color="#ff7f0e", capsize=2.5, error_kw={"lw": 0.8, "ecolor": "#333333"})
    else:
        ax.barh(positions, values, color="#ff7f0e")
    ax.set_yticks(positions, frame["feature"])
    ax.set_xlabel(_importance_label(value_column))
    ax.set_title(title)
    return save(fig, name, directory)


def _importance_label(value_column: str) -> str:
    return {
        "importance": "Mean |SHAP| (normalised)",
        "mean_mae_increase_w": "Increase in MAE when permuted (W)",
        "relative_importance": "Relative importance",
    }.get(value_column, value_column)


def plot_importance_agreement(tables: Mapping[str, pd.DataFrame],
                              value_column: str = "relative_importance",
                              top_n: int = 12,
                              name: str = "fig10b_importance_method_agreement",
                              directory: Path | None = None) -> Path:
    """Two importance methods side by side, to expose where they disagree."""
    usable = {k: v.dropna(subset=[value_column]).head(int(top_n))
              for k, v in tables.items() if v is not None and not v.empty}
    if not usable:
        raise ValueError("no importance tables to compare")
    fig, axes = plt.subplots(1, len(usable), figsize=(5.0 * len(usable), 0.32 * top_n + 1.4))
    axes = np.atleast_1d(axes)
    for ax, (label, frame) in zip(axes, usable.items()):
        ordered = frame.iloc[::-1]
        ax.barh(np.arange(len(ordered)), ordered[value_column].to_numpy(dtype=float),
                color="#4c72b0")
        ax.set_yticks(np.arange(len(ordered)), ordered["feature"], fontsize=7)
        ax.set_title(label)
        ax.set_xlabel(_importance_label(value_column))
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 11. Residual distribution
# ---------------------------------------------------------------------------
def plot_residual_distribution(predictions: Mapping[str, pd.DataFrame],
                               models: Sequence[str], rated_w: float | None,
                               name: str = "fig11_residual_distribution",
                               directory: Path | None = None) -> Path:
    """Normalised residual histograms against the normal density."""
    from scipy import stats as _stats

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    scale = rated_w or 1.0
    for model in models:
        frame = predictions[model]
        daylight = frame[frame["daylight"].astype(bool)]
        residual = (daylight["actual"].to_numpy(dtype=float)
                    - daylight["predicted"].to_numpy(dtype=float)) / scale
        if residual.size < 10:
            continue
        ax.hist(residual, bins=60, histtype="step", density=True, lw=1.2,
                color=color(model), label=pretty(model))
    x = np.linspace(-0.5, 0.5, 200)
    ax.plot(x, _stats.norm.pdf(x), color="#000000", lw=1.0, ls="--", label="Standard normal")
    ax.set_xlabel("Residual / rated capacity")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of normalised daylight residuals")
    ax.legend(ncol=2)
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 12. Predicted versus actual scatter
# ---------------------------------------------------------------------------
def plot_predicted_vs_actual(predictions: Mapping[str, pd.DataFrame], models: Sequence[str],
                             rated_w: float | None,
                             name: str = "fig12_predicted_vs_actual_scatter",
                             directory: Path | None = None) -> Path:
    """Parity plots against the rated-capacity line."""
    scale = (rated_w or float(
        max(p["actual"].max() for p in predictions.values()))) / 1000.0
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models), 3.0), sharex=True,
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        frame = predictions[model]
        daylight = frame[frame["daylight"].astype(bool)]
        actual = daylight["actual"].to_numpy(dtype=float) / scale
        predicted = daylight["predicted"].to_numpy(dtype=float) / scale
        ax.hexbin(actual, predicted, gridsize=45, bins="log", cmap="viridis", mincnt=1)
        limit = max(float(np.nanmax(actual)), float(np.nanmax(predicted))) * 1.02
        ax.plot([0, limit], [0, limit], color="#cc4444", lw=1.0, ls="--")
        correlation = float(np.corrcoef(actual, predicted)[0, 1]) if actual.size > 2 else float("nan")
        ax.set_title(f"{pretty(model)}\nr = {correlation:.4f}", fontsize=8.5)
        ax.set_xlabel("Observed (kW)")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
    axes[0].set_ylabel("Forecast (kW)")
    fig.suptitle("Forecast versus observed, daylight steps of the test year", y=1.02)
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 13. Persistence skill
# ---------------------------------------------------------------------------
def plot_skill_comparison(table: pd.DataFrame,
                          column: str = "skill_vs_persistence_rmse",
                          name: str = "fig13_persistence_skill",
                          ci: Mapping[str, tuple[float, float]] | None = None,
                          directory: Path | None = None) -> Path:
    """Skill against both persistence references, with bootstrap intervals."""
    frame = table.dropna(subset=[column]).copy().sort_values(column)
    if frame.empty:
        raise ValueError(f"no rows with a finite {column!r} to plot")
    positions = np.arange(len(frame))
    values = frame[column].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 0.4 * len(frame) + 1.4))
    ax.barh(positions, values, color=[color(m) for m in frame["model"]])
    if ci:
        for position, (low, high) in zip(positions, ci.values()):
            if np.isfinite(low) and np.isfinite(high):
                ax.plot([low, high], [position, position], color="#222222", lw=1.2)
    ax.set_yticks(positions, [pretty(m) for m in frame["model"]])
    ax.axvline(0.0, color="#000000", lw=0.8)
    ax.set_xlabel("Skill against persistence, 1 - RMSE(model) / RMSE(persistence)")
    ax.set_title("Forecast skill relative to the persistence reference\n"
                 "(bars: moving-block bootstrap 95% interval where drawn)")
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 14. Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(histories: Mapping[str, Sequence[Mapping[str, float]]],
                         name: str = "fig14_training_curves",
                         directory: Path | None = None) -> Path:
    """Validation and training loss per epoch, with the early-stopping point marked."""
    usable = {k: list(v) for k, v in histories.items() if v}
    if not usable:
        raise ValueError("no training histories to plot")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for model, history in usable.items():
        epochs = [row.get("epoch", i + 1) for i, row in enumerate(history)]
        for ax, key, style, label in ((axes[0], "train_loss", "-", "training"),
                                      (axes[1], "val_loss_w", "--", "validation")):
            if key not in history[0]:
                continue
            values = [row.get(key, np.nan) for row in history]
            ax.plot(epochs, values, style, lw=1.2, color=color(model),
                    label=f"{pretty(model)} {label}")
        best = history[0].get("best_epoch")
        if best is not None and "val_loss_w" in history[0]:
            best_value = history[min(int(best) - 1, len(history) - 1)].get("val_loss_w")
            if best_value is not None:
                axes[1].scatter([best], [best_value], color=color(model), s=18, zorder=5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (standardised target, MAE)")
    axes[0].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation MAE (W)")
    axes[1].set_title("Validation loss (markers: restored best epoch)")
    axes[0].legend(ncol=2)
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 15. Error distribution by model
# ---------------------------------------------------------------------------
def plot_error_distribution_by_model(predictions: Mapping[str, pd.DataFrame],
                                     models: Sequence[str], rated_w: float | None,
                                     name: str = "fig15_error_distribution_by_model",
                                     directory: Path | None = None) -> Path:
    """Box and violin view of the absolute error distribution, scaled by capacity."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    data, labels = [], []
    for model in models:
        frame = predictions[model]
        daylight = frame[frame["daylight"].astype(bool)]
        if len(daylight) < 10:
            continue
        data.append(daylight["abs_error"].to_numpy(dtype=float) / (rated_w or 1.0))
        labels.append(pretty(model))
    if not data:
        raise ValueError("no models with enough daylight predictions to plot")
    axes[0].boxplot(data, tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("|Error| / rated capacity")
    axes[0].set_title("Absolute error distribution (daylight)")
    axes[0].tick_params(axis="x", rotation=30)
    positions = np.arange(len(labels))
    for position, values in zip(positions, data):
        axes[1].plot(np.sort(values), np.linspace(0, 1, len(values)), lw=1.3,
                     color=color([k for k, v in PRETTY_NAMES.items() if v == labels[position]][0]
                                 if any(v == labels[position] for v in PRETTY_NAMES.values())
                                 else ""))
    axes[1].set_xlabel("|Error| / rated capacity")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("Error exceedance curves")
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 16. Intraday error profile (mandatory error analysis)
# ---------------------------------------------------------------------------
def plot_intraday_error(stratified: pd.DataFrame, model: str, rated_w: float | None,
                        name: str = "fig16_intraday_error_profile",
                        directory: Path | None = None) -> Path:
    """MAE and nRMSE against time of day, to locate the morning and evening ramps."""
    frame = stratified[(stratified["model"] == model)
                       & (stratified["stratum_type"] == "time_of_day")]
    if frame.empty:
        raise ValueError(f"no time-of-day strata for model {model!r}")
    frame = frame.dropna(subset=["mae"]).sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(frame) + 1.4))
    positions = np.arange(len(frame))
    ax.barh(positions, frame["nrmse_capacity"], color=color(model))
    ax.set_yticks(positions, frame["stratum"])
    ax.set_xlabel("nRMSE (fraction of rated capacity)")
    ax.set_title(f"Where {pretty(model).lower()} fails: error by time of day (daylight bands)")
    for position, (n, mae) in enumerate(zip(frame["n"], frame["mae"])):
        ax.text(frame["nrmse_capacity"].iloc[position] * 1.01, position,
                f"MAE {mae:,.0f} W, n = {int(n):,}", va="center", fontsize=7.5)
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 17. Cost versus accuracy
# ---------------------------------------------------------------------------
def plot_cost_accuracy(cost: pd.DataFrame, metric: str = "nrmse_capacity",
                       name: str = "fig17_cost_accuracy",
                       directory: Path | None = None) -> Path:
    """Accuracy against training cost, with the Pareto frontier drawn."""
    frame = cost.dropna(subset=[metric, "train_seconds"]).copy()
    if frame.empty:
        raise ValueError("cost table has no plottable rows")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for _, row in frame.iterrows():
        ax.scatter(max(row["train_seconds"], 1e-2), row[metric], s=60,
                   color=color(row["model"]), zorder=3)
        ax.annotate(pretty(row["model"]), (max(row["train_seconds"], 1e-2), row[metric]),
                    textcoords="offset points", xytext=(6, 3), fontsize=7.5)
    finite = frame[frame[metric].notna()].sort_values("train_seconds")
    if len(finite) > 1:
        pareto = [finite.iloc[0]]
        best = finite.iloc[0][metric]
        for _, row in finite.iloc[1:].iterrows():
            if row[metric] < best:
                pareto.append(row)
                best = row[metric]
        ax.plot([max(r["train_seconds"], 1e-2) for r in pareto],
                [r[metric] for r in pareto], ls="--", color="#666666", lw=1.0,
                label="Pareto frontier")
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale; rule-based models at the left edge)")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title("Accuracy against training cost")
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 18. Conformal interval behaviour
# ---------------------------------------------------------------------------
def plot_intervals(intervals: pd.DataFrame, rated_w: float | None,
                    name: str = "fig18_conformal_intervals",
                    directory: Path | None = None) -> Path:
    """Predicted series with conformal bounds, and the miss rate through the year."""
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.0), sharex=False)
    window = intervals.head(24 * 20)
    scale = (rated_w or 1.0) / 1000.0
    axes[0].fill_between(window["Time"], window["lower"] / scale, window["upper"] / scale,
                         color="#ff7f0e", alpha=0.22, label="90% conformal interval")
    axes[0].plot(window["Time"], window["actual"] / scale, color="#111111", lw=1.0,
                 label="Observed")
    axes[0].plot(window["Time"], window["predicted"] / scale, color="#1f77b4", lw=1.0,
                 ls="--", label="Forecast")
    axes[0].set_ylabel("AC power (kW)")
    axes[0].set_title("Split-conformal prediction intervals (first 20 days after calibration)")
    axes[0].legend(ncol=3)
    monthly = intervals.set_index("Time")["inside"].resample("ME").mean()
    axes[1].plot(monthly.index, monthly.to_numpy() * 100.0, marker="o", color="#1f77b4")
    axes[1].axhline(90.0, color="#cc4444", ls="--", lw=1.0, label="Nominal 90%")
    axes[1].set_ylabel("Empirical coverage (%)")
    axes[1].set_xlabel("Target time")
    axes[1].set_title("Coverage through the evaluation period")
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 19. Dataset overview
# ---------------------------------------------------------------------------
def plot_dataset_overview(featured: pd.DataFrame, rated_w: float | None,
                          name: str = "fig19_dataset_overview",
                          directory: Path | None = None) -> Path:
    """Raw power, irradiance and a sample week: what the model is asked to predict."""
    frame = featured.copy()
    frame["Time"] = pd.to_datetime(frame["Time"])
    scale = (rated_w or 1.0) / 1000.0
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.4), sharex=False)
    axes[0].plot(frame["Time"], frame["pv_power_w"] / scale, lw=0.35, color="#1f77b4")
    axes[0].set_ylabel("PV power (kW)")
    axes[0].set_title("Case-study station: measured AC power, full record")
    if "ghi" in frame.columns:
        axes[1].plot(frame["Time"], frame["ghi"], lw=0.35, color="#ff7f0e")
    axes[1].set_ylabel("GHI (W/m$^2$)")
    axes[1].set_title("Global horizontal irradiance (15-minute means of the 1-minute record)")
    week = frame.head(24 * 7)
    axes[2].plot(week["Time"], week["pv_power_w"] / scale, lw=1.0, color="#1f77b4",
                 label="PV power")
    if "ghi_clear" in week.columns:
        axes[2].plot(week["Time"], week["ghi_clear"] / scale, lw=1.0, ls="--",
                     color="#888888", label="Clear-sky reference (scaled by capacity)")
    axes[2].set_ylabel("Power (kW)")
    axes[2].set_title("First week in detail")
    axes[2].legend(ncol=2)
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return save(fig, name, directory)


# ---------------------------------------------------------------------------
# 20. Ablation
# ---------------------------------------------------------------------------
def plot_ablation(ablation: pd.DataFrame, metric: str = "nrmse_capacity",
                  reference_spec: str = "full",
                  name: str = "fig20_feature_ablation",
                  directory: Path | None = None) -> Path:
    """Degradation relative to the full input set, per model."""
    frame = ablation.dropna(subset=[metric]).copy()
    if frame.empty:
        raise ValueError("ablation table has no plottable rows")
    models = list(dict.fromkeys(frame["model"]))
    specs = [s for s in frame["feature_spec"].unique() if s != reference_spec]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(specs) + 1.6))
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(specs))
    for i, model in enumerate(models):
        subset = (frame[frame["model"] == model]
                  .set_index("feature_spec").reindex([reference_spec] + specs))
        full_value = subset[metric].iloc[0] if len(subset) else np.nan
        degradation = 100.0 * (subset[metric].to_numpy(dtype=float)[1:] - full_value) / full_value
        ax.bar(positions + (i - (len(models) - 1) / 2) * width, degradation, width=width,
               color=color(model), label=pretty(model))
    ax.axhline(0.0, color="#000000", lw=0.8)
    ax.set_xticks(positions, [s.replace("_", "\n") for s in specs])
    ax.set_ylabel(f"Change in {metric} versus the full input set (%)")
    ax.set_title("Feature-group ablation: what each input group is worth")
    ax.legend(ncol=len(models))
    fig.tight_layout()
    return save(fig, name, directory)
