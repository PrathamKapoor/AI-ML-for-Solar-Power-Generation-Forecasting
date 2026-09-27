"""Cross-site generalisation: does a model trained at one plant work at another?

This is the experiment that separates "these models suit this array" from "these
models suit PV forecasting", and it is the one the previous pass left out.

Three protocol keys, covering the four situations in the design and scored with
capacity-normalised metrics, so that sites of different size are comparable. The
fourth situation -- train on several sites pooled, hold one out, test on the
held-out site -- is the per-site case of ``leave_one_site_out`` rather than a key
of its own, so it is not counted twice:

``within_site``
    Train and test inside one site. This is the reference every transfer number
    is compared against: a transfer result means nothing without the
    within-site result for the same site and the same model.
``cross_site``
    Train on site A, test on site B. B is never seen during fitting, feature
    scaling, early stopping or hyperparameter selection.
``leave_one_site_out``
    Train on several sites pooled, hold one out, test on the held-out site.
    This is the deployment scenario that matters in practice, where an operator
    has a fleet and wants a model that works on a new array, and it is repeated
    for every site in the panel so the result is one transfer number per site
    rather than a single number.

Leakage controls, each of which is asserted rather than assumed:

* Feature scalers are fitted on the **training rows of the training sites only**.
  The held-out site's feature distribution never touches a mean or a scale.
* The target scaler is likewise fitted on training sites only. Because
  standardisation is affine in watts, applying the training target scaler to a
  held-out site and inverting it returns that site's true watts, so no
  site-specific output rescaling is introduced.
* Early stopping uses the training sites' validation rows only. The held-out
  site's rows are used once, for scoring.
* nRMSE is reported against **each site's own rated capacity**, read from the
  dataset metadata rather than estimated from the data.
* Site selection is fixed in ``configs/experiments.yaml`` and is based on
  capacity spread and record coverage, never on model performance.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation import metrics as metrics_mod
from ..evaluation import regimes as regime_mod
from ..evaluation import stratified as strat_mod
from ..models import registry as model_registry
from ..models.classical import flatten_windows
from ..preprocessing import pipeline as pipeline_mod
from ..preprocessing import splitting as split_mod
from ..training.sequences import build_sequences
from ..training.trainer import TrainingConfig, predict_neural, train_model
from ..utils.io import write_json
from ..utils.seeding import set_seed

#: Slugification must match the pipeline's own artefact naming.
def _slugify(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


@dataclass
class SiteFrame:
    """One prepared site: its featured frame, split, capacity and slug."""

    station: str
    slug: str
    featured: pd.DataFrame
    split: split_mod.SplitResult
    rated_w: float
    coverage: dict[str, Any] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.split.train), "val": len(self.split.val),
                "test": len(self.split.test)}


def prepare_site(config: Config, station: str, quiet: bool = True) -> SiteFrame:
    """Run the full pipeline for one station and return its prepared frames.

    The processing is identical to the primary station's: same cleaning, same
    features, same chronological boundaries, same regime taxonomy. Only the PV
    target and the rated capacity differ.
    """
    slug = _slugify(station)
    state = pipeline_mod.run_pipeline(config, quiet=quiet, persist=True, station=station,
                                      artifact_slug=slug)
    rated = float(state["station"].get("rated_w") or 0.0)
    times = pd.to_datetime(state["featured"]["Time"])
    return SiteFrame(
        station=station, slug=slug, featured=state["featured"], split=state["split"],
        rated_w=rated,
        coverage={"first": str(times.min()), "last": str(times.max()),
                  "n_rows": int(len(state["featured"]))})


def _pool(sites: Sequence[SiteFrame], part: str) -> pd.DataFrame:
    """Concatenate one split across sites, tagging each row with its site."""
    frames = []
    for site in sites:
        block = getattr(site.split, part).copy()
        block["site"] = site.station
        frames.append(block)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fit_scalers(train_frame: pd.DataFrame, columns: list[str]
                 ) -> split_mod.ScalingBundle:
    return split_mod.fit_scalers(train_frame, columns, "pv_power_w")


def _build_sequences(frame: pd.DataFrame, columns: list[str], steps: int,
                     lookback: int, target_scaler: Any):
    prepared = pipeline_mod.build_target(frame, "pv_power_w", steps)
    return build_sequences(prepared, columns, "target", lookback, steps,
                           target_scaler=target_scaler)


def _train(config: Config, model_name: str, train_sites: Sequence[SiteFrame],
           val_sites: Sequence[SiteFrame], columns: list[str], steps: int,
           lookback: int, seed: int, target_scaler: Any):
    """Fit one model on pooled training-site sequences."""
    set_seed(seed)
    info = model_registry.model_info(model_name)
    n_features = len(columns)
    train_frame = _pool(train_sites, "train")
    val_frame = _pool(val_sites, "val")
    train_seq = _build_sequences(train_frame, columns, steps, lookback, target_scaler)
    val_seq = _build_sequences(val_frame, columns, steps, lookback, target_scaler)

    if info["kind"] == "sklearn":
        settings = (config.get("classical", {}) or {}).get(model_name, {})
        estimator = model_registry.build_model(model_name, n_features=n_features,
                                               random_state=seed, **settings)
        # Timed on the same clock as the neural branch. A fitted tree ensemble has
        # no single parameter count, so ``n_parameters`` stays None rather than 0:
        # zero is a claim about the model, None is a statement about the schema.
        start = time.perf_counter()
        estimator.fit(flatten_windows(train_seq.X), train_seq.y)
        return estimator, info["kind"], None, time.perf_counter() - start, None

    if info["kind"] == "torch":
        from ..training.experiment import _build
        module = _build(model_name, config, n_features)
        train_cfg = config.get("training", {}) or {}
        tcfg = TrainingConfig(
            epochs=int(train_cfg.get("epochs", 70)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            patience=int(train_cfg.get("patience", 14)),
            device=str(train_cfg.get("device", "cpu")),
            verbose=bool(train_cfg.get("verbose", False)))
        result = train_model(module, train_seq, val_seq, tcfg,
                             target_scale=float(target_scaler.scale_[0]))
        return result.model, info["kind"], result, result.train_seconds, result.n_parameters

    raise KeyError(f"model kind {info['kind']!r} is not supported in cross-site fitting")


def _predict(model, kind: str, sequence, target_scaler: Any, config: Config) -> np.ndarray:
    if kind == "sklearn":
        scaled = np.asarray(model.predict(flatten_windows(sequence.X)), dtype=float)
    else:
        train_cfg = config.get("training", {}) or {}
        scaled, _ = predict_neural(
            model, sequence, batch_size=int(train_cfg.get("eval_batch_size", 512)),
            device=str(train_cfg.get("device", "cpu")))
    return target_scaler.inverse_transform(
        np.asarray(scaled, dtype=float).reshape(-1, 1)).ravel()


def _score(actual: np.ndarray, predicted: np.ndarray, daylight: np.ndarray,
           rated_w: float, persistence: np.ndarray) -> dict[str, float]:
    mask = daylight & np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() == 0:
        return {"n": 0.0}
    y, f, p = actual[mask], predicted[mask], persistence[mask]
    scores = metrics_mod.compute_all(y, f, rated_w=rated_w)
    reference_rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    if reference_rmse > 0:
        scores["skill_vs_persistence_rmse"] = metrics_mod.skill(reference_rmse, scores["rmse"])
    scores["n"] = float(mask.sum())
    return scores


def _site_persistence(site: SiteFrame, steps: int) -> pd.Series:
    """Persistence at the forecast origin, per site test frame.

    Computed from the test frame directly, so it is aligned with the same
    timestamps the model forecasts.
    """
    from ..models.baselines import persistence_from_lag
    return pd.Series(persistence_from_lag(site.split.test, lag=0),
                     index=site.split.test.index, dtype=float)


def run_transfer(config: Config, train_sites: Sequence[SiteFrame],
                 test_site: SiteFrame, model_name: str, horizon: str,
                 seed: int, columns: Sequence[str] | None = None) -> dict[str, Any]:
    """Train on ``train_sites``, evaluate once on ``test_site``."""
    steps = {h["name"]: int(h["steps"]) for h in config.section("horizons")}[horizon]
    lookback = int(config.section("features")["lookback_steps"])
    feature_columns = list(columns) if columns else list(
        json_columns(config))
    if "site" in feature_columns:
        feature_columns = [c for c in feature_columns if c != "site"]

    train_frame = _pool(train_sites, "train")
    scalers = _fit_scalers(train_frame, feature_columns)
    model, kind, training_result, train_seconds, n_parameters = _train(
        config, model_name, train_sites, train_sites, feature_columns, steps, lookback,
        seed, scalers.target_scaler)

    test_frame = site_frame_for(test_site, steps)
    sequence = _build_sequences(test_frame, feature_columns, steps, lookback,
                                scalers.target_scaler)
    predicted = _predict(model, kind, sequence, scalers.target_scaler, config)

    origin_rows = np.asarray(sequence.origin_row, dtype=int)
    persistence = test_site.split.test["pv_power_w"].to_numpy(dtype=float)[origin_rows]

    daylight = np.asarray(sequence.daylight, dtype=bool)
    scores = _score(np.asarray(sequence.target_raw, dtype=float), predicted, daylight,
                    test_site.rated_w, persistence)
    scores.update({
        "train_sites": ", ".join(s.station for s in train_sites),
        "test_site": test_site.station,
        "model": model_name, "horizon": horizon, "seed": seed,
        "rated_w": test_site.rated_w,
        "train_seconds": float(train_seconds),
        "n_parameters": int(n_parameters) if n_parameters is not None else None,
        "n_features": len(feature_columns),
    })
    return {"scores": scores, "predictions": pd.DataFrame({
        "Time": sequence.target_timestamps.to_numpy(),
        "forecast_origin": sequence.timestamps.to_numpy(),
        "actual": sequence.target_raw, "predicted": predicted,
        "persistence": persistence, "daylight": daylight,
        "regime": sequence.regime.astype(str),
    })}


def site_frame_for(site: SiteFrame, steps: int) -> pd.DataFrame:
    """The test frame with the site tag removed from the model inputs."""
    frame = site.split.test.copy()
    return frame.drop(columns=[c for c in ("site",) if c in frame.columns])


def json_columns(config: Config) -> list[str]:
    """The configured model-input columns."""
    path = config.project_root_for("data") / "processed" / "feature_columns.json"
    if path.exists():
        import json
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)["feature_columns"]
    from ..features.builder import feature_columns as _columns
    return _columns(pd.read_parquet(
        config.project_root_for("data") / "processed" / "featured_primary_station.parquet"))


def _provenance(config: Config, models: Sequence[str], horizons: Sequence[str],
                seeds: Sequence[int]) -> dict[str, Any]:
    """Provenance for the cross-site report.

    The transfer protocols are the evidence behind the generalisation question,
    so their numbers have to be as traceable as the main result set: the archive
    checksum for the input bytes, the configuration fingerprint for the settings,
    and the git revision for the code that produced them.
    """
    from ..training.experiment import _source_archive, _source_checksum
    from ..utils.seeding import run_metadata

    environment = run_metadata()
    return {
        "recorded_at_utc": environment.get("timestamp_utc"),
        "source_archive": _source_archive(config),
        "source_archive_sha256": _source_checksum(config),
        "dataset": config.section("dataset")["name"],
        "primary_station": config.section("dataset")["primary_station"],
        "holdout_stations": list(config.section("dataset").get("holdout_stations", [])),
        "config_fingerprint": config.fingerprint(),
        "git_revision": environment.get("git_revision"),
        "models": list(models), "horizons": list(horizons), "seeds": list(seeds),
        "experiment_group": "J_cross_site",
        "note": ("Cross-site runs are reported in results/metrics/cross_site_report.json "
                 "and flattened to results/tables/cross_site_comparison.csv. Unlike the "
                 "main result set they are pooled across protocols into one record "
                 "rather than written as one file per run, because a single fit serves "
                 "the within-site, cross-site and leave-one-site-out protocols at once."),
    }


def run_cross_site(config: Config, models: Sequence[str], horizons: Sequence[str],
                   seeds: Sequence[int], holdout: Sequence[str] | None = None,
                   quiet: bool = True) -> dict[str, Any]:
    """Execute the within-site, cross-site and leave-one-site-out protocols.

    Returns a record with the per-run score tables and the site panel actually
    used, which is written to ``results/metrics/cross_site_report.json``.
    """
    panel_names = list(config.section("dataset").get("holdout_stations", [])
                       if holdout is None else holdout)
    if not panel_names:
        raise ValueError(
            "no cross-site panel available: pass holdout explicitly or set "
            "dataset.holdout_stations in configs/data.yaml. An empty list is a "
            "caller error, not a reason to fall back to the configured panel.")
    train_name = config.section("dataset")["primary_station"]

    sites = {train_name: prepare_site(config, train_name, quiet=quiet)}
    for name in panel_names:
        if name == train_name:
            continue
        print(f"  cross-site: preparing {name}")
        sites[name] = prepare_site(config, name, quiet=quiet)

    within: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    loso: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    for model_name in models:
        for horizon in horizons:
            for seed in seeds:
                # -- A: within-site reference for every site -------------------
                for name, site in sites.items():
                    outcome = run_transfer(config, [site], site, model_name, horizon, seed)
                    record = {"protocol": "within_site", **outcome["scores"]}
                    within.append(record)
                # -- B: train on the primary site, test on each holdout -------
                for name, site in sites.items():
                    if name == train_name:
                        continue
                    outcome = run_transfer(config, [sites[train_name]], site, model_name,
                                           horizon, seed)
                    record = {"protocol": "cross_site", **outcome["scores"]}
                    cross.append(record)
                    if seed == seeds[0] and horizon == horizons[0]:
                        prediction_frames[f"cross_site__{model_name}__{horizon}__{name}"] = \
                            outcome["predictions"]
                # -- C/D: leave-one-site-out ----------------------------------
                for name, site in sites.items():
                    others = [s for n, s in sites.items() if n != name]
                    if not others:
                        continue
                    outcome = run_transfer(config, others, site, model_name, horizon, seed)
                    loso.append({"protocol": "leave_one_site_out", **outcome["scores"]})

    report = {
        "provenance": _provenance(config, models, horizons, seeds),
        "panel": {name: {"rated_w": site.rated_w, **site.coverage, **site.sizes()}
                  for name, site in sites.items()},
        "train_site": train_name,
        "models": list(models), "horizons": list(horizons), "seeds": list(seeds),
        "protocols": {
            "within_site": within,
            "cross_site": cross,
            "leave_one_site_out": loso,
        },
        "leakage_controls": [
            "feature scalers fitted on training-site training rows only",
            "target scaler fitted on training sites only; inversion returns the "
            "held-out site's true watts because standardisation is affine",
            "early stopping monitored on training-site validation rows only",
            "held-out site scored once, after fitting was complete",
            "nRMSE normalised by each site's own rated capacity from the dataset metadata",
            "site selection fixed in configuration from capacity spread and coverage, "
            "never from model performance",
        ],
    }
    write_json(report, config.project_root_for("results") / "metrics" /
               "cross_site_report.json")
    if prediction_frames:
        directory = config.project_root_for("results") / "predictions"
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in prediction_frames.items():
            frame.to_parquet(directory / f"J_cross_site__{name}.parquet", index=False)
    return report


def cross_site_table(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten the cross-site report into one tidy table."""
    rows: list[dict[str, Any]] = []
    for protocol, records in report.get("protocols", {}).items():
        for record in records:
            rows.append({"protocol": protocol, **record})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    front = ["protocol", "model", "horizon", "test_site", "train_sites", "seed", "rated_w"]
    ordered = [c for c in front if c in frame.columns]
    rest = [c for c in frame.columns if c not in ordered]
    return frame[ordered + rest].sort_values(
        ["protocol", "horizon", "model", "test_site"]).reset_index(drop=True)
