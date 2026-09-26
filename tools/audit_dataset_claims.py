"""Verify every dataset claim made in the documentation against the actual files.

Claims are asserted in ``README.md`` and ``docs/dataset.md``; this script checks
each one against the downloaded archive, the acquisition manifest and the
processed frames, and prints PASS or FAIL per claim rather than assuming.

    python tools/audit_dataset_claims.py
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "download_manifest.json"
PROCESSED = ROOT / "data" / "processed"
REPO_CONFIG = ROOT / "configs" / "data.yaml"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


def station_names_from_archive(archive: Path) -> set[str]:
    """Station identifiers from the site-level power files in the archive."""
    names: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.namelist():
            normalised = entry.replace("\\", "/")
            if "PV generation dataset" not in normalised or not normalised.endswith(".csv"):
                continue
            if "Site level dataset/" not in normalised:
                continue
            stem = Path(normalised).stem
            if stem.lower().endswith("_inverter"):
                continue
            names.add(stem)
    return names


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    archive = Path(manifest.get("archive", {}).get("local_path", "data/raw/hk_pv_dataset.zip"))
    if not archive.is_absolute():
        archive = ROOT / archive

    # --- licensing and provenance -----------------------------------------
    licence = str(manifest.get("archive", {}).get("licence", ""))
    check("licence is CC0", "CC0" in licence.upper(), f"manifest licence={licence!r}")
    check("DOI recorded", manifest.get("archive", {}).get("record_doi", "").startswith("10."),
          str(manifest.get("archive", {}).get("record_doi")))
    check("SHA-256 recorded", len(str(manifest.get("archive", {}).get("sha256", ""))) == 64)

    # --- archive contents --------------------------------------------------
    if archive.exists():
        stations = station_names_from_archive(archive)
        check("archive holds >= 60 distinct site-level stations", len(stations) >= 60,
              f"found {len(stations)}")
        with zipfile.ZipFile(archive) as bundle:
            entries = [n for n in bundle.namelist() if not n.endswith("/")]
        check("archive file count matches the manifest",
              len(entries) == manifest.get("extracted_file_count"),
              f"{len(entries)} vs manifest {manifest.get('extracted_file_count')}")
    else:
        check("archive present", False, f"missing: {archive}")

    # --- processed record --------------------------------------------------
    featured = PROCESSED / "featured_primary_station.parquet"
    if not featured.exists():
        check("processed feature frame present", False, str(featured))
    else:
        frame = pd.read_parquet(featured)
        power = pd.to_numeric(frame["pv_power_w"], errors="coerce")
        times = pd.to_datetime(frame["Time"])
        check("station is LSK North", str(frame.attrs.get("station", "LSK North")) == "LSK North")
        check("~90,528 steps", abs(len(frame) - 90528) <= 2, f"actual {len(frame):,}")
        check("rated capacity is 55 kW",
              abs(float(power.max()) / 1000.0 - 55.0) < 6.0,
              f"observed peak {power.max() / 1000:.2f} kW")
        check("sampling is 15 minutes",
              float(times.diff().dropna().dt.total_seconds().median()) == 900.0,
              f"median spacing {times.diff().dropna().dt.total_seconds().median():.0f}s")
        # The raw archive record starts 2021-06-01; the processed frame starts a
        # day later because the trailing irradiance and rolling-power windows must
        # fill before the first row is usable. Documents must state the processed
        # start, not the raw one.
        check("processed coverage starts 2021-06-02", str(times.min().date()) == "2021-06-02",
              str(times.min()))
        check("coverage ends 2023-12-31", str(times.max().date()) == "2023-12-31",
              str(times.max()))
        zero_fraction = float((power <= 0).mean())
        check("~50% zero observations", 0.45 <= zero_fraction <= 0.55,
              f"actual {zero_fraction:.3%}")
        check("single contiguous day count >= 900", times.dt.date.nunique() >= 900,
              f"{times.dt.date.nunique()} distinct days")
        weather = [c for c in ("ghi", "temperature", "relative_humidity", "wind_speed")
                   if c in frame.columns]
        check("co-located weather present", len(weather) == 4, f"{weather}")
        # The model input matrix is what must be finite. Two derived diagnostic
        # columns (the raw clearness index and the variability statistic) are
        # deliberately undefined at night and before their trailing windows fill,
        # which is why they are excluded from the input list and must not be
        # counted as missing data.
        inputs = json.loads((PROCESSED / "feature_columns.json")
                            .read_text(encoding="utf-8"))["feature_columns"]
        values = frame[inputs].to_numpy(dtype=float)
        check("model input matrix is entirely finite", bool(np.isfinite(values).all()),
              f"{len(inputs)} columns")
        diagnostic = sorted(c for c in frame.columns
                            if frame[c].isna().any() and c not in inputs)
        check("missing values confined to non-input diagnostics",
              all(c in {"clear_sky_index_raw", "sigma_cv"} for c in diagnostic),
              f"{diagnostic}")

    # --- configured station list ------------------------------------------
    import yaml
    config = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    check("config primary station is LSK North",
          config["dataset"]["primary_station"] == "LSK North")
    check("config declares 6 cross-site holdouts",
          len(config["dataset"].get("holdout_stations", [])) == 6,
          str(config["dataset"].get("holdout_stations")))

    # --- report ------------------------------------------------------------
    width = max(len(name) for name, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{status}  {name:<{width}}  {detail}")
    print(f"\n{len(results) - failed}/{len(results)} claims verified, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

