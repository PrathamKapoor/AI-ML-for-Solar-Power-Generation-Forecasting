"""Dataset acquisition: download, verify and extract the source archive.

The dataset is a single CC0 archive hosted on Zenodo. This module fetches it
once, records a SHA-256 checksum so the exact bytes used are auditable, and
extracts only the files the pipeline needs rather than unpacking all 296 MB on
every run.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from ..config import Config, data_dir

CHUNK_SIZE = 1 << 20  # 1 MiB


class AcquisitionError(RuntimeError):
    """Raised when the archive cannot be downloaded or fails verification."""


@dataclass
class ArchiveInfo:
    path: Path
    size_bytes: int
    sha256: str
    already_present: bool


def sha256_of(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_record_metadata(url: str, timeout: int = 60) -> dict:
    """Fetch the Zenodo record so licence and DOI are recorded from the source."""
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "solar-pv-forecasting/1.0 (research; contact in repository README)"},
    )
    response.raise_for_status()
    return response.json()


def download_archive(url: str, destination: Path, timeout: int = 300,
                     expected_sha256: str | None = None,
                     progress: bool = True) -> ArchiveInfo:
    """Download the dataset archive unless it is already present and valid.

    Parameters
    ----------
    expected_sha256
        If provided, an existing file whose checksum differs is rejected and a
        corrupt download is never silently reused.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        checksum = sha256_of(destination)
        if expected_sha256 is None or checksum == expected_sha256:
            return ArchiveInfo(destination, destination.stat().st_size, checksum, True)
        if progress:
            print(f"  cached archive checksum mismatch, re-downloading: {destination.name}")

    if progress:
        print(f"  downloading {url}")
    tmp = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    total = 0
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        declared = int(response.headers.get("Content-Length") or 0)
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if progress and declared:
                    pct = 100.0 * total / declared
                    print(f"\r    {total / 1e6:7.1f} / {declared / 1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
    if progress and declared:
        print()

    checksum = digest.hexdigest()
    if expected_sha256 and checksum != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise AcquisitionError(
            f"checksum mismatch for downloaded archive: expected {expected_sha256}, got {checksum}")

    tmp.replace(destination)
    return ArchiveInfo(destination, total, checksum, False)


def extract_members(archive: Path, members: list[str], target_root: Path) -> list[Path]:
    """Extract only the requested archive members, preserving relative paths.

    Directory entries (names ending in ``/``) are recreated as directories rather
    than written as zero-byte files, which would otherwise shadow the real
    directory on extraction.

    Extraction is guarded against path traversal: a member whose resolved path
    escapes ``target_root`` raises rather than being written.
    """
    target_root.mkdir(parents=True, exist_ok=True)
    root = target_root.resolve()
    written: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        available = set(zf.namelist())
        for member in members:
            if member not in available:
                continue
            destination = (target_root / member).resolve()
            if not str(destination).startswith(str(root)):
                raise AcquisitionError(f"refusing to extract outside target: {member}")
            if member.endswith("/"):
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=CHUNK_SIZE)
            written.append(destination)
    return written


def acquire(config: Config, force: bool = False, progress: bool = True) -> dict:
    """Download the archive and extract the PV, meteorological and metadata files.

    Returns a manifest describing exactly what was retrieved, which is written to
    ``data/raw/download_manifest.json`` so the acquisition step is auditable.
    """
    dataset = config.section("dataset")
    raw_root = data_dir("raw")
    archive_path = raw_root / "hk_pv_dataset.zip"
    extract_root = data_dir("interim")

    expected = dataset.get("archive_sha256")
    if force and archive_path.exists():
        archive_path.unlink()

    info = download_archive(dataset["archive_url"], archive_path,
                            expected_sha256=expected, progress=progress)

    pv_prefix = dataset["pv_prefix"]
    met_prefix = dataset["met_prefix"]
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()

    members = [n for n in names
               if n == dataset["metadata_file"]
               or n.startswith(pv_prefix)
               or n.startswith(met_prefix)]
    written = extract_members(archive_path, members, extract_root)

    # Confirm the primary station and its holdout panel are actually present.
    station_dir = f"{pv_prefix}/PV stations with panel level optimizer/Site level dataset"
    available_stations = sorted(
        Path(n).stem for n in names
        if n.startswith(station_dir) and n.endswith(".csv")
    )
    required = [dataset["primary_station"], *dataset["holdout_stations"]]
    missing = [s for s in required if s not in available_stations]
    if missing:
        raise AcquisitionError(
            f"required station files absent from archive: {missing}. "
            f"Available: {available_stations}")

    manifest = {
        "archive": {
            "url": dataset["archive_url"],
            "record_doi": dataset["record_doi"],
            "zenodo_record_id": dataset["zenodo_record_id"],
            "licence": dataset["licence"],
            "local_path": str(archive_path.relative_to(archive_path.parents[2])),
            "size_bytes": info.size_bytes,
            "sha256": info.sha256,
            "reused_cached_copy": info.already_present,
        },
        "extracted_file_count": len(written),
        "extracted_bytes": sum(p.stat().st_size for p in written),
        "stations_available": available_stations,
        "stations_required": required,
        "working_resolution": dataset["working_resolution"],
        "primary_station": dataset["primary_station"],
    }

    from ..utils.io import write_json

    write_json(manifest, raw_root / "download_manifest.json")
    if progress:
        print(f"  archive: {info.size_bytes / 1e6:.1f} MB  sha256={info.sha256[:16]}...")
        print(f"  extracted {len(written)} files ({manifest['extracted_bytes'] / 1e6:.1f} MB)")
        print(f"  stations available: {len(available_stations)}")
    return manifest
