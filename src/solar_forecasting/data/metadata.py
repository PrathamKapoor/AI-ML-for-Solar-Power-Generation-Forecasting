"""Station metadata extraction from the Brick (TTL) metadata file.

The archive ships a Brick-schema description of the 60 PV systems. Parsing it
gives authoritative rated capacity, coordinates and altitude for each station,
which is used for physical consistency checks and for capacity-normalised
metrics across the cross-site experiment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SYSTEM_PATTERN = re.compile(r"pvsystem:(\S+)\s+a\s+brick:PV_Generation_System\s*;")


def _extract(pattern: str, block: str) -> str | None:
    match = re.search(pattern, block, re.S)
    return match.group(1).strip() if match else None


def parse_station_metadata(ttl_path: Path) -> dict[str, dict[str, Any]]:
    """Parse ``PV generation system metadata.ttl`` into a station dictionary.

    Brick entity names use underscores where the CSV filenames use spaces, so the
    result is keyed by the underscore form; :func:`station_key` converts an
    archive filename to that form.
    """
    text = ttl_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?=pvsystem:\S+\s+a\s+brick:PV_Generation_System\s*;)", text)

    stations: dict[str, dict[str, Any]] = {}
    for block in blocks:
        head = SYSTEM_PATTERN.match(block)
        if not head:
            continue
        name = head.group(1)

        def number(field: str) -> float | None:
            raw = _extract(rf"ext:{field}\s*\[.*?brick:value\s+([-\d.]+)", block)
            try:
                return float(raw) if raw is not None else None
            except ValueError:
                return None

        rated_kw = number("ratedPowerOutput")

        def coordinate(field: str) -> float | None:
            raw = _extract(rf"brick:{field}\s+([-\d.]+)", block)
            try:
                return float(raw) if raw is not None else None
            except ValueError:
                return None

        stations[name] = {
            "brick_name": name,
            "latitude": coordinate("latitude"),
            "longitude": coordinate("longitude"),
            "altitude_m": number("altitude"),
            "location": _extract(r"brick:hasLocation\s*\[\s*brick:value\s+\"([^\"]+)\"", block),
            "connection_date": _extract(
                r"ext:connectionDate\s*\[.*?brick:value\s+\"([^\"]+)\"", block),
            "rated_kw": rated_kw,
            "rated_w": rated_kw * 1000.0 if rated_kw is not None else None,
        }
    return stations


def station_key(filename_stem: str) -> str:
    """Convert an archive CSV stem to the Brick entity name.

    ``"LSK North"`` -> ``"LSK_North"``; parentheses are stripped so that
    ``"CYT (Station 1)"`` maps onto ``"CYT__Station_1_"``.
    """
    stem = filename_stem.replace("(", "").replace(")", "")
    return re.sub(r"\s+", "_", stem.strip()) + ("" if stem.endswith(")") else "")


def lookup_station(stations: dict[str, dict[str, Any]], display_name: str
                   ) -> dict[str, Any] | None:
    """Find a station by its human-readable name, tolerating formatting drift."""
    candidates = {
        display_name,
        display_name.replace(" ", "_"),
        display_name.replace(" ", "_") + "_",
        re.sub(r"\s+", "_", display_name),
    }
    for candidate in candidates:
        if candidate in stations:
            return stations[candidate]
    # Fall back to a normalised comparison ignoring case and underscores.
    normalised = {k.lower().replace("_", ""): v for k, v in stations.items()}
    for candidate in candidates:
        key = candidate.lower().replace("_", "")
        if key in normalised:
            return normalised[key]
    return None


def station_table(stations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the station dictionary into a tabular list for reporting."""
    rows = []
    for name, meta in sorted(stations.items()):
        rows.append({
            "brick_name": name,
            "location": meta.get("location"),
            "latitude": meta.get("latitude"),
            "longitude": meta.get("longitude"),
            "altitude_m": meta.get("altitude_m"),
            "rated_kw": meta.get("rated_kw"),
            "connection_date": meta.get("connection_date"),
        })
    return rows
