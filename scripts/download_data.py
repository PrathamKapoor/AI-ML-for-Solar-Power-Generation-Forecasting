#!/usr/bin/env python
"""Download, verify and extract the source dataset.

    python scripts/download_data.py
    python scripts/download_data.py --force          # ignore the cached archive
    python scripts/download_data.py --metadata-only  # print the source record only

The dataset is CC0 1.0 and requires no registration or API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from solar_forecasting.config import load_config  # noqa: E402
from solar_forecasting.data.acquisition import acquire, fetch_record_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a valid cached archive exists")
    parser.add_argument("--metadata-only", action="store_true",
                        help="print the source record metadata and exit")
    args = parser.parse_args()

    config = load_config()
    dataset = config.section("dataset")

    if args.metadata_only:
        record = fetch_record_metadata(dataset["metadata_url"])
        summary = {
            "title": record.get("metadata", {}).get("title"),
            "doi": record.get("doi"),
            "publication_date": record.get("metadata", {}).get("publication_date"),
            "licence": record.get("metadata", {}).get("license"),
            "creators": [c.get("name") for c in record.get("metadata", {}).get("creators", [])],
            "keywords": record.get("metadata", {}).get("keywords"),
            "files": [{"key": f.get("key"), "size": f.get("size")}
                      for f in record.get("files", [])],
        }
        print(json.dumps(summary, indent=2))
        return 0

    print("Dataset")
    print(f"  name     : {dataset['name']}")
    print(f"  DOI      : {dataset['record_doi']}")
    print(f"  licence  : {dataset['licence']}")
    print(f"  source   : {dataset['archive_url']}")
    print(f"  station  : {dataset['primary_station']}")
    print()

    manifest = acquire(config, force=args.force)
    print()
    print("Done. Next: python scripts/preprocess.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
