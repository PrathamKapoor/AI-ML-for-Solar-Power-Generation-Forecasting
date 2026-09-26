"""Print the cross-site panel with each station's rated capacity.

Used when writing the cross-site section of the paper, so the site table is
generated from the dataset metadata rather than transcribed.

    python tools/describe_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_forecasting.config import load_config  # noqa: E402
from solar_forecasting.data import loader  # noqa: E402


def main() -> int:
    config = load_config()
    metadata = loader.load_station_metadata(config)
    primary = config.section("dataset")["primary_station"]
    panel = [primary] + list(config.section("dataset")["holdout_stations"])
    print(f"{'station':10} {'rated_kW':>9} {'role':<18}")
    for name in panel:
        info = loader.describe_station(metadata, name)
        rated = (info.get("rated_w") or 0.0) / 1000.0
        role = "primary (training)" if name == primary else "holdout (never trained on)"
        print(f"{name:10} {rated:9.1f} {role:<18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
