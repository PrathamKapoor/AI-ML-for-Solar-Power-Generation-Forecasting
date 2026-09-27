"""Print the measured footprint of every model.

Fitted model size, serialised artefact size and peak training memory, so the
computational-cost table can be written from measurement rather than from a
parameter count that is zero for every tree ensemble.

    python tools/measure_footprint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_forecasting.evaluation.reporting import model_footprint  # noqa: E402
from solar_forecasting.models import registry as model_registry  # noqa: E402


def main() -> int:
    print(f"{'model':20} {'fitted size':>13} {'artefact B':>12} {'python peak':>12} {'resident MB':>12}")
    for name in model_registry.all_model_names():
        footprint = model_footprint(name)
        size = footprint["n_parameters"]
        size_text = f"{size:,.0f}" if size == size else "n/a"
        peak = footprint['peak_train_memory_mb']
        resident = footprint['resident_memory_mb']
        print(f"{name:20} {size_text:>13} {footprint['model_size_bytes']:>12,.0f} "
              f"{peak:>9.2f} MB {resident:>9.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

