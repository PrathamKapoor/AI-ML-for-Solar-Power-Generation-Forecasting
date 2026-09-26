"""Visualisation subpackage.

All figures are written to ``results/figures`` at 200 dpi with a non-interactive
backend. See :mod:`solar_forecasting.visualization.figures` for the individual
plots and the conventions they share.
"""

from . import figures  # noqa: F401
from .figures import apply_style, figures_dir, pretty, save  # noqa: F401
