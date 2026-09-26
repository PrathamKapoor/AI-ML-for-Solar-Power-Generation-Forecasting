"""Evaluation subpackage.

The modules are:

``metrics``     the metric implementations and their documentation
``regimes``     the transparent weather-regime classifier
``stratified``  post-hoc error stratification (regime, season, time of day, level, ramp)
``uncertainty`` split-conformal prediction intervals
``reporting``   the results tables built from the registry and stored predictions
"""

from . import metrics, regimes, reporting, stratified, uncertainty  # noqa: F401
