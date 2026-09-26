"""Solar geometry and daylight handling.

Solar position is computed with :mod:`pvlib` from the station coordinates and the
UTC timestamp, using the NREL SPA algorithm. It depends only on the clock and the
site, never on measurements, so it is legitimately available at forecast time and
is used for:

* the daylight/night mask that scopes the headline metrics,
* the clear-sky irradiance used to form the clearness index,
* solar-geometry model features,
* the night-time gate on irradiance cleaning.

Timestamps in this project are naive local standard time (UTC+08:00 for Hong
Kong, with no daylight saving). They are localised to that fixed offset before
solar positioning and converted back to naive local time afterwards, so no
ambiguous or non-existent local times are ever produced.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

try:  # pvlib is a hard requirement for solar geometry, but degrade gracefully
    import pvlib

    PVLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    PVLIB_AVAILABLE = False

# Hong Kong observes UTC+08:00 year round and does not use daylight saving.
DEFAULT_UTC_OFFSET_HOURS = 8.0
# NREL SPA. pvlib exposes the SPA implementation under the name 'nrel_numpy'
# (with a numba-accelerated variant and the legacy 'nrel_c' series). The numpy
# variant is selected because it is the fastest option that has no optional
# compilation dependency, which keeps the pipeline installable from
# requirements.txt alone.
SOLAR_POSITION_ALGORITHM = "nrel_numpy"


@dataclass
class SolarFrame:
    """Solar-geometry columns aligned with the source index."""

    elevation: pd.Series          # degrees, negative below the horizon
    azimuth: pd.Series            # degrees from north
    sin_elevation: pd.Series
    cos_elevation: pd.Series
    zenith: pd.Series
    day_length_proxy: pd.Series | None = None

    def as_frame(self) -> pd.DataFrame:
        data = {
            "solar_elevation": self.elevation,
            "solar_azimuth": self.azimuth,
            "sin_solar_elevation": self.sin_elevation,
            "cos_solar_elevation": self.cos_elevation,
            "solar_zenith": self.zenith,
        }
        if self.day_length_proxy is not None:
            data["day_length_proxy"] = self.day_length_proxy
        return pd.DataFrame(data)


def _fixed_offset_tz(utc_offset_hours: float) -> timezone:
    """Return a fixed-offset timezone.

    A fixed offset is used rather than an IANA zone because the study site
    (Hong Kong) observes UTC+08:00 with no daylight saving, so a fixed offset is
    both exact and free of the ambiguous/nonexistent-local-time hazards that a
    ``tz_localize`` against a zone database would have to handle. Note that the
    POSIX ``Etc/GMT`` sign convention is inverted, so an explicit offset avoids
    that footgun entirely.
    """
    return timezone(timedelta(hours=utc_offset_hours))


def compute_solar_position(timestamps: pd.Series | pd.DatetimeIndex,
                           latitude: float, longitude: float,
                           utc_offset_hours: float = DEFAULT_UTC_OFFSET_HOURS
                           ) -> SolarFrame:
    """Compute solar elevation and azimuth for a naive-local-time index.

    Parameters
    ----------
    timestamps
        Timezone-naive timestamps expressed in local standard time.
    latitude, longitude
        Decimal degrees; north and east positive.
    utc_offset_hours
        Fixed offset from UTC.
    """
    if not PVLIB_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "pvlib is required for solar position computation. "
            "Install it with: pip install pvlib")

    index = pd.DatetimeIndex(timestamps)
    if index.tz is not None:
        raise ValueError("compute_solar_position expects timezone-naive local timestamps")

    localized = index.tz_localize(_fixed_offset_tz(utc_offset_hours))
    utc = localized.tz_convert("UTC")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        position = pvlib.solarposition.get_solarposition(
            utc, latitude=latitude, longitude=longitude, method=SOLAR_POSITION_ALGORITHM)

    # pvlib returns a UTC-indexed frame in the same order as the input. The
    # index is replaced positionally rather than by reindex, because the pvlib
    # index is timezone-aware and a reindex onto the naive local index would
    # align nothing and silently produce all-NaN columns.
    position = position.set_axis(index)

    elevation = position["apparent_elevation"].astype(float)
    azimuth = position["azimuth"].astype(float)
    sin_elev = np.sin(np.radians(elevation.to_numpy()))
    cos_elev = np.cos(np.radians(elevation.to_numpy()))
    zenith = position["zenith"].astype(float)

    return SolarFrame(
        elevation=elevation,
        azimuth=azimuth,
        sin_elevation=pd.Series(sin_elev, index=index),
        cos_elevation=pd.Series(cos_elev, index=index),
        zenith=zenith,
    )


def daylight_mask(solar_elevation: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Boolean mask of steps where the sun is above the horizon."""
    return solar_elevation > threshold


def clear_sky_irradiance(timestamps: pd.Series, solar_frame: SolarFrame,
                         latitude: float, longitude: float,
                         linke_turbidity: float | None = None,
                         altitude_m: float | None = None,
                         utc_offset_hours: float = DEFAULT_UTC_OFFSET_HOURS
                         ) -> pd.Series:
    """Clear-sky global horizontal irradiance (W/m2) for the given site.

    Uses the Ineichen-Perez clear-sky model. A single fixed Linke turbidity is
    used rather than a per-timestamp lookup, because the archive's turbidity
    table would introduce a data dependency that is not needed for a
    *relative* clearness index: any smooth, physically reasonable clear-sky
    envelope produces a valid ``kt`` for regime classification, and the
    sensitivity of the regime boundaries to this choice is reported.

    The result is used only to form the clearness index, which is what makes the
    weather regime classification independent of the irradiance sensor's
    absolute calibration.
    """
    if not PVLIB_AVAILABLE:  # pragma: no cover
        raise ImportError("pvlib is required for clear-sky irradiance")

    index = pd.DatetimeIndex(timestamps)
    localized = index.tz_localize(_fixed_offset_tz(utc_offset_hours))
    utc = localized.tz_convert("UTC")

    zenith = np.asarray(solar_frame.zenith, dtype=float)
    # Airmass is undefined at and beyond the zenith; the Ineichen model needs a
    # finite value, so the horizon is used as a floor and the result is masked
    # out for night below.
    zenith_safe = np.clip(zenith, 0.0, 89.9)
    cos_zenith = np.cos(np.radians(zenith_safe))
    airmass = np.where(cos_zenith > 1e-6, 1.0 / np.maximum(cos_zenith, 1e-6), 38.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clearsky = pvlib.clearsky.ineichen(
            apparent_zenith=zenith_safe,
            airmass_absolute=airmass,
            linke_turbidity=linke_turbidity if linke_turbidity is not None else 3.0,
            altitude=altitude_m if altitude_m is not None else 0.0,
            dni_extra=1364.0,
        )
    # pvlib >= 0.13 returns a dict of {'ghi', 'dni', 'dhi'}; older versions
    # returned the GHI array directly. Both shapes are accepted.
    if isinstance(clearsky, dict) or hasattr(clearsky, "keys"):
        clearsky = clearsky["ghi"]
    out = pd.Series(np.asarray(clearsky, dtype=float), index=index)
    out = out.mask(np.asarray(solar_frame.elevation) <= 0.0, 0.0)
    return out


def clearness_index(ghi: pd.Series, ghi_clear: pd.Series,
                    daylight: pd.Series,
                    min_clear_reference: float = 50.0) -> pd.Series:
    """Clearness index kt = GHI / GHI_clear, defined only for usable daylight steps.

    Two guards are applied, both standard practice in solar-resource assessment:

    * ``kt`` is undefined when the clear-sky reference is below
      ``min_clear_reference`` (default 50 W/m2, roughly a 3-5 degree solar
      elevation). Near the horizon the modelled reference approaches zero while
      the instrument still reports small positive values, so the ratio becomes
      numerically meaningless and can exceed 50.
    * night is excluded outright.

    Returns NaN outside those conditions, which forces every downstream consumer
    (regime labelling, feature construction) to handle the undefined case
    explicitly rather than inheriting a divide-by-zero artefact.
    """
    ghi = pd.to_numeric(ghi, errors="coerce").astype(float)
    ghi_clear = pd.to_numeric(ghi_clear, errors="coerce").astype(float)
    usable_clear = ghi_clear > float(min_clear_reference)
    safe_clear = ghi_clear.where(usable_clear)
    kt = ghi / safe_clear
    return kt.where(daylight & usable_clear)


def estimate_clear_sky_scale(kt: pd.Series, daylight: pd.Series,
                             quantile: float = 0.99) -> dict[str, Any]:
    """Estimate the multiplicative offset between the sensor and the clear-sky model.

    The clearness index is only comparable with conventional thresholds such as
    the 0.72 clear-sky boundary if the clear-sky reference is calibrated to the
    site. This function estimates a single factor ``alpha`` from the training
    split such that the ``quantile``-th percentile of the raw clearness index
    maps to 1.0.

    Only the dimensionless index is scaled; the measured irradiance values are
    never modified. A factor near 1.0 is the expected and desirable outcome: it
    means the clear-sky model is already consistent with the instrument, and it
    is reported as a data-quality diagnostic either way. The factor must be
    estimated on the **training split only** and applied unchanged elsewhere,
    otherwise the regime labels would carry information from the evaluation
    period.
    """
    mask = daylight & np.isfinite(kt.to_numpy(dtype=float))
    values = kt.to_numpy(dtype=float)[np.asarray(mask)]
    values = values[np.isfinite(values)]
    if len(values) < 100:
        return {"scale": 1.0, "n_usable": int(len(values)),
                "note": "too few usable clearness values to calibrate; using scale 1.0"}
    scale = float(np.quantile(values, quantile))
    p99 = float(np.percentile(values, 99))
    calibrated = bool(0.95 <= scale <= 1.05)
    return {
        "scale": scale if scale > 1e-6 else 1.0,
        "quantile": quantile,
        "n_usable": int(len(values)),
        "raw_kt_quantiles": {
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": p99,
            "max": float(values.max()),
        },
        "fraction_raw_kt_above_1p2": float((values > 1.2).mean()),
        "calibrated": calibrated,
        "interpretation": (
            "The 99th percentile of the raw clearness index is "
            f"{p99:.3f}. A value close to 1.0 means the clear-sky reference is already "
            "consistent with the irradiance instrument, so the applied scale factor is "
            f"{scale:.3f} and no meaningful correction is implied. A value far from 1.0 "
            "would indicate either an unsuitable clear-sky turbidity or a multiplicative "
            "instrument scale error."),
    }


def apply_clear_sky_scale(kt: pd.Series, scale: float,
                          max_index: float = 1.2) -> pd.Series:
    """Scale and clip a clearness index into its physical range."""
    return (kt / float(scale)).clip(lower=0.0, upper=float(max_index))


def day_of_year_progress(timestamps: pd.Series) -> pd.Series:
    """Fractional position through the year in [0, 1], used for seasonal splits."""
    index = pd.DatetimeIndex(timestamps)
    day_of_year = index.dayofyear.to_numpy()
    year_length = np.where(index.is_leap_year, 366.0, 365.0)
    return pd.Series((day_of_year - 1) / year_length, index=index)


def solar_summary(solar_frame: SolarFrame, daylight: pd.Series) -> dict[str, Any]:
    """Daylight statistics used by the dataset-statistics table."""
    elevation = solar_frame.elevation
    return {
        "daylight_fraction_of_steps": float(daylight.mean()),
        "mean_daylight_steps_per_day": float(daylight.groupby(
            daylight.index.normalize()).sum().mean()),
        "solar_elevation_min_deg": float(elevation.min()),
        "solar_elevation_max_deg": float(elevation.max()),
        "mean_elevation_daylight_deg": float(elevation[daylight].mean()),
        "first_sunrise_hour": float(
            elevation.groupby(elevation.index.normalize()).apply(
                lambda s: s[s > 0].index.min().hour if (s > 0).any() else np.nan).mean()),
        "last_sunset_hour": float(
            elevation.groupby(elevation.index.normalize()).apply(
                lambda s: s[s > 0].index.max().hour if (s > 0).any() else np.nan).mean()),
    }
