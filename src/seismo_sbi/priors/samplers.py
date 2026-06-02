"""Catalogue-driven sampler factories for dataset generation.

Each factory builds a closure with the standard dataset-generator sampler
signature ``f(args, num_samples)`` (where ``args`` is ``parameters.bounds[key]``),
yielding one per-parameter sample per iteration, so it slots directly into
``DatasetGenerator.sampler_lookup_map`` usage without any change to the generator
call sites.

  * :func:`make_catalogue_location_sampler` - source-location Gaussian-mixture / KDE.
  * :func:`make_gutenberg_richter_mt_sampler` - truncated-GR M0 + uniform-on-sphere MT.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .catalogue import EventCatalogue, load_catalogue
from .geo import km_offsets_to_latlon
from .gutenberg_richter import (
    GutenbergRichterModel,
    estimate_mc_maxcurvature,
    fit_b_value_aki,
    magnitude_to_m0,
)
from .moment_tensor import uniform_moment_tensor_on_sphere

CatalogueLike = Union[EventCatalogue, str]


def _as_catalogue(catalogue: CatalogueLike, *, magnitude_type=None) -> EventCatalogue:
    if isinstance(catalogue, EventCatalogue):
        return catalogue
    return load_catalogue(catalogue, magnitude_type=magnitude_type)


def make_catalogue_location_sampler(
    *,
    catalogue: CatalogueLike,
    std_x_km: float,
    std_y_km: float,
    std_z_km: float,
    time_shift: str = "constant",
    use_event_errors: bool = False,
    magnitude_type: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Source-location prior: catalogue Gaussian-mixture (fixed-bandwidth KDE).

    For each sample, a catalogue event is chosen uniformly at random and its
    ``(lat, lon, depth)`` is perturbed by an anisotropic Gaussian whose standard
    deviations are given in kilometres (east ``x``, north ``y``, depth ``z``).
    Offsets are converted to degrees about the chosen event's latitude.

    Parameters
    ----------
    catalogue:
        An :class:`EventCatalogue` or a path to load one from.
    std_x_km, std_y_km, std_z_km:
        Gaussian kernel widths (km) for east, north and depth.
    time_shift:
        ``"constant"`` -> the lower bound (fiducial) is used for the 4th component;
        ``"uniform"`` -> drawn uniformly over the time-shift bound.
    use_event_errors:
        If True and the catalogue carries ``err_h``/``err_z``, add them in
        quadrature with the user widths (per event), broadening poorly-located
        events.
    seed:
        Optional seed for the sampler's private RNG.

    Returns
    -------
    callable
        ``f(args, num_samples)`` yielding 4-vectors ``[lat, lon, depth, time_shift]``,
        clipped to ``args`` (= ``parameters.bounds['source_location']``).
    """
    cat = _as_catalogue(catalogue, magnitude_type=magnitude_type)
    rng = np.random.default_rng(seed)
    n_events = len(cat)
    if time_shift not in ("constant", "uniform"):
        raise ValueError("time_shift must be 'constant' or 'uniform'")

    def sampler(args, num_samples):
        bounds = np.asarray(args, dtype=float)
        if bounds.ndim != 2 or bounds.shape != (2, 4):
            raise ValueError(
                "source_location bounds must be [[lat,lon,depth,t]_low, "
                f"[...]_high] of shape (2,4); got shape {bounds.shape}. The "
                "catalogue prior needs real variable bounds (a superset of the "
                "sampled support)."
            )
        lower, upper = bounds[0], bounds[1]

        idx = rng.integers(0, n_events, size=num_samples)
        ev_lat = cat.latitude[idx]
        ev_lon = cat.longitude[idx]
        ev_depth = cat.depth[idx]

        sx = np.full(num_samples, std_x_km, dtype=float)
        sy = np.full(num_samples, std_y_km, dtype=float)
        sz = np.full(num_samples, std_z_km, dtype=float)
        if use_event_errors:
            if cat.err_h is not None:
                eh = cat.err_h[idx]
                sx = np.sqrt(sx ** 2 + eh ** 2)
                sy = np.sqrt(sy ** 2 + eh ** 2)
            if cat.err_z is not None:
                sz = np.sqrt(sz ** 2 + cat.err_z[idx] ** 2)

        dx = rng.normal(0.0, 1.0, num_samples) * sx
        dy = rng.normal(0.0, 1.0, num_samples) * sy
        dz = rng.normal(0.0, 1.0, num_samples) * sz

        lat, lon = km_offsets_to_latlon(dx, dy, ev_lat, ev_lon)
        depth = ev_depth + dz
        # depth cannot be negative (above the surface)
        depth = np.maximum(depth, 0.0)

        if time_shift == "uniform":
            tshift = rng.uniform(lower[3], upper[3], num_samples)
        else:
            tshift = np.full(num_samples, lower[3], dtype=float)

        samples = np.column_stack([lat, lon, depth, tshift])
        # Keep within the configured bounds so FlexibleScaler stays in [0, 1].
        samples = np.clip(samples, lower, upper)
        for row in samples:
            yield row

    return sampler


def _resolve_b_value(b_value, catalogue: EventCatalogue, mc, delta_m):
    if isinstance(b_value, str):
        if b_value.lower() != "fit":
            raise ValueError(f"b_value must be a number or 'fit', got {b_value!r}")
        if mc is None:
            mc = estimate_mc_maxcurvature(catalogue.magnitude, delta_m=delta_m)
        return fit_b_value_aki(catalogue.magnitude, mc, delta_m=delta_m), mc
    return float(b_value), mc


def _make_magnitude_converter(magnitude_conversion):
    """Build a callable mapping catalogue magnitude -> Mw."""
    if magnitude_conversion in (None, "identity"):
        return lambda m: m
    if isinstance(magnitude_conversion, dict):
        slope = float(magnitude_conversion.get("slope", 1.0))
        intercept = float(magnitude_conversion.get("intercept", 0.0))
        return lambda m: slope * np.asarray(m, dtype=float) + intercept
    if callable(magnitude_conversion):
        return magnitude_conversion
    raise ValueError(
        "magnitude_conversion must be 'identity', a {slope,intercept} dict, or a callable"
    )


def make_gutenberg_richter_mt_sampler(
    *,
    b_value: Union[float, str],
    mw_min: float,
    mw_max: float,
    mc: Optional[float] = None,
    catalogue: Optional[CatalogueLike] = None,
    magnitude_conversion="identity",
    magnitude_type: Optional[str] = None,
    delta_m: float = 0.1,
    seed: Optional[int] = None,
):
    """Moment-tensor prior: truncated Gutenberg-Richter M0 + uniform-on-sphere.

    Hierarchical draw: magnitude ~ truncated GR on ``[mw_min, mw_max]`` (rate
    ``b*ln10``); converted to Mw (identity by default) and then to scalar moment
    ``M0 = 10**(1.5*Mw + 9.1)``; finally a moment tensor is oriented uniformly on
    the sphere of that ``M0``.

    Parameters
    ----------
    b_value:
        A positive float, or ``"fit"`` to estimate it from ``catalogue`` by Aki MLE
        above ``mc`` (auto-estimated by maximum curvature when ``mc`` is omitted).
    mw_min, mw_max:
        Truncation limits of the magnitude prior.
    mc:
        Completeness magnitude for the b-value fit (ignored when ``b_value`` is a
        number).
    catalogue:
        Required only when ``b_value == "fit"``.
    magnitude_conversion:
        ``"identity"`` (Ml ~ Mw), a ``{"slope": a, "intercept": c}`` dict for
        ``Mw = a*M + c``, or a callable.
    seed:
        Optional seed for the sampler's private RNG.

    Returns
    -------
    callable
        ``f(args, num_samples)`` yielding 6-vectors ``[m_rr,m_tt,m_pp,m_rt,m_rp,m_tp]``
        (N.m). ``args`` (= ``parameters.bounds['moment_tensor']``) is accepted to
        satisfy the sampler contract but does not shape the prior.
    """
    if b_value == "fit" or isinstance(b_value, str):
        if catalogue is None:
            raise ValueError("b_value='fit' requires a catalogue")
        cat = _as_catalogue(catalogue, magnitude_type=magnitude_type)
        resolved_b, mc = _resolve_b_value(b_value, cat, mc, delta_m)
    else:
        resolved_b = float(b_value)

    model = GutenbergRichterModel(resolved_b, mw_min, mw_max)
    convert = _make_magnitude_converter(magnitude_conversion)
    rng = np.random.default_rng(seed)
    # expose the resolved values for inspection / notebook plots
    sampler_info = {"b_value": resolved_b, "mc": mc, "model": model}

    def sampler(args, num_samples):
        mags = model.sample_magnitudes(num_samples, rng)
        m0s = magnitude_to_m0(convert(mags))
        for m0 in m0s:
            yield uniform_moment_tensor_on_sphere(m0, rng)

    sampler.info = sampler_info  # type: ignore[attr-defined]
    return sampler
