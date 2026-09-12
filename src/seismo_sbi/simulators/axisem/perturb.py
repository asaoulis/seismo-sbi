"""Perturb AxiSEM external background models to build a velocity-model ensemble.

The signed increments of ``vpv`` and ``vsv`` between successive nodes are scaled by a mean-one
log-normal factor ``exp(N(-sigma^2/2, sigma))`` and the profile is rebuilt from a fixed deep
anchor, which keeps the profile monotone and the ensemble centred on the reference
(``E[v'] = v``). The radial gaps between nodes are perturbed and renormalised so the surface
and centre radii stay fixed, shifting discontinuity depths. Fluid rows stay fluid and
``vsv < vpv/sqrt(2)`` is enforced; density is fixed or re-derived from Brocher (2005).
"""

from __future__ import annotations

import numpy as np

from seismo_sbi.simulators.cps.smooth_perturbations import brocher_rho
from .model_io import BackgroundModel

#: The largest shear-to-compressional speed ratio a solid can have, at Poisson's ratio zero.
MAX_VS_OVER_VP = 1.0 / np.sqrt(2.0)


def _perturb_widths(radius: np.ndarray, width_sigma: float, rng) -> np.ndarray:
    """Return a new descending radius array with perturbed layer widths.

    Gaps between successive radii are scaled by ``exp(N(0, width_sigma))``;
    zero gaps (the duplicated-radius discontinuity rows) stay exactly zero so
    the double-line structure is preserved.  Gaps are renormalised so the total
    span (surface radius minus centre radius) is unchanged, pinning both
    endpoints.
    """
    if width_sigma <= 0 or len(radius) < 3:
        return radius.copy()

    gaps = -np.diff(radius)               # positive gaps (descending radius)
    nonzero = gaps > 0
    total = gaps.sum()

    factors = np.ones_like(gaps)
    factors[nonzero] = np.exp(rng.normal(0.0, width_sigma, size=nonzero.sum()))
    new_gaps = gaps * factors

    # Renormalise non-zero gaps so the total span is preserved (endpoints fixed).
    span_nonzero = new_gaps[nonzero].sum()
    if span_nonzero > 0:
        new_gaps[nonzero] *= total / span_nonzero

    new_radius = np.empty_like(radius)
    new_radius[0] = radius[0]
    new_radius[1:] = radius[0] - np.cumsum(new_gaps)
    return new_radius


def _perturb_monotone_increments(v, anchor_idx, sigma, rng):
    """Perturb a velocity column while preserving its monotonic structure.

    Instead of perturbing absolute node values (which lets closely-spaced nodes
    cross and produce unphysical velocity reversals / low-velocity zones), we
    perturb the *increments* between successive nodes multiplicatively:

        g_i  = v[i+1] - v[i]                  (signed gap, top -> down)
        g_i' = g_i * exp(N(-sigma^2/2, sigma)) (exp > 0 -> sign of g_i preserved)

    and rebuild the profile **upward from a fixed anchor** at ``anchor_idx``
    (``v[anchor_idx]`` and everything below it are left unchanged):

        v'[i] = v[anchor_idx] - sum_{j>=i} g_j'

    Because every gap keeps its sign, the perturbed profile has exactly the same
    monotonicity as the fiducial (increases stay increases; a genuine LVZ stays
    an LVZ) — no backward bending is ever introduced — and the shallow nodes can
    never overtake the fixed deep anchor.  Spread is largest at the surface and
    shrinks toward the anchor (the shallow crust being the least constrained),
    which is the physically sensible uncertainty structure.

    **Mean-preserving:** the log-normal factor uses a ``-sigma^2/2`` drift so it
    has *mean* 1 (``E[exp(N(-sigma^2/2, sigma))] = 1``), not just median 1.  Then
    ``E[g_i'] = g_i`` and, since the reconstruction is linear in the gaps with a
    fixed anchor, ``E[v'] = v`` exactly at every node.  A plain ``N(0, sigma)``
    factor would have mean ``exp(sigma^2/2) > 1``, inflating every increment and
    systematically biasing the shallow velocities below the fiducial (the bias
    grows toward the surface and scales as ``exp(sigma^2/2)``), so the ensemble
    mean would drift away from the 1-D reference.  The drift correction keeps the
    *marginal* deliberately non-log-normal (no LVZ/cliff artifacts) while
    centring the ensemble on the reference model.
    """
    v_new = v.copy()
    if sigma <= 0 or anchor_idx < 1:
        return v_new
    gaps = v[1:anchor_idx + 1] - v[:anchor_idx]              # g_i, i=0..anchor_idx-1
    factors = np.exp(rng.normal(-0.5 * sigma ** 2, sigma, size=anchor_idx))  # E[factor]=1
    gaps_p = gaps * factors
    suffix = np.cumsum(gaps_p[::-1])[::-1]                   # suffix[i] = sum_{j>=i} g_j'
    v_new[:anchor_idx] = v[anchor_idx] - suffix
    return v_new


def perturb_background_model(
    model: BackgroundModel,
    *,
    vp_sigma: float = 0.02,
    vs_sigma: float = 0.02,
    width_sigma: float = 0.0,
    rho_mode: str = "fixed",
    max_depth_km: float = None,
    seed=None,
) -> BackgroundModel:
    """Return a perturbed copy of ``model``.

    Parameters
    ----------
    vp_sigma, vs_sigma : float
        Std-dev of the log-normal fractional perturbation on Vp / Vs.
    width_sigma : float
        Std-dev of the log-normal layer-width / depth perturbation (0 disables).
    rho_mode : {'fixed', 'brocher'}
        Keep density as-is, or re-derive it from the perturbed Vp via Brocher
        (2005).  Brocher expects Vp in km/s; we convert based on the model's
        UNITS (``m`` -> m/s assumed, scaled to km/s).
    max_depth_km : float or None
        If set, only nodes **shallower** than this depth are perturbed (Vp/Vs,
        layer widths, and Brocher density); deeper nodes are left exactly as the
        fiducial.  Use this to perturb only the (uncertain) crust and keep the
        deep reference (e.g. PREM) fixed — Brocher is a *crustal* relation and is
        invalid for the mantle/core, and perturbing deep discontinuities is not
        intended.  ``None`` perturbs the whole model (legacy).
    seed : int or None
        Seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    out = model.copy()
    data = out.data

    vp_i = out.col_index("vpv")
    vs_i = out.col_index("vsv")
    r_i = out.col_index("radius")
    n = out.n_rows

    # Which nodes to perturb (leading shallow block, since radius is descending).
    radius = out.radius
    surface_r = float(radius.max())
    if max_depth_km is None:
        mask = np.ones(n, dtype=bool)
    else:
        depth_km = (surface_r - radius) / 1000.0
        mask = depth_km < float(max_depth_km)
    k = int(mask.sum())                     # crust = leading block 0..k-1
    # Build upward from the first FIXED node below the perturbed block (so the
    # deep reference stays put and the crust connects to it without overtaking).
    anchor_idx = k if k < n else n - 1

    # -- velocities (monotonicity-preserving increment perturbation) -------
    vp_new = _perturb_monotone_increments(data[:, vp_i], anchor_idx, vp_sigma, rng)
    vs_new = _perturb_monotone_increments(data[:, vs_i], anchor_idx, vs_sigma, rng)

    fluid = data[:, vs_i] == 0.0          # keep fluid layers fluid
    vs_new[fluid] = 0.0
    over = (~fluid) & (vs_new > MAX_VS_OVER_VP * vp_new)   # enforce Vs < Vp/sqrt(2)
    vs_new[over] = vp_new[over] * (0.99 * MAX_VS_OVER_VP)

    data[:, vp_i] = vp_new
    data[:, vs_i] = vs_new

    # -- layer widths / depths --------------------------------------------
    # Perturb only the shallow block's radii, pinning the surface and the first
    # un-perturbed (deep) node so deeper structure is untouched.
    if width_sigma > 0:
        k = int(mask.sum())               # mask is a leading contiguous block
        if 0 < k < n:
            sub = radius[:k + 1].copy()    # surface .. first fixed deep node
            data[:k + 1, r_i] = _perturb_widths(sub, width_sigma, rng)
        elif k == n:
            data[:, r_i] = _perturb_widths(radius, width_sigma, rng)

    # -- density (Brocher) on perturbed nodes only ------------------------
    if rho_mode == "brocher" and "rho" in out.columns:
        rho_i = out.col_index("rho")
        units = out.meta.get("UNITS", "m").lower()
        to_km_s = 1.0e-3 if units == "m" else 1.0
        rho = brocher_rho(vp_new[mask] * to_km_s)         # g/cm^3
        rho = np.maximum(rho, 1.0)
        data[mask, rho_i] = rho * 1000.0 if units == "m" else rho
    elif rho_mode not in ("fixed", "brocher"):
        raise ValueError(f"Unknown rho_mode {rho_mode!r}")

    return out


def unit_smooth_field(n_points: int, spacing_km: float, correlation_km: float,
                      generator) -> np.ndarray:
    """A zero-mean, unit-variance Gaussian field smoothed to a correlation length."""
    from scipy.ndimage import gaussian_filter

    field = gaussian_filter(generator.normal(size=n_points),
                            sigma=max(0.5, correlation_km / spacing_km), mode="reflect")
    field = field - field.mean()
    return field / (field.std() + 1e-16)


def depth_tapered_sigma(depth_km, sigma_surface: float = 0.05, sigma_deep: float = 0.015,
                        taper_start_km: float = 30.0,
                        taper_end_km: float = 65.0) -> np.ndarray:
    """Fractional spread per depth: a half-cosine from the surface value to the deep floor."""
    depth_km = np.asarray(depth_km, float)
    fraction = np.clip((depth_km - taper_start_km) / (taper_end_km - taper_start_km), 0.0, 1.0)
    taper = 0.5 * (1.0 + np.cos(np.pi * fraction))
    return sigma_deep + (sigma_surface - sigma_deep) * taper


def perturb_layered_model(top_depth_km, vp_km_s, vs_km_s, density, *, sigma_profile,
                          correlation_km: float = 30.0, vp_vs_correlation: float = 0.9,
                          width_sigma: float = 0.0, density_coupling: float = 0.25,
                          seed=None) -> tuple:
    """One perturbed member as ``(layer thicknesses, vp, vs, density)``.

    ``density_coupling`` is Birch's law: a fractional change in density of that many times the
    fractional change in compressional speed, which keeps the reference density and is
    mean-preserving. Shear speed is capped below the compressional one, because a draw that
    crosses it is not an elastic solid.
    """
    generator = np.random.default_rng(seed)
    n_points = len(vp_km_s)
    spacing_km = (float(np.median(np.diff(top_depth_km))) if n_points > 1 else 0.5)
    sigma = np.asarray(sigma_profile, float)

    shared = unit_smooth_field(n_points, spacing_km, correlation_km, generator)
    own_vp = unit_smooth_field(n_points, spacing_km, correlation_km, generator)
    own_vs = unit_smooth_field(n_points, spacing_km, correlation_km, generator)
    weight = np.sqrt(np.clip(vp_vs_correlation, 0.0, 1.0))
    independent = np.sqrt(1.0 - weight * weight)
    epsilon_vp = (weight * shared + independent * own_vp) * sigma
    epsilon_vs = (weight * shared + independent * own_vs) * sigma

    perturbed_vp = vp_km_s * np.exp(epsilon_vp)
    perturbed_vs = vs_km_s * np.exp(epsilon_vs)
    crossed = perturbed_vs > MAX_VS_OVER_VP * perturbed_vp
    perturbed_vs[crossed] = perturbed_vp[crossed] * (0.99 * MAX_VS_OVER_VP)
    perturbed_density = density * np.exp(density_coupling * epsilon_vp)

    thickness = np.diff(np.concatenate((top_depth_km,
                                        [top_depth_km[-1] + spacing_km])))
    if width_sigma > 0:
        widths = width_sigma * (sigma / sigma.max())
        thickness = thickness * np.exp(
            unit_smooth_field(n_points, spacing_km, correlation_km, generator) * widths)
    return thickness, perturbed_vp, perturbed_vs, perturbed_density


def control_point_sigma(depth_km, control_depth_km, control_sigma, *,
                        smoothing_km: float = 1.0) -> np.ndarray:
    """A spread profile interpolated between named depths and then smoothed.

    Where a taper is a shape assumed in advance, this is a spread measured from how far
    independent models of the same region sit apart, so it is given as the control points that
    measurement produced. The smoothing removes the corners interpolation leaves at them.
    """
    from scipy.ndimage import gaussian_filter1d

    depth_km = np.asarray(depth_km, float)
    spacing_km = float(np.median(np.diff(depth_km))) if len(depth_km) > 1 else smoothing_km
    sigma = np.interp(depth_km, control_depth_km, control_sigma)
    return gaussian_filter1d(sigma, sigma=max(1, int(smoothing_km / spacing_km)),
                             mode="nearest")


def perturb_velocity_ratio_model(top_depth_km, vp_km_s, vs_km_s, density, *, sigma_vp,
                                 sigma_ratio, correlation_km: float = 8.0,
                                 width_sigma: float = 0.0, density_coupling: float = 0.25,
                                 min_ratio: float = 1.43, seed=None) -> tuple:
    """One member, perturbing the compressional speed and the speed ratio separately.

    The ratio carries its own spread because it is what the data constrain separately: a
    region can be uncertain in absolute speed while its ratio is known, or the reverse. The
    ratio is held above ``min_ratio``, below which the material is not an elastic solid.
    """
    generator = np.random.default_rng(seed)
    n_points = len(vp_km_s)
    spacing_km = (float(np.median(np.diff(top_depth_km))) if n_points > 1 else 0.5)
    epsilon_vp = unit_smooth_field(n_points, spacing_km, correlation_km,
                                   generator) * np.asarray(sigma_vp, float)
    epsilon_ratio = unit_smooth_field(n_points, spacing_km, correlation_km,
                                      generator) * np.asarray(sigma_ratio, float)

    perturbed_vp = vp_km_s * np.exp(epsilon_vp)
    ratio = np.maximum((vp_km_s / vs_km_s) * np.exp(epsilon_ratio), min_ratio)
    perturbed_vs = perturbed_vp / ratio
    perturbed_density = density * np.exp(density_coupling * epsilon_vp)

    thickness = np.diff(np.concatenate((top_depth_km, [top_depth_km[-1] + spacing_km])))
    if width_sigma > 0:
        widths = width_sigma * (np.asarray(sigma_vp, float) / np.max(sigma_vp))
        thickness = thickness * np.exp(
            unit_smooth_field(n_points, spacing_km, correlation_km, generator) * widths)
    return thickness, perturbed_vp, perturbed_vs, perturbed_density
