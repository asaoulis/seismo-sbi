"""Unit tests for MomentTensorScaler and its FlexibleScaler integration."""

import numpy as np
import pytest

from seismo_sbi.sbi.scalers import (
    FlexibleScaler,
    MomentTensorScaler,
    ZeroOneScaler,
    build_flexible_scaler,
)
from seismo_sbi.sbi.types.parameters import ModelParameters
from seismo_sbi.priors.moment_tensor import uniform_moment_tensor_on_sphere, scalar_moment
from seismo_sbi.priors.gutenberg_richter import magnitude_to_m0

# bounds = +/- 2e18 per component -> M0_max = 2e18/sqrt(2) ~ 1.41e18 (Mw ~ 6.1)
MT_BOUNDS = np.array([[-2e18] * 6, [2e18] * 6])


def _gr_like_tensors(n, mw_min=1.0, mw_max=6.0, seed=0):
    rng = np.random.default_rng(seed)
    mws = rng.uniform(mw_min, mw_max, n)
    m0s = magnitude_to_m0(mws)
    return np.array([uniform_moment_tensor_on_sphere(m0, rng) for m0 in m0s])


def test_round_trip_over_many_orders_of_magnitude():
    scaler = MomentTensorScaler(MT_BOUNDS, n_decades=9.0)
    mts = _gr_like_tensors(5000)
    scaled = scaler.transform(mts)
    recovered = scaler.inverse_transform(scaled)
    assert np.allclose(recovered, mts, rtol=1e-6, atol=1e-3 * np.abs(mts).max())


def test_scaled_output_within_unit_cube():
    scaler = MomentTensorScaler(MT_BOUNDS, n_decades=9.0)
    mts = _gr_like_tensors(5000)
    scaled = scaler.transform(mts)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    assert scaled.shape == mts.shape  # dimension-preserving (6 -> 6)


def test_radius_encodes_log_magnitude_monotonically():
    """The packed radius u = ||2*scaled - 1|| must increase monotonically with log10 M0."""
    scaler = MomentTensorScaler(MT_BOUNDS, n_decades=9.0)
    mts = _gr_like_tensors(4000)
    scaled = scaler.transform(mts)
    u = np.linalg.norm(2 * scaled - 1, axis=1)
    log_m0 = np.log10(np.array([scalar_moment(m) for m in mts]))
    # strong positive rank correlation between radius and log-magnitude
    order = np.argsort(log_m0)
    assert np.corrcoef(log_m0[order], u[order])[0, 1] > 0.99


def test_better_conditioned_than_linear_for_small_events():
    """Small-magnitude events should NOT all collapse to ~0.5 the way linear min-max does."""
    small = _gr_like_tensors(2000, mw_min=2.0, mw_max=3.0, seed=1)
    lin = ZeroOneScaler(MT_BOUNDS)
    ss = MomentTensorScaler(MT_BOUNDS, n_decades=9.0)
    spread_lin = ZeroOneScaler(MT_BOUNDS).transform(small).std()
    spread_ss = ss.transform(small).std()
    # linear scaling crushes small events into a near-delta at 0.5
    assert spread_lin < 1e-3
    assert spread_ss > 20 * spread_lin


def _mt_and_location_params():
    p = ModelParameters()
    p.names = {
        "source_location": ["latitude", "longitude", "depth", "time_shift"],
        "moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
    }
    p.theta_fiducial = {
        "source_location": [36.5, 25.5, 10.0, 0.0],
        "moment_tensor": [1e15] * 6,
    }
    p.bounds = {
        "source_location": [[36.0, 25.0, 0.0, -2.0], [37.0, 26.0, 55.0, 2.0]],
        "moment_tensor": MT_BOUNDS.tolist(),
    }
    return p


def test_flexible_scaler_scale_shape_round_trips_and_keeps_location_linear():
    p = _mt_and_location_params()
    scaler = FlexibleScaler(p, moment_tensor_scaling="scale_shape")
    assert isinstance(scaler.scalers[0], ZeroOneScaler)        # source_location block
    assert isinstance(scaler.scalers[1], MomentTensorScaler)   # moment_tensor block

    rng = np.random.default_rng(0)
    mts = _gr_like_tensors(500)
    locs = rng.uniform([36.1, 25.1, 1.0, -1.0], [36.9, 25.9, 50.0, 1.0], size=(500, 4))
    theta = np.hstack([locs, mts])
    scaled = scaler.transform(theta)
    assert scaled.shape == theta.shape
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    recovered = scaler.inverse_transform(scaled)
    assert np.allclose(recovered, theta, rtol=1e-6, atol=1e3)


def test_default_is_linear_backward_compatible():
    p = _mt_and_location_params()
    default = FlexibleScaler(p)
    assert default.moment_tensor_scaling == "linear"
    assert all(isinstance(s, ZeroOneScaler) for s in default.scalers)


def test_build_flexible_scaler_reads_config_block():
    p = _mt_and_location_params()
    linear = build_flexible_scaler(p, None)
    assert linear.moment_tensor_scaling == "linear"

    ss = build_flexible_scaler(p, {"ml_scaler": {"moment_tensor": "scale_shape",
                                                 "mt_log_decades": 8.0}})
    assert ss.moment_tensor_scaling == "scale_shape"
    assert isinstance(ss.scalers[1], MomentTensorScaler)


def test_invalid_scaling_option_raises():
    p = _mt_and_location_params()
    with pytest.raises(ValueError):
        FlexibleScaler(p, moment_tensor_scaling="nonsense")


def _auto_raw_config(mw_min=3.5, mw_max=6.0, magnitude_conversion="identity"):
    return {
        "ml_scaler": {"moment_tensor": "scale_shape", "mt_log_decades": "auto"},
        "simulations": {"sampling_method": {"moment_tensor": {
            "type": "gutenberg_richter",
            "mw_min": mw_min, "mw_max": mw_max,
            "magnitude_conversion": magnitude_conversion,
        }}},
    }


def test_auto_mt_log_decades_matches_prior_edges_in_scaled_space():
    """mt_log_decades: auto sets the log-M0 window to the GR prior's [mw_min, mw_max]
    so the sampled magnitude maps to u in [0,1] with the edges hit exactly (no clip,
    no wasted range) — the whole point of the dynamic mode."""
    p = _mt_and_location_params()
    scaler = build_flexible_scaler(p, _auto_raw_config(3.5, 6.0))
    mt = scaler.scalers[1]
    assert isinstance(mt, MomentTensorScaler)
    # window edges == prior magnitude edges (M0 = 10**(1.5 Mw + 9.1))
    assert np.isclose(mt.log10_m0_min, 1.5 * 3.5 + 9.1)
    assert np.isclose(mt.log10_m0_max, 1.5 * 6.0 + 9.1)

    mts = _gr_like_tensors(5000, mw_min=3.5, mw_max=6.0)
    scaled = mt.transform(mts)
    u = np.linalg.norm(2 * scaled - 1, axis=1)
    assert u.min() >= 0.0 and u.max() <= 1.0 + 1e-9      # inside [0,1]
    assert u.min() < 0.02 and u.max() > 0.98             # edges reached (perfect fit)
    # no clipping => fully invertible
    recovered = mt.inverse_transform(scaled)
    assert np.allclose(recovered, mts, rtol=1e-6, atol=1e-3 * np.abs(mts).max())


def test_auto_honours_magnitude_conversion():
    """A {slope, intercept} magnitude_conversion shifts the derived window accordingly."""
    p = _mt_and_location_params()
    scaler = build_flexible_scaler(
        p, _auto_raw_config(3.0, 5.0, magnitude_conversion={"slope": 1.0, "intercept": 0.5})
    )
    mt = scaler.scalers[1]
    # Mw = 1.0*M + 0.5  =>  window edges at Mw 3.5 and 5.5
    assert np.isclose(mt.log10_m0_min, 1.5 * 3.5 + 9.1)
    assert np.isclose(mt.log10_m0_max, 1.5 * 5.5 + 9.1)


def test_auto_requires_gr_sampler():
    p = _mt_and_location_params()
    bad = {"ml_scaler": {"moment_tensor": "scale_shape", "mt_log_decades": "auto"}}
    with pytest.raises(ValueError):
        build_flexible_scaler(p, bad)


def test_invalid_mt_log_decades_string_raises():
    p = _mt_and_location_params()
    bad = {"ml_scaler": {"moment_tensor": "scale_shape", "mt_log_decades": "nonsense"}}
    with pytest.raises(ValueError):
        build_flexible_scaler(p, bad)


def test_auto_reads_resolved_sampler_callable():
    """The LIVE pipeline: SBI_Configuration resolves sampling_method.moment_tensor into a
    built sampler CALLABLE (not a dict) before build_flexible_scaler runs. auto must read
    the derived window off the sampler's .info (this is exactly what a naive dict-only
    implementation crashes on: 'function' object has no attribute 'get')."""
    from seismo_sbi.priors.samplers import make_gutenberg_richter_mt_sampler
    p = _mt_and_location_params()
    sampler = make_gutenberg_richter_mt_sampler(
        b_value=0.691, mw_min=3.5, mw_max=6.0, mc=3.8,
        magnitude_conversion="identity", seed=0,
    )
    raw = {
        "ml_scaler": {"moment_tensor": "scale_shape", "mt_log_decades": "auto"},
        # sampling_method already RESOLVED to the callable, as in the live pipeline:
        "simulations": {"sampling_method": {"moment_tensor": sampler}},
    }
    scaler = build_flexible_scaler(p, raw)
    mt = scaler.scalers[1]
    assert isinstance(mt, MomentTensorScaler)
    assert np.isclose(mt.log10_m0_min, 1.5 * 3.5 + 9.1)
    assert np.isclose(mt.log10_m0_max, 1.5 * 6.0 + 9.1)


def test_gr_sampler_exposes_log10_m0_range():
    """The GR sampler stashes its derived log10(M0) window in .info (conversion applied)."""
    from seismo_sbi.priors.samplers import make_gutenberg_richter_mt_sampler
    s = make_gutenberg_richter_mt_sampler(b_value=0.7, mw_min=3.5, mw_max=6.0, mc=3.8, seed=0)
    lo, hi = s.info["log10_m0_range"]
    assert np.isclose(lo, 1.5 * 3.5 + 9.1) and np.isclose(hi, 1.5 * 6.0 + 9.1)
    # with a {slope,intercept} conversion the window shifts to the converted Mw
    s2 = make_gutenberg_richter_mt_sampler(
        b_value=0.7, mw_min=3.0, mw_max=5.0, mc=3.5,
        magnitude_conversion={"slope": 1.0, "intercept": 0.5}, seed=0)
    lo2, hi2 = s2.info["log10_m0_range"]
    assert np.isclose(lo2, 1.5 * 3.5 + 9.1) and np.isclose(hi2, 1.5 * 5.5 + 9.1)
