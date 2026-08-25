"""Distance-scaled scattering (ScatteringCodaEffect.distance_mode) + legacy back-compat."""
import numpy as np
import pytest

from seismo_sbi.instaseis_simulator import post_processing as pp
from seismo_sbi.instaseis_simulator.post_processing import (
    ScatteringCodaEffect, build_post_processing_chain, distance_scaled_alpha,
    distance_tail_energy, _apply_distance_coda_kernel, _apply_random_coda_filter,
    _apply_stahler_phase_filter, _apply_per_station_gated,
)
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers

SRC = (35.0, 137.0)
N = 2000


def _recs():
    # ~110 km per degree of latitude: near (~1 deg), mid (~5 deg), far (~11 deg)
    return Receivers(receivers=[
        Receiver(36.0, 137.0, "XX", "NEAR", ["Z", "R"]),
        Receiver(40.0, 137.0, "XX", "MID", ["Z", "R"]),
        Receiver(46.0, 137.0, "XX", "FAR", ["Z", "R"]),
    ])


def _white_map(recs, seed=0):
    rng = np.random.default_rng(seed)
    return {r.station_name: {c: rng.standard_normal(N) for c in r.components}
            for r in recs.iterate()}


def _wavelet_map(recs):
    t = np.arange(N) / 4.0
    w = np.sin(2 * np.pi * t / 20.0) * np.exp(-((t - 250.0) / 40.0) ** 2)
    return {r.station_name: {c: w.copy() for c in r.components} for r in recs.iterate()}


def _xc0(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# --------------------------------------------------------------------------- helpers

def test_distance_scaled_alpha_ramp_and_cap():
    assert distance_scaled_alpha(0.0, 0.1, 0.5) == pytest.approx(0.1)
    assert distance_scaled_alpha(1000.0, 0.1, 0.5) == pytest.approx(0.6)
    assert distance_scaled_alpha(5000.0, 0.1, 0.5) == 1.0            # clipped
    assert distance_scaled_alpha(5000.0, 0.1, 0.5, distance_cap_km=1000.0) == pytest.approx(0.6)
    np.testing.assert_allclose(distance_scaled_alpha([0.0, 2000.0], 0.0, 0.2), [0.0, 0.4])


def test_distance_tail_energy_inverts_dex_ramp():
    e = distance_tail_energy(1000.0, 0.25)
    assert 0.5 * np.log10(1.0 + e) == pytest.approx(0.25)
    assert distance_tail_energy(800.0, 0.0) == 0.0
    assert distance_tail_energy(3000.0, 0.25, distance_cap_km=1000.0) == pytest.approx(e)


def test_distance_kernel_identity_cases_and_energy():
    x = np.random.default_rng(3).standard_normal(N)
    np.testing.assert_array_equal(_apply_distance_coda_kernel(x, 0.0, 1.0), x)
    np.testing.assert_array_equal(_apply_distance_coda_kernel(x, 0.5, 0.0), x)
    np.random.seed(11)
    y = _apply_distance_coda_kernel(x, 0.6, tail_energy=2.0)
    # kernel [1, tail] with ||tail||^2 = 2 -> output energy ~ 3x on white input (energy is ADDED)
    assert (y ** 2).sum() / (x ** 2).sum() == pytest.approx(3.0, rel=0.15)
    # direct arrival pinned: impulse response starts with the unit spike
    imp = np.zeros(N); imp[10] = 1.0
    np.random.seed(11)
    h = _apply_distance_coda_kernel(imp, 0.6, 2.0)
    assert h[10] == pytest.approx(1.0) and np.all(h[:10] == 0.0)


# --------------------------------------------------------------------------- legacy back-compat

@pytest.mark.parametrize("mode", ["causal", "stahler"])
def test_legacy_path_rng_order_unchanged(mode):
    """Default (distance_mode=False) must reproduce the legacy per-station draw order:
    gate draw, then one alpha draw, then the filter's own draws — station by station."""
    recs = _recs(); m = _white_map(recs)
    eff = ScatteringCodaEffect(alpha_range=(0.2, 0.6), mode=mode)
    np.random.seed(123)
    out = eff(m, recs, scattering_coda=0.5, source_location=SRC)  # extra kwarg must be ignored

    filt = (_apply_stahler_phase_filter if mode == "stahler" else _apply_random_coda_filter)
    np.random.seed(123)
    exp = {}
    for st, comps in m.items():
        if np.random.uniform() < 0.5:
            a = np.random.uniform(0.2, 0.6)
            exp[st] = {c: filt(tr, a, 0.25) for c, tr in comps.items()}
        else:
            exp[st] = {c: tr.astype(np.float64) for c, tr in comps.items()}
    for st in m:
        for c in m[st]:
            np.testing.assert_array_equal(out[st][c], exp[st][c])


def test_default_is_not_distance_mode():
    assert ScatteringCodaEffect().distance_mode is False


# --------------------------------------------------------------------------- distance mode

def _dist_effect(**kw):
    base = dict(distance_mode=True, alpha_intercept=0.05, alpha_per_1000km=0.5,
                excess_dex_per_1000km=0.25, distance_cap_km=1500.0)
    base.update(kw)
    return ScatteringCodaEffect(**base)


def test_distance_mode_identity_when_off_or_zero():
    recs = _recs(); m = _white_map(recs); eff = _dist_effect()
    assert eff(m, recs) is m
    assert eff(m, recs, scattering_coda=0.0, source_location=SRC) is m


def test_distance_mode_requires_source():
    recs = _recs(); m = _white_map(recs)
    with pytest.raises(ValueError, match="source location"):
        _dist_effect()(m, recs, scattering_coda=1.0)


def test_distance_mode_source_from_nuisance_wins_and_ctor_fallback_works():
    recs = _recs(); m = _white_map(recs)
    eff_ctor = _dist_effect(source_latitude=SRC[0], source_longitude=SRC[1])
    np.random.seed(5); a = eff_ctor(m, recs, scattering_coda=1.0)
    np.random.seed(5); b = _dist_effect()(m, recs, scattering_coda=1.0, source_location=SRC)
    np.testing.assert_array_equal(a["FAR"]["Z"], b["FAR"]["Z"])
    # a 4-vector (lat, lon, depth, tshift) as forwarded by the simulator is accepted
    np.random.seed(5); c = _dist_effect()(m, recs, scattering_coda=1.0,
                                          source_location=np.array([*SRC, 10.0, 0.0]))
    np.testing.assert_array_equal(a["FAR"]["Z"], c["FAR"]["Z"])


def test_station_params_grow_with_distance_and_scale_with_multiplier():
    recs = _recs(); eff = _dist_effect()
    p1 = eff.station_scattering_params(recs, SRC, 1.0)
    d = [p1[s][0] for s in ("NEAR", "MID", "FAR")]
    assert d[0] < d[1] < d[2] and 100 < d[0] < 130 and 1100 < d[2] < 1300
    assert p1["NEAR"][1] < p1["MID"][1] < p1["FAR"][1]
    assert p1["NEAR"][2] < p1["MID"][2] < p1["FAR"][2]
    p2 = eff.station_scattering_params(recs, SRC, 2.0)
    assert p2["MID"][1] == pytest.approx(min(1.0, 2 * p1["MID"][1]))
    assert 0.5 * np.log10(1 + p2["MID"][2]) == pytest.approx(2 * 0.5 * np.log10(1 + p1["MID"][2]))


def test_distance_mode_energy_excess_and_decoherence_grow_with_distance():
    recs = _recs(); m = _white_map(recs); eff = _dist_effect()
    np.random.seed(9)
    out = eff(m, recs, scattering_coda=1.0, source_location=SRC)
    p = eff.station_scattering_params(recs, SRC, 1.0)
    ex, xc = {}, {}
    for st in ("NEAR", "MID", "FAR"):
        ex[st] = 0.5 * np.log10((out[st]["Z"] ** 2).sum() / (m[st]["Z"] ** 2).sum())
        xc[st] = _xc0(out[st]["Z"], m[st]["Z"])
        # kernel-level target: 0.25 dex per 1000 km (capped), tolerance for finite tail length
        target = 0.5 * np.log10(1.0 + p[st][2])
        assert ex[st] == pytest.approx(target, abs=0.05)
    assert ex["NEAR"] < ex["MID"] < ex["FAR"]
    assert xc["NEAR"] > xc["MID"] > xc["FAR"]
    assert xc["NEAR"] > 0.9 and xc["FAR"] < 0.7


def test_distance_mode_no_bulk_shift_and_causal():
    """Spike-pinned kernel: onset preserved (causal) and no systematic bulk shift.

    Same protocol as the legacy kernel test: broadband Ricker, mean correlation-peak lag
    over seeds (a single narrowband realisation can legitimately peak a fraction of a
    cycle away once delayed replicas are added)."""
    recs = _recs(); eff = _dist_effect()
    t = (np.arange(N) - 400) / 4.0
    x = (1 - (t / 5.0) ** 2) * np.exp(-((t / 5.0) ** 2) / 2)
    x[:300] = 0.0  # exact zeros before the onset -> causality check is exact
    m = {r.station_name: {c: x.copy() for c in r.components} for r in recs.iterate()}
    lags = {"NEAR": [], "FAR": []}
    for seed in range(30):
        np.random.seed(seed)
        out = eff(m, recs, scattering_coda=1.0, source_location=SRC)
        for st in lags:
            y = out[st]["Z"]
            lags[st].append(np.argmax(np.correlate(y, x, "full")) - (N - 1))
            assert np.all(y[:300] == 0.0), f"{st}: energy before onset"
    # Where the direct arrival dominates (NEAR: tail energy ~0.14) there is no bulk shift.
    # At FAR the tail carries ~1x the direct energy by design, so the correlation peak may
    # sit on a replica — that is the growing far-station lag seen in the data, not a shift
    # of the (pinned) onset, which the causality assertion above covers.
    assert abs(np.mean(lags["NEAR"])) < 3.0, f"NEAR: mean bulk shift {np.mean(lags['NEAR']):.1f}"


def test_distance_mode_stahler_conserves_energy_only_decoheres():
    recs = _recs(); m = _white_map(recs)
    eff = _dist_effect(mode="stahler", alpha_intercept=0.0, alpha_per_1000km=0.8)
    np.random.seed(4)
    out = eff(m, recs, scattering_coda=1.0, source_location=SRC)
    for st in ("NEAR", "FAR"):
        assert (out[st]["Z"] ** 2).sum() == pytest.approx((m[st]["Z"] ** 2).sum(), rel=0.1)
    assert _xc0(out["FAR"]["Z"], m["FAR"]["Z"]) < _xc0(out["NEAR"]["Z"], m["NEAR"]["Z"])


def test_distance_mode_jitter_validation_and_effect():
    with pytest.raises(ValueError):
        _dist_effect(alpha_jitter=1.0)
    with pytest.raises(ValueError):
        _dist_effect(alpha_per_1000km=-0.1)
    recs = _recs(); m = _white_map(recs)
    eff = _dist_effect(alpha_jitter=0.5)
    np.random.seed(1); a = eff(m, recs, scattering_coda=1.0, source_location=SRC)
    np.random.seed(2); b = eff(m, recs, scattering_coda=1.0, source_location=SRC)
    assert not np.allclose(a["FAR"]["Z"], b["FAR"]["Z"])


def test_distance_mode_via_effect_configs_and_chain():
    chain = build_post_processing_chain(
        ["scattering_coda"],
        {"scattering_coda": dict(distance_mode=True, alpha_per_1000km=0.4,
                                 excess_dex_per_1000km=0.2, mode="causal")})
    eff = chain.effects[0]
    assert isinstance(eff, ScatteringCodaEffect) and eff.distance_mode
    recs = _recs(); m = _white_map(recs)
    np.random.seed(0)
    out = chain(m, recs, {"scattering_coda": 1.0, "source_location": np.array([*SRC, 5.0, 0.0])})
    assert not np.allclose(out["FAR"]["Z"], m["FAR"]["Z"])


def test_simulator_forwards_source_location_to_chain():
    """Simulator.run_simulation must hand the source position to the post-processing chain."""
    import inspect
    from seismo_sbi.instaseis_simulator import simulator as sim_mod
    src = inspect.getsource(sim_mod.Simulator.run_simulation)
    assert 'post_proc_params.setdefault("source_location"' in src
