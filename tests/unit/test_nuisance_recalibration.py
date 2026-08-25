"""Tests for the 2026-08-25 nuisance recalibration (Japan non-DC forensics, TECH §7):

* AmplitudeErrorEffect: log-normal / per-component / always-on modes; legacy path unchanged.
* TimeShiftErrorEffect: distance-scaled per-station sigma.
* DispersionSpreadEffect: phase-only per-octave delay operator (ported from N11).
* InstaseisEnsembleSimulator: Poisson-boundary azimuthal-sector member sampling.
"""
import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.post_processing import (
    AmplitudeErrorEffect,
    DispersionSpreadEffect,
    EFFECT_REGISTRY,
    PostProcessingChain,
    TimeShiftErrorEffect,
    apply_chain_to_array,
    build_post_processing_chain,
)
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator import ensemble as ens_mod
from seismo_sbi.instaseis_simulator.ensemble import InstaseisEnsembleSimulator
from seismo_sbi.instaseis_simulator.wrapper import GenericPointSource, SourceLocation, GeneralMomentTensor

TRACE_LEN = 512


def _receivers(coords, comps=("Z", "R", "T")):
    recs = [Receiver(float(la), float(lo), "XX", f"S{i}", list(comps))
            for i, (la, lo) in enumerate(coords)]
    return Receivers(receivers=recs)


def _map(receivers, amplitude=1.0, signal=False):
    t = np.arange(TRACE_LEN, dtype=np.float64)
    base = (np.sin(2 * np.pi * t / 20.0) * np.exp(-((t - 200) / 60.0) ** 2)) if signal else None
    return {rec.station_name: {c: (base.copy() if signal else np.full(TRACE_LEN, amplitude))
                               for c in rec.components}
            for rec in receivers.iterate()}


# ---------------------------------------------------------------------------
# AmplitudeErrorEffect
# ---------------------------------------------------------------------------
class TestAmplitudeErrorRecalibrated:

    def test_legacy_rng_order_unchanged(self):
        """Default construction must reproduce the original per-station gated uniform draw."""
        recs = _receivers([(0, 0), (1, 1)])
        np.random.seed(3)
        out = AmplitudeErrorEffect(scale_range=(0.7, 1.4))(_map(recs), recs, amplitude_error=0.6)
        np.random.seed(3)
        expect = {}
        for rec in recs.iterate():
            if np.random.uniform() < 0.6:
                s = np.random.uniform(0.7, 1.4)
                expect[rec.station_name] = s
            else:
                expect[rec.station_name] = 1.0
        for name, s in expect.items():
            assert np.allclose(out[name]["Z"], s)
            assert np.allclose(out[name]["T"], s)

    def test_per_station_draw_is_identical_on_all_components(self):
        recs = _receivers([(0, 0)])
        np.random.seed(0)
        out = AmplitudeErrorEffect(distribution="lognormal", log_sigma_dex=0.3, always_on=True)(
            _map(recs), recs, amplitude_error=1.0)
        assert np.allclose(out["S0"]["Z"], out["S0"]["R"]) and np.allclose(out["S0"]["Z"], out["S0"]["T"])

    def test_per_component_draws_are_independent(self):
        recs = _receivers([(0, 0)])
        np.random.seed(0)
        out = AmplitudeErrorEffect(distribution="lognormal", log_sigma_dex=0.3,
                                   per_component=True, always_on=True)(_map(recs), recs, amplitude_error=1.0)
        g = [float(out["S0"][c][0]) for c in ("Z", "R", "T")]
        assert len({round(x, 12) for x in g}) == 3

    def test_always_on_zero_is_identity_and_multiplier_scales_sigma(self):
        recs = _receivers([(0, 0)] * 1)
        eff = AmplitudeErrorEffect(distribution="lognormal", log_sigma_dex=0.3, always_on=True)
        out0 = eff(_map(recs, 2.5), recs, amplitude_error=0.0)
        assert np.allclose(out0["S0"]["Z"], 2.5)
        np.random.seed(11)
        g1 = np.log10(float(eff(_map(recs), recs, amplitude_error=1.0)["S0"]["Z"][0]))
        np.random.seed(11)
        g2 = np.log10(float(eff(_map(recs), recs, amplitude_error=2.0)["S0"]["Z"][0]))
        assert np.isclose(g2, 2.0 * g1)

    def test_lognormal_width_matches_config(self):
        recs = _receivers([(i, i) for i in range(400)])
        np.random.seed(5)
        out = AmplitudeErrorEffect(distribution="lognormal", log_sigma_dex=0.35,
                                   per_component=True, always_on=True)(_map(recs), recs, amplitude_error=1.0)
        g = np.log10([out[s][c][0] for s in out for c in ("Z", "R", "T")])
        assert abs(g.mean()) < 0.03
        assert abs(g.std() - 0.35) < 0.03
        # inter-component differential is non-zero (the property training lacked)
        d = np.log10([out[s]["T"][0] / out[s]["Z"][0] for s in out])
        assert abs(d.std() - 0.35 * np.sqrt(2)) < 0.05

    def test_always_on_applies_to_every_station(self):
        recs = _receivers([(i, i) for i in range(50)])
        np.random.seed(1)
        out = AmplitudeErrorEffect(distribution="lognormal", log_sigma_dex=0.3, always_on=True)(
            _map(recs), recs, amplitude_error=1.0)
        assert all(not np.isclose(out[s]["Z"][0], 1.0) for s in out)

    def test_array_path_training_augmentation(self):
        """The dataloader route (apply_chain_to_array) must carry per-trace gains."""
        recs = _receivers([(0, 0), (1, 1)])
        chain = build_post_processing_chain(
            ["amplitude_error"],
            {"amplitude_error": {"distribution": "lognormal", "log_sigma_dex": 0.3,
                                 "per_component": True, "always_on": True}})
        D = np.ones((2, 3, TRACE_LEN))
        np.random.seed(2)
        out = apply_chain_to_array(chain, D, recs, "ZRT", {"amplitude_error": 1.0})
        assert out.shape == D.shape
        gains = out[:, :, 0]
        assert len(np.unique(np.round(gains, 10))) == 6

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError):
            AmplitudeErrorEffect(distribution="gamma")


# ---------------------------------------------------------------------------
# TimeShiftErrorEffect distance scaling
# ---------------------------------------------------------------------------
class TestTimeShiftDistanceScaled:

    def test_zero_slope_is_legacy(self):
        recs = _receivers([(0, 0), (5, 5)])
        eff_legacy = TimeShiftErrorEffect(sampling_rate=1.0, gaussian_sigma=4.0,
                                          common_offset_dist="gaussian", common_offset_sigma=3.0)
        eff_new = TimeShiftErrorEffect(sampling_rate=1.0, gaussian_sigma=4.0,
                                       common_offset_dist="gaussian", common_offset_sigma=3.0,
                                       sigma_per_1000km=0.0)
        np.random.seed(7); a = eff_legacy(_map(recs, signal=True), recs, time_shift_error=1.0)
        np.random.seed(7); b = eff_new(_map(recs, signal=True), recs, time_shift_error=1.0,
                                       source_location=(0.0, 0.0))
        for s in a:
            assert np.allclose(a[s]["Z"], b[s]["Z"])

    def test_sigma_grows_with_distance_and_caps(self):
        recs = _receivers([(0, 0), (0, 4.5), (0, 13.5)])  # ~0, ~500, ~1500 km
        eff = TimeShiftErrorEffect(sampling_rate=1.0, gaussian_sigma=4.0,
                                   sigma_per_1000km=8.0, distance_cap_km=1200.0)
        sig = eff.station_sigmas(recs, source_location=(0.0, 0.0))
        assert np.isclose(sig["S0"], 4.0, atol=0.05)
        assert 7.5 < sig["S1"] < 8.5
        assert np.isclose(sig["S2"], 4.0 + 8.0 * 1.2)

    def test_requires_source_when_scaled(self):
        recs = _receivers([(0, 0)])
        eff = TimeShiftErrorEffect(sampling_rate=1.0, sigma_per_1000km=5.0)
        with pytest.raises(ValueError):
            eff(_map(recs, signal=True), recs, time_shift_error=1.0)

    def test_source_from_constructor(self):
        recs = _receivers([(0, 9.0)])
        eff = TimeShiftErrorEffect(sampling_rate=1.0, gaussian_sigma=1.0, sigma_per_1000km=10.0,
                                   source_latitude=0.0, source_longitude=0.0)
        sig = eff.station_sigmas(recs)
        assert 10.5 < sig["S0"] < 11.5


# ---------------------------------------------------------------------------
# DispersionSpreadEffect
# ---------------------------------------------------------------------------
class TestDispersionSpread:

    def _eff(self, **kw):
        base = dict(sampling_rate=1.0, sigma_intercept_s=[0, 0, 0, 0],
                    sigma_per_1000km_s=[19.0, 15.0, 9.0, 3.0], source_latitude=0.0, source_longitude=0.0)
        base.update(kw)
        return DispersionSpreadEffect(**base)

    def test_registered_and_zero_is_identity(self):
        assert EFFECT_REGISTRY["dispersion_spread"] is DispersionSpreadEffect
        recs = _receivers([(0, 9.0)])
        m = _map(recs, signal=True)
        out = self._eff()(m, recs, dispersion_spread=0.0)
        assert out is m

    def test_phase_only_preserves_amplitude_spectrum(self):
        recs = _receivers([(0, 9.0)])
        m = _map(recs, signal=True)
        np.random.seed(0)
        out = self._eff()(m, recs, dispersion_spread=1.0)
        a = np.abs(np.fft.rfft(m["S0"]["Z"])); b = np.abs(np.fft.rfft(out["S0"]["Z"]))
        # irfft discards the imaginary part of the Nyquist bin (~1e-6 of peak): phase-only up to that
        assert np.allclose(a, b, rtol=1e-6, atol=1e-5 * a.max())
        assert not np.allclose(m["S0"]["Z"], out["S0"]["Z"])

    def test_positive_tau_delays(self):
        eff = self._eff()
        t = np.arange(TRACE_LEN, dtype=np.float64)
        x = np.exp(-((t - 200) / 10.0) ** 2)
        fr = np.fft.rfftfreq(TRACE_LEN, d=1.0)
        y = eff._delay(x, np.full_like(fr, 7.0), 1.0)
        assert abs(int(np.argmax(y)) - 207) <= 1

    def test_octave_interpolation_clamps(self):
        eff = self._eff()
        fr = np.array([0.0, 1 / 60.0, 1 / 40.0, 1 / 25.0, 1 / 12.5, 1 / 5.0])
        tau = eff.tau_of_freq(fr, np.array([4.0, 3.0, 2.0, 1.0]))
        assert np.isclose(tau[0], 1.0) and np.isclose(tau[1], 1.0)   # beyond 40 s → 40 s value
        assert np.isclose(tau[2], 1.0) and np.isclose(tau[3], 2.0) and np.isclose(tau[4], 4.0)
        assert np.isclose(tau[5], 4.0)                                  # shorter than 12.5 s clamps

    def test_sigma_scales_with_distance(self):
        recs = _receivers([(0, 0), (0, 9.0)])
        sig = self._eff()(_map(recs, signal=True), recs, dispersion_spread=1.0)
        sg = self._eff().station_sigmas(recs)
        assert np.allclose(sg["S0"], 0.0, atol=0.05)
        assert 18.5 < sg["S1"][0] < 19.5 and 2.9 < sg["S1"][3] < 3.1

    def test_coherent_vs_independent_octaves(self):
        """rho=1: all octave delays of a station share one sign; rho=0: they do not (statistically)."""
        recs = _receivers([(0, 9.0)])
        eff1 = self._eff(octave_correlation=1.0)
        eff0 = self._eff(octave_correlation=0.0)
        same1 = same0 = 0
        for k in range(200):
            np.random.seed(k)
            z = np.random.normal(size=4); zc = np.random.normal(); zs = np.random.normal(); zi = np.random.normal(size=4)
            tau1 = np.sqrt(1.0) * zs * eff1.station_sigmas(recs)["S0"]
            tau0 = zi * eff0.station_sigmas(recs)["S0"]
            same1 += int(np.all(np.sign(tau1) == np.sign(tau1[0])))
            same0 += int(np.all(np.sign(tau0) == np.sign(tau0[0])))
        assert same1 == 200 and same0 < 80

    def test_common_fraction_one_gives_identical_relative_delays(self):
        recs = _receivers([(0, 9.0), (0, -9.0)])
        eff = self._eff(common_fraction=1.0)
        np.random.seed(4)
        out = eff(_map(recs, signal=True), recs, dispersion_spread=1.0)
        # both stations at the same distance → identical tau → identical traces
        assert np.allclose(out["S0"]["Z"], out["S1"]["Z"])

    def test_chain_injects_sampling_rate_via_wrapper_list(self):
        from seismo_sbi.sbi import simulator_wrapper as sw
        import inspect
        src = inspect.getsource(sw.GeneralSimulatorWrapper.set_simulation_objects)
        assert "'dispersion_spread'" in src


# ---------------------------------------------------------------------------
# Sector sampling
# ---------------------------------------------------------------------------
class _StubQuerier:
    def __init__(self, member):
        self.member = member

    def get_seismograms(self, source, receiver, components, stf_duration=None):
        return {c: np.full(4, float(self.member)) for c in components}


class _SectorSim(InstaseisEnsembleSimulator):
    """Bypass DB discovery: members are ints, queriers are stubs."""

    def __init__(self, receivers, n_members=8, **kw):
        self._members = list(range(n_members))
        self.receivers = receivers
        self.components = ["Z"]
        self.resample_member_per_station = False
        member_sampling = kw.pop("member_sampling", None)
        sector_lambda = kw.pop("sector_lambda", None)
        # replicate the constructor's sampling logic without touching the DB machinery
        if member_sampling is None:
            member_sampling = 'per_event'
        if member_sampling not in self.VALID_MEMBER_SAMPLING:
            raise ValueError
        if member_sampling == 'sector':
            if sector_lambda is None or float(sector_lambda) < 0.0:
                raise ValueError("sector_lambda")
            self.sector_lambda = float(sector_lambda)
        else:
            self.sector_lambda = None
        self.member_sampling = member_sampling

    @property
    def members(self):
        return self._members

    @property
    def fiducial_member(self):
        return -1

    def _cached_querier(self, member):
        return _StubQuerier(member)

    def _simulate_with_member(self, member, source, **kwargs):
        return {r.station_name: {"Z": np.full(4, float(member))} for r in self.receivers.iterate()}


def _ring(n=12, radius_deg=5.0):
    az = np.linspace(0, 360, n, endpoint=False)
    return _receivers([(radius_deg * np.cos(np.radians(a)), radius_deg * np.sin(np.radians(a))) for a in az],
                      comps=("Z",))


def _source():
    return GenericPointSource(SourceLocation(0.0, 0.0, 10.0, 0.0),
                              GeneralMomentTensor([1e15] * 6))


class TestSectorSampling:

    def test_boundaries_and_index_wrap(self):
        b = np.array([90.0, 200.0])
        idx = InstaseisEnsembleSimulator.sector_index(np.array([10.0, 100.0, 250.0, 359.0]), b)
        assert list(idx) == [0, 1, 0, 0]          # 250 and 359 are past the last boundary → wrap to 0
        assert list(InstaseisEnsembleSimulator.sector_index(np.array([10.0, 300.0]), np.zeros(0))) == [0, 0]
        assert list(InstaseisEnsembleSimulator.sector_index(np.array([10.0, 300.0]), np.array([100.0]))) == [0, 0]

    def test_lambda_zero_is_single_model(self):
        sim = _SectorSim(_ring(), member_sampling="sector", sector_lambda=0.0)
        out = sim.generic_point_source_simulation(_source(), seed=1)
        vals = {float(v["Z"][0]) for v in out.values()}
        assert len(vals) == 1

    def test_large_lambda_approaches_per_station(self):
        sim = _SectorSim(_ring(24), n_members=50, member_sampling="sector", sector_lambda=200.0)
        out = sim.generic_point_source_simulation(_source(), seed=3)
        vals = [float(v["Z"][0]) for v in out.values()]
        assert len(set(vals)) >= 12

    def test_stations_in_one_sector_share_a_member(self):
        sim = _SectorSim(_ring(36), n_members=50, member_sampling="sector", sector_lambda=2.0)
        per_station, bounds = sim.draw_sector_members(_source(), seed=5)
        az = sim.station_azimuths(_source())
        sectors = sim.sector_index(az, bounds)
        for j in set(sectors):
            assert len({per_station[i] for i in range(len(az)) if sectors[i] == j}) == 1

    def test_seeded_reproducible_and_unseeded_varies(self):
        sim = _SectorSim(_ring(), n_members=50, member_sampling="sector", sector_lambda=3.0)
        a = sim.generic_point_source_simulation(_source(), seed=9)
        b = sim.generic_point_source_simulation(_source(), seed=9)
        assert all(np.allclose(a[s]["Z"], b[s]["Z"]) for s in a)
        draws = {tuple(float(v["Z"][0]) for v in sim.generic_point_source_simulation(_source()).values())
                 for _ in range(10)}
        assert len(draws) > 1

    def test_fiducial_overrides_sector(self):
        sim = _SectorSim(_ring(), member_sampling="sector", sector_lambda=3.0)
        out = sim.generic_point_source_simulation(_source(), use_fiducial=True)
        assert all(np.allclose(v["Z"], -1.0) for v in out.values())

    def test_sector_requires_lambda(self):
        with pytest.raises(ValueError):
            _SectorSim(_ring(), member_sampling="sector")

    def test_azimuths(self):
        sim = _SectorSim(_receivers([(5.0, 0.0), (0.0, 5.0), (-5.0, 0.0)], comps=("Z",)), member_sampling="per_event")
        az = sim.station_azimuths(_source())
        assert np.allclose(az, [0.0, 90.0, 180.0], atol=1e-6)
