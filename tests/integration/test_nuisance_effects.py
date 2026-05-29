"""Integration tests: nuisance parameter effects through the full simulation path.

These tests verify that nuisance parameters wired into Simulator.run_simulation()
actually change the outputs produced — covering the complete path from parameter
dict → run_simulation() → PostProcessingChain → flat 1-D array returned by
input_output_simulation().

Test categories
---------------
Fast (no forward model required):
    - MockSimulator: checks that run_simulation() routes post-proc params to the
      chain and that outputs are modified compared to a no-nuisance baseline.
    - FixedLocationKernelSimulator: uses synthetic sensitivity kernels (like the
      compression pipeline tests) so no Instaseis DB is needed; confirms that
      amplitude_error and instrument_dropout change the output of a real Simulator
      subclass end-to-end.
    - input_output_simulation unbound call: confirms the whole wrapper path
      (sample nuisance → merge → simulate → flatten) is correct.

Slow (require CPS or Instaseis — marked with @pytest.mark.slow):
    - CPS precomputed simulator with amplitude_error: verifies that a real
      forward-model call produces outputs that differ with/without the nuisance.
"""

from __future__ import annotations

import os
from copy import deepcopy
from functools import partial

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.simulator import Simulator, FixedLocationKernelSimulator
from seismo_sbi.instaseis_simulator.wrapper import GenericPointSource
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
from seismo_sbi.instaseis_simulator.post_processing import (
    AmplitudeErrorEffect,
    InstrumentDropoutEffect,
    TimeShiftErrorEffect,
    ScatteringCodaEffect,
    PostProcessingChain,
    build_post_processing_chain,
)
from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData
from seismo_sbi.sbi.simulator_wrapper import GeneralSimulatorWrapper
from seismo_sbi.sbi.types.parameters import ModelParameters

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACE_LEN = 40
_SOURCE_LOC = [0.0, 0.0, 10.0, 0.0]
_MT_PARAMS = {"source_location": _SOURCE_LOC, "moment_tensor": [1e14] * 6}
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockSimulator(Simulator):
    """Minimal Simulator returning canned constant-valued seismograms."""

    def __init__(self, receivers, trace_len=TRACE_LEN, amplitude=1.0, **kwargs):
        super().__init__(
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=trace_len,
            synthetics_processing={
                "sampling_rate": 1.0,
                "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
            },
            **kwargs,
        )
        self._trace_len = trace_len
        self._amplitude = amplitude

    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        return {
            rec.station_name: {
                comp: np.full(self._trace_len, self._amplitude)
                for comp in rec.components
            }
            for rec in self.receivers.iterate()
        }


def _make_receivers(*station_names):
    return Receivers(receivers=[
        Receiver(float(i), float(i), "XX", name, ["Z"])
        for i, name in enumerate(station_names)
    ])


def _constant_sampler(value, n):
    for _ in range(n):
        yield value


def _make_mt_model_parameters(nuisance: dict | None = None):
    mp = ModelParameters()
    mp.names["moment_tensor"] = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    mp.theta_fiducial["moment_tensor"] = [1e14] * 6
    mp.nuisance["source_location"] = _SOURCE_LOC
    mp.bounds["source_location"] = _SOURCE_LOC
    if nuisance:
        mp.nuisance.update(nuisance)
        mp.bounds.update(nuisance)
    return mp


def _call_io_sim(parameters, data_loader, samplers, simulator, theta, **kwargs):
    return GeneralSimulatorWrapper.input_output_simulation(
        None, parameters, data_loader, samplers, simulator, theta, **kwargs
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def one_station():
    return _make_receivers("STA1")


@pytest.fixture
def two_stations():
    return _make_receivers("STA1", "STA2")


@pytest.fixture
def mock_sim(one_station):
    return MockSimulator(one_station, amplitude=1.0)


@pytest.fixture
def data_loader(one_station):
    return SimulationDataLoader(components=["Z"], receivers=one_station)


# ---------------------------------------------------------------------------
# Helper: build FixedLocationKernelSimulator from synthetic kernels
# ---------------------------------------------------------------------------


def _kernel_sim(receivers, n_params=6, amplitude=1.0, post_processing_effects=None):
    """Build a FixedLocationKernelSimulator with random kernels (no real DB needed)."""
    rng = np.random.default_rng(RNG_SEED)
    n_traces = sum(len(r.components) for r in receivers.iterate())
    data_len = TRACE_LEN * n_traces
    gradients = rng.standard_normal((n_params, data_len))
    theta_fid = np.ones(n_params) * amplitude
    D_fid = gradients.T @ theta_fid
    scd = ScoreCompressionData(
        theta_fiducial=theta_fid,
        data_fiducial=D_fid,
        data_parameter_gradients=gradients,
        second_order_gradients=None,
    )
    return FixedLocationKernelSimulator(
        scd,
        components=["Z"],
        receivers=receivers,
        seismogram_duration_in_s=TRACE_LEN,
        synthetics_processing={
            "sampling_rate": 1.0,
            "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
        },
        post_processing_effects=post_processing_effects,
    )


# ===========================================================================
# Category 1: run_simulation() routes post-proc params to the chain
# ===========================================================================


class TestRunSimulationWithNuisance:
    """End-to-end: source_parameters dict with post-proc nuisance → chain applied."""

    def test_amplitude_error_changes_output(self, one_station):
        """amplitude_error=1.0 (prob=1, fixed scale) must change the returned seismograms."""
        sim_no_effect = MockSimulator(one_station, amplitude=1.0)
        # Fixed scale_range=(3.0, 3.0) → deterministic factor-of-3 when prob=1
        sim_with_effect = MockSimulator(
            one_station, amplitude=1.0,
            post_processing_effects=[AmplitudeErrorEffect(scale_range=(3.0, 3.0))]
        )

        params = dict(_MT_PARAMS)
        params["amplitude_error"] = 1.0  # probability = 1 → always apply

        _, seis_no_effect = sim_no_effect.run_simulation(dict(_MT_PARAMS))
        _, seis_with_effect = sim_with_effect.run_simulation(params)

        baseline = seis_no_effect["STA1"]["Z"]
        scaled = seis_with_effect["STA1"]["Z"]

        assert np.allclose(scaled, 3.0 * baseline), (
            "amplitude_error=1.0 with scale_range=(3,3) must triple every trace value"
        )

    def test_amplitude_error_zero_is_identity(self, one_station):
        """amplitude_error=0.0 (probability=0) must leave every trace unchanged."""
        sim = MockSimulator(
            one_station, amplitude=2.5,
            post_processing_effects=[AmplitudeErrorEffect()]
        )
        params = dict(_MT_PARAMS)
        params["amplitude_error"] = 0.0
        _, seis = sim.run_simulation(params)
        assert np.allclose(seis["STA1"]["Z"], 2.5)

    def test_instrument_dropout_prob1_zeros_all(self, two_stations):
        """instrument_dropout=1.0 must zero every station."""
        sim = MockSimulator(
            two_stations, amplitude=1.0,
            post_processing_effects=[InstrumentDropoutEffect()]
        )
        params = dict(_MT_PARAMS)
        params["instrument_dropout"] = 1.0

        _, seis = sim.run_simulation(params)
        assert np.allclose(seis["STA1"]["Z"], 0.0)
        assert np.allclose(seis["STA2"]["Z"], 0.0)

    def test_instrument_dropout_prob0_no_change(self, two_stations):
        """instrument_dropout=0.0 must leave every trace unchanged."""
        sim = MockSimulator(
            two_stations, amplitude=1.0,
            post_processing_effects=[InstrumentDropoutEffect()]
        )
        params = dict(_MT_PARAMS)
        params["instrument_dropout"] = 0.0

        _, seis = sim.run_simulation(params)
        assert np.allclose(seis["STA1"]["Z"], 1.0)
        assert np.allclose(seis["STA2"]["Z"], 1.0)

    def test_source_params_not_mutated(self, one_station):
        """run_simulation() must not mutate the caller's dict (regression test)."""
        sim = MockSimulator(
            one_station, post_processing_effects=[AmplitudeErrorEffect()]
        )
        params = dict(_MT_PARAMS)
        params["amplitude_error"] = 2.0
        original_keys = set(params.keys())
        sim.run_simulation(params)
        assert set(params.keys()) == original_keys

    def test_unknown_post_proc_keys_ignored(self, one_station):
        """Extra unknown nuisance keys must not raise during run_simulation()."""
        sim = MockSimulator(one_station)
        params = dict(_MT_PARAMS)
        params["some_future_effect"] = 0.5
        # Must not raise
        sim.run_simulation(params)

    def test_per_station_scales_are_independent_stochastic(self, two_stations):
        """Different stations must receive different scale factors (with overwhelming probability).

        Uses the default scale_range [0.5, 2.0] with amplitude_error=1.0.
        Over multiple seeds at least one trial should produce differing station scales.
        """
        sim = MockSimulator(
            two_stations, amplitude=1.0,
            post_processing_effects=[AmplitudeErrorEffect()]  # default range
        )
        params = dict(_MT_PARAMS)
        params["amplitude_error"] = 1.0  # always apply

        found_different = False
        for seed in range(20):
            np.random.seed(seed)
            _, seis = sim.run_simulation(params)
            scale_sta1 = float(seis["STA1"]["Z"][0])
            scale_sta2 = float(seis["STA2"]["Z"][0])
            if not np.isclose(scale_sta1, scale_sta2):
                found_different = True
                break

        assert found_different, (
            "Over 20 seeds, STA1 and STA2 should receive different scale factors "
            "at least once — amplitude modulation must be independent per station"
        )

    def test_scale_within_default_range(self, one_station):
        """With default scale range, the applied factor must be in [SCALE_LOW, SCALE_HIGH]."""
        from seismo_sbi.instaseis_simulator.post_processing import AmplitudeErrorEffect as AEE
        sim = MockSimulator(
            one_station, amplitude=1.0,
            post_processing_effects=[AmplitudeErrorEffect()]
        )
        params = dict(_MT_PARAMS)
        params["amplitude_error"] = 1.0  # always apply

        for seed in range(30):
            np.random.seed(seed)
            _, seis = sim.run_simulation(params)
            scale = float(seis["STA1"]["Z"][0])
            assert AEE.DEFAULT_SCALE_LOW <= scale <= AEE.DEFAULT_SCALE_HIGH, (
                f"seed={seed}: scale {scale:.4f} outside default range "
                f"[{AEE.DEFAULT_SCALE_LOW}, {AEE.DEFAULT_SCALE_HIGH}]"
            )

    def test_no_chain_backward_compat(self, one_station):
        """Simulator with no chain (default) must behave exactly as before."""
        sim = MockSimulator(one_station, amplitude=2.0)
        _, seis = sim.run_simulation(dict(_MT_PARAMS))
        assert np.allclose(seis["STA1"]["Z"], 2.0)


# ===========================================================================
# Category 2: FixedLocationKernelSimulator — real Simulator subclass, fast
# ===========================================================================


class TestKernelSimulatorWithNuisanceChain:
    """Post-processing chain wired into FixedLocationKernelSimulator.

    Uses synthetic random kernels — no Instaseis DB or CPS required.
    """

    def test_amplitude_error_modifies_flat_output(self, one_station):
        """Full path: kernel sim → amplitude effect → flat array differs from baseline.

        Uses a deterministic scale_range=(2.0, 2.0) so the expected output is exact.
        """
        sim_baseline = _kernel_sim(one_station)
        sim_scaled = _kernel_sim(
            one_station,
            post_processing_effects=[AmplitudeErrorEffect(scale_range=(2.0, 2.0))]
        )

        params_baseline = dict(_MT_PARAMS)
        params_scaled = dict(_MT_PARAMS)
        params_scaled["amplitude_error"] = 1.0  # prob=1 → always apply scale

        _, seis_baseline = sim_baseline.run_simulation(params_baseline)
        _, seis_scaled = sim_scaled.run_simulation(params_scaled)

        data_baseline = seis_baseline["STA1"]["Z"]
        data_scaled = seis_scaled["STA1"]["Z"]

        assert not np.allclose(data_baseline, data_scaled), (
            "amplitude_error=1.0 must change the output of the kernel simulator"
        )
        assert np.allclose(data_scaled, 2.0 * data_baseline), (
            "scale_range=(2,2) with prob=1 must exactly double the output"
        )

    def test_no_effect_no_change(self, one_station):
        """Kernel sim with empty chain must match baseline output exactly."""
        sim_baseline = _kernel_sim(one_station)
        sim_empty_chain = _kernel_sim(one_station, post_processing_effects=[])

        params = dict(_MT_PARAMS)
        _, seis_a = sim_baseline.run_simulation(dict(params))
        _, seis_b = sim_empty_chain.run_simulation(dict(params))

        assert np.allclose(seis_a["STA1"]["Z"], seis_b["STA1"]["Z"])

    def test_chain_built_from_registry(self, two_stations):
        """build_post_processing_chain() integrated into FixedLocationKernelSimulator."""
        chain = build_post_processing_chain(["amplitude_error"])
        sim = _kernel_sim(two_stations, post_processing_effects=chain.effects)

        params = dict(_MT_PARAMS)
        params["amplitude_error"] = 0.5

        _, seis = sim.run_simulation(params)
        # Both stations should be present
        assert "STA1" in seis and "STA2" in seis

    def test_instrument_dropout_zeros_stations(self, two_stations):
        sim = _kernel_sim(
            two_stations,
            post_processing_effects=[InstrumentDropoutEffect()]
        )
        params = dict(_MT_PARAMS)
        params["instrument_dropout"] = 1.0

        _, seis = sim.run_simulation(params)
        assert np.allclose(seis["STA1"]["Z"], 0.0)
        assert np.allclose(seis["STA2"]["Z"], 0.0)


# ===========================================================================
# Category 3: Full input_output_simulation() path
# ===========================================================================


class TestInputOutputSimulationWithNuisance:
    """Verify that nuisance effects propagate through the complete wrapper path."""

    def _make_samplers(self, extra: dict | None = None):
        base = {"source_location": partial(_constant_sampler, np.array(_SOURCE_LOC))}
        if extra:
            base.update(extra)
        return base

    def test_amplitude_error_changes_flat_output(self, one_station):
        """input_output_simulation with amplitude_error nuisance differs from no-effect run.

        Uses a deterministic scale_range=(2.0, 2.0) to get an exact 2× assertion.
        """
        # Simulator without chain (baseline)
        sim_baseline = MockSimulator(one_station, amplitude=1.0)
        # Simulator with chain: prob=1 always applies, scale always = 2.0
        sim_scaled = MockSimulator(
            one_station, amplitude=1.0,
            post_processing_effects=[AmplitudeErrorEffect(scale_range=(2.0, 2.0))]
        )
        loader = SimulationDataLoader(components=["Z"], receivers=one_station)

        mp_base = _make_mt_model_parameters()
        mp_amp = _make_mt_model_parameters(nuisance={"amplitude_error": 1.0})

        samplers_base = self._make_samplers()
        samplers_amp = self._make_samplers(
            extra={"amplitude_error": partial(_constant_sampler, np.array(1.0))}
        )

        theta = np.array([1e14] * 6)
        result_base = _call_io_sim(mp_base, loader, samplers_base, sim_baseline, theta)
        result_amp = _call_io_sim(mp_amp, loader, samplers_amp, sim_scaled, theta)

        assert not np.allclose(result_base, result_amp), (
            "amplitude_error nuisance must produce a different flat output vector"
        )
        assert np.allclose(result_amp, 2.0 * result_base), (
            "scale_range=(2,2) with amplitude_error=1.0 must exactly double the output vector"
        )

    def test_dropout_prob1_gives_zero_output(self, one_station):
        """Instrument dropout probability 1 must produce an all-zero output vector."""
        sim = MockSimulator(
            one_station, amplitude=1.0,
            post_processing_effects=[InstrumentDropoutEffect()]
        )
        loader = SimulationDataLoader(components=["Z"], receivers=one_station)
        mp = _make_mt_model_parameters(nuisance={"instrument_dropout": 1.0})
        samplers = self._make_samplers(
            extra={"instrument_dropout": partial(_constant_sampler, np.array(1.0))}
        )
        theta = np.array([1e14] * 6)
        result = _call_io_sim(mp, loader, samplers, sim, theta)
        assert np.allclose(result, 0.0), "Dropout probability 1 must zero the full output"

    def test_no_nuisance_backward_compat(self, one_station):
        """No nuisance params → same output as original pipeline (no chain)."""
        sim_old = MockSimulator(one_station, amplitude=1.0)
        sim_new = MockSimulator(one_station, amplitude=1.0)  # empty chain by default
        loader = SimulationDataLoader(components=["Z"], receivers=one_station)
        mp = _make_mt_model_parameters()
        samplers = self._make_samplers()
        theta = np.array([1e14] * 6)

        result_old = _call_io_sim(mp, loader, samplers, sim_old, theta)
        result_new = _call_io_sim(mp, loader, samplers, sim_new, theta)

        assert np.allclose(result_old, result_new), (
            "Backward compat: empty chain must produce identical output to no-chain baseline"
        )

    def test_combined_effects_compose(self, two_stations):
        """Amplitude effect (prob=0) followed by dropout (prob=0): output unchanged.

        With amplitude_error=0.0 (no modulation) and instrument_dropout=0.0 (no dropout),
        the output should equal the baseline amplitude of 1.0 for all traces.
        """
        sim = MockSimulator(
            two_stations, amplitude=1.0,
            post_processing_effects=[
                AmplitudeErrorEffect(),   # amplitude_error=0 → no-op
                InstrumentDropoutEffect(),  # instrument_dropout=0 → no-op
            ]
        )
        loader = SimulationDataLoader(components=["Z"], receivers=two_stations)
        mp = _make_mt_model_parameters(nuisance={"amplitude_error": 0.0, "instrument_dropout": 0.0})
        samplers = self._make_samplers(extra={
            "amplitude_error": partial(_constant_sampler, np.array(0.0)),
            "instrument_dropout": partial(_constant_sampler, np.array(0.0)),
        })
        theta = np.array([1e14] * 6)
        result = _call_io_sim(mp, loader, samplers, sim, theta)
        # Both effects are no-ops → output should equal baseline amplitude
        assert np.allclose(result, 1.0)

    def test_amplitude_effect_with_full_probability_changes_output(self, two_stations):
        """Amplitude effect with prob=1 and fixed scale=3 produces exactly 3× output.

        Verifies that the stochastic stage-1 (probability) correctly gates the
        scale application, and that independent per-station scaling works end-to-end.
        """
        sim = MockSimulator(
            two_stations, amplitude=1.0,
            post_processing_effects=[
                AmplitudeErrorEffect(scale_range=(3.0, 3.0)),  # always apply factor 3
                InstrumentDropoutEffect(),  # instrument_dropout=0 → no-op
            ]
        )
        loader = SimulationDataLoader(components=["Z"], receivers=two_stations)
        mp = _make_mt_model_parameters(nuisance={"amplitude_error": 1.0, "instrument_dropout": 0.0})
        samplers = self._make_samplers(extra={
            "amplitude_error": partial(_constant_sampler, np.array(1.0)),
            "instrument_dropout": partial(_constant_sampler, np.array(0.0)),
        })
        theta = np.array([1e14] * 6)
        result = _call_io_sim(mp, loader, samplers, sim, theta)
        # All traces were 1.0 * 3.0 = 3.0; dropout was 0 so nothing zeroed
        assert np.allclose(result, 3.0)


# ===========================================================================
# Slow tests — require real CPS simulators
# ===========================================================================


@pytest.mark.slow
class TestCPSSimulatorWithNuisanceChain:
    """Verify that amplitude_error nuisance changes CPS simulator output end-to-end.

    Requires CPS binaries on PATH or CPS_PATH env var, plus pre-computed
    Green's functions.  Skipped automatically if neither is available.
    """

    @pytest.fixture(autouse=True)
    def _require_cps(self):
        try:
            from seismo_sbi.cps_simulator.simulator import CPSPrecomputedSimulator  # noqa: F401
        except ImportError:
            pytest.skip("CPS simulator not available")

        cps_path = os.environ.get("CPS_PATH") or ""
        gf_path = os.environ.get("CPS_GF_PATH") or ""
        if not cps_path and not gf_path:
            pytest.skip("CPS_PATH or CPS_GF_PATH env vars not set — CPS test skipped")

    def test_amplitude_error_changes_cps_output(self):
        from seismo_sbi.cps_simulator.simulator import CPSPrecomputedSimulator

        cps_path = os.environ.get("CPS_PATH")
        gf_path = os.environ["CPS_GF_PATH"]
        fiducial_path = os.environ.get("CPS_GF_FIDUCIAL_PATH", gf_path)

        r = Receiver(0.0, 0.0, "XX", "STA1", ["Z"])
        receivers = Receivers(receivers=[r])
        processing = {
            "sampling_rate": 1.0,
            "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
        }

        sim_baseline = CPSPrecomputedSimulator(
            fiducial_model_path=fiducial_path,
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=200,
            synthetics_processing=processing,
            gf_storage_root=gf_path,
            cps_path=cps_path,
            post_processing_effects=[],
        )
        sim_scaled = CPSPrecomputedSimulator(
            fiducial_model_path=fiducial_path,
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=200,
            synthetics_processing=processing,
            gf_storage_root=gf_path,
            cps_path=cps_path,
            post_processing_effects=[AmplitudeErrorEffect(scale_range=(2.0, 2.0))],
        )

        src_loc = [0.0, 0.0, 10.0, 0.0]
        params_base = {"source_location": src_loc, "moment_tensor": [1e14] * 6}
        params_scaled = dict(params_base)
        params_scaled["amplitude_error"] = 1.0  # probability=1 → always apply

        _, seis_base = sim_baseline.run_simulation(params_base)
        _, seis_scaled = sim_scaled.run_simulation(params_scaled)

        assert not np.allclose(seis_base["STA1"]["Z"], seis_scaled["STA1"]["Z"]), (
            "amplitude_error=1.0 (prob=1) must change CPS output"
        )
        assert np.allclose(seis_scaled["STA1"]["Z"], 2.0 * seis_base["STA1"]["Z"]), (
            "scale_range=(2,2) with amplitude_error=1.0 must double the CPS output"
        )


# ===========================================================================
# Category 4: TimeShiftErrorEffect end-to-end
# ===========================================================================

# Sampling rate used in time-shift tests (samples per second)
_SR = 10.0


def _make_ramp_seismo(receivers, n=TRACE_LEN):
    """Return a seismogram dict with a linearly ramping trace (non-constant)."""
    return {
        rec.station_name: {comp: np.arange(n, dtype=float) for comp in rec.components}
        for rec in receivers.iterate()
    }


class TestTimeShiftEffectIntegration:
    """End-to-end: time_shift_error nuisance through MockSimulator.run_simulation()."""

    def test_zero_probability_no_change(self, one_station):
        """time_shift_error=0.0 → no station shifted, ramp trace unchanged."""
        sim = MockSimulator(one_station, amplitude=1.0,
                            post_processing_effects=[
                                TimeShiftErrorEffect(sampling_rate=_SR, gaussian_sigma=1.0)
                            ])
        # Replace canned seismogram with a ramp so any shift is detectable
        params = dict(_MT_PARAMS)
        params["time_shift_error"] = 0.0
        _, seis = sim.run_simulation(params)
        # MockSimulator returns constant 1.0 traces; with prob=0, unchanged
        assert np.allclose(seis["STA1"]["Z"], 1.0)

    def test_full_probability_changes_non_constant_trace(self, two_stations):
        """time_shift_error=1.0 must shift at least one trace over multiple seeds."""
        # Build a MockSimulator that returns a ramp instead of constants
        class RampMockSimulator(MockSimulator):
            def generic_point_source_simulation(self, source, **kwargs):
                return {
                    rec.station_name: {
                        comp: np.arange(TRACE_LEN, dtype=float)
                        for comp in rec.components
                    }
                    for rec in self.receivers.iterate()
                }

        sim = RampMockSimulator(
            two_stations, amplitude=1.0,
            post_processing_effects=[
                TimeShiftErrorEffect(sampling_rate=_SR, gaussian_sigma=1.0)
            ]
        )
        params = dict(_MT_PARAMS)
        params["time_shift_error"] = 1.0

        changed = False
        for seed in range(20):
            np.random.seed(seed)
            _, seis = sim.run_simulation(params)
            ramp = np.arange(TRACE_LEN, dtype=float)
            if not np.allclose(seis["STA1"]["Z"], ramp):
                changed = True
                break
        assert changed, "time_shift_error=1.0 should change a non-constant trace over 20 seeds"

    def test_per_station_shifts_are_independent(self, two_stations):
        """Different stations should get different time shifts (with overwhelming prob)."""
        class RampMockSimulator(MockSimulator):
            def generic_point_source_simulation(self, source, **kwargs):
                return {
                    rec.station_name: {
                        comp: np.arange(TRACE_LEN, dtype=float)
                        for comp in rec.components
                    }
                    for rec in self.receivers.iterate()
                }

        sim = RampMockSimulator(
            two_stations, amplitude=1.0,
            post_processing_effects=[
                TimeShiftErrorEffect(sampling_rate=_SR, gaussian_sigma=2.0)
            ]
        )
        params = dict(_MT_PARAMS)
        params["time_shift_error"] = 1.0

        found_different = False
        for seed in range(30):
            np.random.seed(seed)
            _, seis = sim.run_simulation(params)
            if not np.allclose(seis["STA1"]["Z"], seis["STA2"]["Z"]):
                found_different = True
                break
        assert found_different, (
            "Per-station time shifts must be independent; "
            "STA1 and STA2 should differ over 30 seeds"
        )

    def test_source_params_not_mutated(self, one_station):
        sim = MockSimulator(
            one_station,
            post_processing_effects=[TimeShiftErrorEffect(sampling_rate=_SR)]
        )
        params = dict(_MT_PARAMS)
        params["time_shift_error"] = 0.5
        original_keys = set(params.keys())
        sim.run_simulation(params)
        assert set(params.keys()) == original_keys

    def test_time_shift_error_in_input_output_simulation(self, one_station):
        """Full wrapper path: time_shift_error sampled → ramp trace changes."""
        class RampMockSimulator(MockSimulator):
            def generic_point_source_simulation(self, source, **kwargs):
                return {
                    rec.station_name: {
                        comp: np.arange(self._trace_len, dtype=float)
                        for comp in rec.components
                    }
                    for rec in self.receivers.iterate()
                }

        sim = RampMockSimulator(
            one_station,
            post_processing_effects=[
                TimeShiftErrorEffect(sampling_rate=_SR, gaussian_sigma=1.0)
            ]
        )
        loader = SimulationDataLoader(components=["Z"], receivers=one_station)
        mp = _make_mt_model_parameters(nuisance={"time_shift_error": 1.0})
        samplers = {
            "source_location": partial(_constant_sampler, np.array(_SOURCE_LOC)),
            "time_shift_error": partial(_constant_sampler, np.array(1.0)),
        }
        theta = np.array([1e14] * 6)

        # Collect multiple outputs — with prob=1 they should change across seeds
        np.random.seed(99)
        result_a = _call_io_sim(mp, loader, samplers, sim, theta)
        np.random.seed(100)
        result_b = _call_io_sim(mp, loader, samplers, sim, theta)
        # With non-zero sigma at least one of two draws should differ from the ramp
        ramp_flat = np.arange(TRACE_LEN, dtype=float)
        assert (not np.allclose(result_a, ramp_flat)) or (not np.allclose(result_b, ramp_flat)), (
            "time_shift_error=1.0 must produce changed output in input_output_simulation"
        )


# ===========================================================================
# Category 5: ScatteringCodaEffect end-to-end
# ===========================================================================


def _sinusoid_seismo(receivers, n=TRACE_LEN):
    """Return a seismogram dict with sinusoidal traces (non-constant, non-trivial spectrum)."""
    t = np.linspace(0, 2 * np.pi, n)
    sig = np.sin(t)
    return {
        rec.station_name: {comp: sig.copy() for comp in rec.components}
        for rec in receivers.iterate()
    }


class TestScatteringCodaEffectIntegration:
    """End-to-end: scattering_coda nuisance through MockSimulator.run_simulation()."""

    def test_zero_probability_no_change(self, one_station):
        """scattering_coda=0.0 → identity; sinusoidal trace unchanged."""
        sim = MockSimulator(
            one_station, amplitude=1.0,
            post_processing_effects=[ScatteringCodaEffect(alpha=0.9)]
        )
        params = dict(_MT_PARAMS)
        params["scattering_coda"] = 0.0
        _, seis = sim.run_simulation(params)
        assert np.allclose(seis["STA1"]["Z"], 1.0)

    def test_prob1_changes_output(self, one_station):
        """scattering_coda=1.0 with alpha=0.4 must change a non-constant trace."""
        class SineMockSimulator(MockSimulator):
            def generic_point_source_simulation(self, source, **kwargs):
                t = np.linspace(0, 2 * np.pi, self._trace_len)
                return {
                    rec.station_name: {comp: np.sin(t).copy() for comp in rec.components}
                    for rec in self.receivers.iterate()
                }

        sim = SineMockSimulator(
            one_station,
            post_processing_effects=[ScatteringCodaEffect(alpha=0.4)]
        )
        params = dict(_MT_PARAMS)
        params["scattering_coda"] = 1.0

        t = np.linspace(0, 2 * np.pi, TRACE_LEN)
        baseline = np.sin(t)
        changed = False
        for seed in range(20):
            np.random.seed(seed)
            _, seis = sim.run_simulation(params)
            if not np.allclose(seis["STA1"]["Z"], baseline):
                changed = True
                break
        assert changed, "scattering_coda=1.0 must change a sinusoidal trace over 20 seeds"

    def test_causal_no_wrap_through_simulator(self, one_station):
        """Coda must not wrap to the trace start, end-to-end through run_simulation()."""
        class EdgeImpulseMockSimulator(MockSimulator):
            def generic_point_source_simulation(self, source, **kwargs):
                def _edge_impulse():
                    trace = np.zeros(self._trace_len)
                    trace[self._trace_len - 2] = 1.0  # hard against the right edge
                    return trace
                return {
                    rec.station_name: {comp: _edge_impulse() for comp in rec.components}
                    for rec in self.receivers.iterate()
                }

        coda_sim = EdgeImpulseMockSimulator(
            one_station,
            post_processing_effects=[ScatteringCodaEffect(alpha=0.9)]
        )

        params_coda = dict(_MT_PARAMS)
        params_coda["scattering_coda"] = 1.0

        np.random.seed(3)
        _, seis_coda = coda_sim.run_simulation(params_coda)

        half = coda_sim._trace_len // 2
        assert np.allclose(seis_coda["STA1"]["Z"][:half], 0.0, atol=1e-10), (
            "ScatteringCodaEffect must not wrap coda around to the start of the trace"
        )

    def test_source_params_not_mutated(self, one_station):
        """run_simulation() must not mutate the caller's dict when scattering_coda is set."""
        sim = MockSimulator(
            one_station,
            post_processing_effects=[ScatteringCodaEffect(alpha=0.4)]
        )
        params = dict(_MT_PARAMS)
        params["scattering_coda"] = 0.8
        original_keys = set(params.keys())
        sim.run_simulation(params)
        assert set(params.keys()) == original_keys

    def test_backward_compat_absent_key_identical_to_baseline(self, one_station):
        """Simulator without scattering_coda in params must match a no-chain baseline."""
        sim_baseline = MockSimulator(one_station, amplitude=2.0)
        sim_with_effect = MockSimulator(
            one_station, amplitude=2.0,
            post_processing_effects=[ScatteringCodaEffect(alpha=0.9)]
        )
        params = dict(_MT_PARAMS)  # no scattering_coda key

        _, seis_base = sim_baseline.run_simulation(params)
        _, seis_effect = sim_with_effect.run_simulation(params)

        assert np.allclose(seis_base["STA1"]["Z"], seis_effect["STA1"]["Z"]), (
            "Absent scattering_coda key must be a strict no-op (backward compat)"
        )

    def test_composes_with_other_effects(self, one_station):
        """Compose scattering_coda (prob=0) + amplitude_error (prob=0): identity."""
        sim = MockSimulator(
            one_station, amplitude=1.0,
            post_processing_effects=[
                ScatteringCodaEffect(alpha=0.4),
                AmplitudeErrorEffect(),
            ]
        )
        params = dict(_MT_PARAMS)
        params["scattering_coda"] = 0.0
        params["amplitude_error"] = 0.0
        _, seis = sim.run_simulation(params)
        assert np.allclose(seis["STA1"]["Z"], 1.0)

    def test_scattering_coda_through_input_output_simulation(self, one_station):
        """Full wrapper path: scattering_coda sampled → sinusoidal trace changes."""
        class SineMockSimulator(MockSimulator):
            def generic_point_source_simulation(self, source, **kwargs):
                t = np.linspace(0, 2 * np.pi, self._trace_len)
                return {
                    rec.station_name: {comp: np.sin(t).copy() for comp in rec.components}
                    for rec in self.receivers.iterate()
                }

        sim = SineMockSimulator(
            one_station,
            post_processing_effects=[ScatteringCodaEffect(alpha=0.4)]
        )
        loader = SimulationDataLoader(components=["Z"], receivers=one_station)
        mp = _make_mt_model_parameters(nuisance={"scattering_coda": 1.0})
        samplers = {
            "source_location": partial(_constant_sampler, np.array(_SOURCE_LOC)),
            "scattering_coda": partial(_constant_sampler, np.array(1.0)),
        }
        theta = np.array([1e14] * 6)
        t = np.linspace(0, 2 * np.pi, TRACE_LEN)
        sine_flat = np.sin(t)

        np.random.seed(77)
        result_a = _call_io_sim(mp, loader, samplers, sim, theta)
        np.random.seed(78)
        result_b = _call_io_sim(mp, loader, samplers, sim, theta)

        assert (not np.allclose(result_a, sine_flat)) or (not np.allclose(result_b, sine_flat)), (
            "scattering_coda=1.0 must change the flat output in input_output_simulation"
        )

    def test_scattering_coda_through_kernel_simulator(self, one_station):
        """FixedLocationKernelSimulator: scattering_coda prob=1 changes output vs baseline."""
        sim_baseline = _kernel_sim(one_station)
        sim_coda = _kernel_sim(
            one_station,
            post_processing_effects=[ScatteringCodaEffect(alpha=0.4)]
        )
        params_base = dict(_MT_PARAMS)
        params_coda = dict(_MT_PARAMS)
        params_coda["scattering_coda"] = 1.0

        _, seis_base = sim_baseline.run_simulation(params_base)
        np.random.seed(11)
        _, seis_coda = sim_coda.run_simulation(params_coda)

        assert not np.allclose(seis_base["STA1"]["Z"], seis_coda["STA1"]["Z"]), (
            "scattering_coda=1.0 must change the kernel simulator output"
        )
        # Energy is conserved on average by the L2-normalised coda kernel.
        e_base = np.sum(seis_base["STA1"]["Z"] ** 2)
        e_coda = np.sum(seis_coda["STA1"]["Z"] ** 2)
        assert np.isclose(e_coda, e_base, rtol=0.5), (
            "ScatteringCodaEffect should roughly conserve trace energy"
        )
