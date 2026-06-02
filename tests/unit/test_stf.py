"""Unit tests for the GCMT triangular source time function (STF).

Tests are organised in four sections:

1. **Scalar moment helpers** — ``_scalar_moment`` and ``_gcmt_half_duration``.
2. **Triangular STF shape** — ``_build_triangular_stf`` geometry and
   ``build_stf_sliprate`` with a GCMT half-duration.
3. **Dirac delta** — ``build_stf_sliprate(None, dt)`` backward-compat path.
4. **Dispatch** — ``Simulator.run_simulation()`` pops ``stf_duration`` and
   forwards it to ``generic_point_source_simulation()`` as a kwarg.

No Instaseis database is required for any test here.
"""

from __future__ import annotations

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.wrapper import (
    GCMT_SCALE_FACTOR,
    _MIN_STF_SAMPLES,
    _scalar_moment,
    _gcmt_half_duration,
    _build_triangular_stf,
    build_stf_sliprate,
    GenericPointSource,
    GeneralMomentTensor,
)
from seismo_sbi.instaseis_simulator.simulator import Simulator
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers


# ---------------------------------------------------------------------------
# Minimal MockSimulator (avoids any real forward-model calls)
# ---------------------------------------------------------------------------

TRACE_LEN = 40
_SOURCE_LOC = [0.0, 0.0, 5.0, 0.0]


class MockSimulator(Simulator):
    """Concrete Simulator that records kwargs forwarded to it."""

    def __init__(self, receivers, trace_len=TRACE_LEN):
        super().__init__(
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=trace_len,
            synthetics_processing={
                "sampling_rate": 1.0,
                "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
            },
        )
        self._trace_len = trace_len
        self.last_kwargs: dict = {}

    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        self.last_kwargs = dict(kwargs)
        return {
            rec.station_name: {comp: np.ones(self._trace_len) for comp in rec.components}
            for rec in self.receivers.iterate()
        }


@pytest.fixture
def one_station_receivers():
    r = Receiver(latitude=0.0, longitude=0.0, network="XX", station_name="STA1", components=["Z"])
    return Receivers(receivers=[r])


@pytest.fixture
def mock_sim(one_station_receivers):
    return MockSimulator(one_station_receivers)


def _mt_params(stf_duration=None):
    p = {"source_location": _SOURCE_LOC, "moment_tensor": [1e14] * 6}
    if stf_duration is not None:
        p["stf_duration"] = [stf_duration]
    return p


# ===========================================================================
# Section 1: Scalar moment and GCMT half-duration helpers
# ===========================================================================


class TestScalarMoment:
    """Tests for ``_scalar_moment(mt_components)``."""

    def test_isotropic_explosion(self):
        """A pure isotropic source [M, M, M, 0, 0, 0] has M₀ = M√(3/2)."""
        M = 1e15
        mt = np.array([M, M, M, 0.0, 0.0, 0.0])
        # Frobenius: sqrt(0.5 * (M²+M²+M²)) = M * sqrt(3/2)
        expected = M * np.sqrt(1.5)
        assert _scalar_moment(mt) == pytest.approx(expected, rel=1e-10)

    def test_pure_double_couple(self):
        """A canonical DC [M, -M, 0, 0, 0, 0] has M₀ = M."""
        M = 2e14
        mt = np.array([M, -M, 0.0, 0.0, 0.0, 0.0])
        # Frobenius: sqrt(0.5 * (M² + M²)) = M
        assert _scalar_moment(mt) == pytest.approx(M, rel=1e-10)

    def test_zero_tensor(self):
        assert _scalar_moment(np.zeros(6)) == pytest.approx(0.0)

    def test_scales_with_amplitude(self):
        """M₀ is linear in the MT amplitude."""
        base = np.array([1e14, -5e13, -5e13, 1e13, 0.0, 0.0])
        m0_1 = _scalar_moment(base)
        m0_2 = _scalar_moment(2.0 * base)
        assert m0_2 == pytest.approx(2.0 * m0_1, rel=1e-10)

    def test_returns_float(self):
        mt = np.array([1e15] * 6)
        assert isinstance(_scalar_moment(mt), float)


class TestGcmtHalfDuration:
    """Tests for ``_gcmt_half_duration(mt_components)``."""

    def test_known_value(self):
        """T_half = GCMT_SCALE_FACTOR * M0^(1/3) for a pure DC."""
        M = 1e15  # N·m  (Mw ≈ 4)
        mt = np.array([M, -M, 0.0, 0.0, 0.0, 0.0])
        m0 = _scalar_moment(mt)  # = M
        expected = GCMT_SCALE_FACTOR * m0 ** (1.0 / 3.0)
        assert _gcmt_half_duration(mt) == pytest.approx(expected, rel=1e-10)

    def test_cube_root_scaling(self):
        """Increasing M₀ by 8× (two magnitude steps) doubles T_half."""
        mt_small = np.array([1e15, -1e15, 0.0, 0.0, 0.0, 0.0])
        mt_large = 8.0 * mt_small
        t_small = _gcmt_half_duration(mt_small)
        t_large = _gcmt_half_duration(mt_large)
        assert t_large == pytest.approx(2.0 * t_small, rel=1e-6)

    def test_positive(self):
        mt = np.array([1e14, -1e14, 0.0, 0.0, 0.0, 0.0])
        assert _gcmt_half_duration(mt) > 0.0

    def test_realistic_mw4(self):
        """Mw≈4 event (DC representation) should give T_half ≈ 0.26 s.

        For a pure double-couple [M, -M, 0, 0, 0, 0], _scalar_moment returns M
        exactly.  Mw=4 → M0 ≈ 1.26e15 N·m → T_half ≈ 0.26 s via GCMT formula.
        """
        # DC: _scalar_moment([m0, -m0, 0, 0, 0, 0]) = m0 exactly
        m0 = 10 ** (1.5 * 4 + 9.1)   # ≈ 1.26e15 N·m
        mt = np.array([m0, -m0, 0.0, 0.0, 0.0, 0.0])
        t_half = _gcmt_half_duration(mt)
        expected = GCMT_SCALE_FACTOR * m0 ** (1.0 / 3.0)   # ≈ 0.259 s
        assert t_half == pytest.approx(expected, rel=1e-6)
        assert 0.1 < t_half < 0.5, f"T_half = {t_half:.3f} s unexpected for Mw≈4"

    def test_realistic_mw7(self):
        """Mw≈7 event (DC representation) should give T_half ≈ 8 s."""
        m0 = 10 ** (1.5 * 7 + 9.1)   # ≈ 3.98e19 N·m
        mt = np.array([m0, -m0, 0.0, 0.0, 0.0, 0.0])
        t_half = _gcmt_half_duration(mt)
        expected = GCMT_SCALE_FACTOR * m0 ** (1.0 / 3.0)   # ≈ 8.2 s
        assert t_half == pytest.approx(expected, rel=1e-6)
        assert 5.0 < t_half < 15.0, f"T_half = {t_half:.3f} s unexpected for Mw≈7"


# ===========================================================================
# Section 2: Triangular STF shape tests
# ===========================================================================


class TestBuildTriangularStf:
    """Tests for the internal ``_build_triangular_stf`` helper."""

    def test_starts_at_zero(self):
        s = _build_triangular_stf(1.0, dt=0.1)
        assert s[0] == pytest.approx(0.0)

    def test_ends_at_zero(self):
        s = _build_triangular_stf(1.0, dt=0.1)
        # last non-padded value should be ~0
        assert s[-1] == pytest.approx(0.0, abs=1e-10)

    def test_peak_at_half_duration(self):
        """Peak is at index corresponding to T_half."""
        dt = 0.1
        T_half = 1.0
        s = _build_triangular_stf(T_half, dt=dt)
        peak_idx = np.argmax(s)
        expected_peak_time = peak_idx * dt
        assert expected_peak_time == pytest.approx(T_half, abs=dt)

    def test_peak_value(self):
        """Peak = 1/T_half (unit-area triangle)."""
        T_half = 2.0
        dt = 0.05
        s = _build_triangular_stf(T_half, dt=dt)
        assert np.max(s) == pytest.approx(1.0 / T_half, rel=0.01)

    def test_unit_area(self):
        """Riemann sum ≈ 1 (analytic area of isosceles triangle = 1)."""
        dt = 0.05
        s = _build_triangular_stf(2.0, dt=dt)
        area = np.sum(s) * dt
        assert area == pytest.approx(1.0, rel=0.01)

    def test_non_negative(self):
        s = _build_triangular_stf(1.5, dt=0.1)
        assert np.all(s >= 0.0)

    def test_rise_is_linear(self):
        """Values in the rising half are linearly increasing."""
        dt = 0.1
        T_half = 1.0
        s = _build_triangular_stf(T_half, dt=dt)
        n_rise = int(round(T_half / dt))
        diffs = np.diff(s[:n_rise])
        assert np.all(diffs >= 0.0), "Rise should be monotonically increasing"
        # differences should be approximately constant (linearity)
        np.testing.assert_allclose(diffs, diffs[0], rtol=0.02)

    def test_fall_is_linear(self):
        """Values in the falling half are linearly decreasing."""
        dt = 0.1
        T_half = 1.0
        s = _build_triangular_stf(T_half, dt=dt)
        n_rise = int(round(T_half / dt))
        n_fall = int(round(T_half / dt))
        fall = s[n_rise : n_rise + n_fall]
        diffs = np.diff(fall)
        assert np.all(diffs <= 0.0), "Fall should be monotonically decreasing"
        np.testing.assert_allclose(diffs, diffs[0], rtol=0.02)

    def test_total_duration(self):
        """The active support is approximately 2·T_half."""
        dt = 0.05
        T_half = 1.0
        s = _build_triangular_stf(T_half, dt=dt)
        n_active = len(s)
        total_duration = (n_active - 1) * dt
        assert total_duration == pytest.approx(2.0 * T_half, abs=dt)

    def test_returns_float64(self):
        s = _build_triangular_stf(1.0, dt=0.1)
        assert s.dtype == np.float64

    @pytest.mark.parametrize("T_half", [0.5, 1.0, 2.0, 5.0])
    def test_symmetric(self, T_half):
        """The triangle is symmetric: rise values mirror fall values."""
        dt = 0.05
        s = _build_triangular_stf(T_half, dt=dt)
        peak_idx = np.argmax(s)
        # Build the rise and fall arms (same number of samples)
        arm_len = min(peak_idx, len(s) - peak_idx - 1)
        rise = s[peak_idx - arm_len : peak_idx]
        fall = s[peak_idx + 1 : peak_idx + 1 + arm_len]
        np.testing.assert_allclose(rise, fall[::-1], rtol=0.02)


class TestBuildStfSliprateWithGcmt:
    """Tests for ``build_stf_sliprate`` with a GCMT baseline."""

    def test_scale_factor_one_reproduces_baseline(self):
        """scale_factor=1.0 → effective T_half equals the GCMT prediction."""
        T_half = 2.0
        dt = 0.1
        s = build_stf_sliprate(1.0, dt=dt, gcmt_half_duration=T_half)
        peak_idx = np.argmax(s)
        assert peak_idx * dt == pytest.approx(T_half, abs=dt)

    def test_scale_factor_doubles_duration(self):
        """scale_factor=2.0 → peak appears at twice the baseline T_half."""
        T_half = 1.0
        dt = 0.05
        s1 = build_stf_sliprate(1.0, dt=dt, gcmt_half_duration=T_half)
        s2 = build_stf_sliprate(2.0, dt=dt, gcmt_half_duration=T_half)
        peak1 = np.argmax(s1) * dt
        peak2 = np.argmax(s2) * dt
        assert peak2 == pytest.approx(2.0 * peak1, rel=0.05)

    def test_scale_factor_halves_duration(self):
        """scale_factor=0.5 → peak appears at half the baseline T_half."""
        T_half = 2.0
        dt = 0.05
        s1 = build_stf_sliprate(1.0, dt=dt, gcmt_half_duration=T_half)
        s2 = build_stf_sliprate(0.5, dt=dt, gcmt_half_duration=T_half)
        peak1 = np.argmax(s1) * dt
        peak2 = np.argmax(s2) * dt
        assert peak2 == pytest.approx(0.5 * peak1, rel=0.05)

    def test_accepts_array_scale_factor(self):
        """scale_factor may arrive as a 1-element array from the nuisance sampler."""
        s = build_stf_sliprate(np.array([1.0]), dt=0.1, gcmt_half_duration=1.0)
        assert len(s) >= _MIN_STF_SAMPLES

    def test_min_length_enforced(self):
        """Very short events should still be padded to _MIN_STF_SAMPLES."""
        # M0 for Mw ≈ 2 → very short T_half (≈0.1 s); with dt=0.5 → few samples
        s = build_stf_sliprate(1.0, dt=0.5, gcmt_half_duration=0.05)
        assert len(s) >= _MIN_STF_SAMPLES

    def test_non_negative(self):
        s = build_stf_sliprate(1.5, dt=0.1, gcmt_half_duration=1.0)
        assert np.all(s >= 0.0)

    def test_raises_if_gcmt_half_duration_zero(self):
        """Passing gcmt_half_duration=0 with a non-None scale factor must raise."""
        with pytest.raises(ValueError, match="gcmt_half_duration must be positive"):
            build_stf_sliprate(1.0, dt=0.1, gcmt_half_duration=0.0)

    def test_raises_if_gcmt_half_duration_missing(self):
        """Calling with a scale factor but omitting gcmt_half_duration must raise."""
        with pytest.raises(ValueError):
            build_stf_sliprate(1.0, dt=0.1)  # defaults to 0.0

    def test_returns_float64(self):
        s = build_stf_sliprate(1.0, dt=0.1, gcmt_half_duration=1.0)
        assert s.dtype == np.float64

    def test_subsample_half_duration_falls_back_to_dirac(self):
        """A half-duration at/below ~dt/2 is unresolvable and would discretise to a
        zero-area triangle (NaN under Instaseis set_sliprate(normalize=True)); it must fall
        back to a Dirac impulse instead."""
        s = build_stf_sliprate(1.0, dt=1.0, gcmt_half_duration=0.1)
        assert s[0] == pytest.approx(1.0)
        assert np.all(s[1:] == 0.0)

    def test_always_positive_finite_area_at_1hz(self):
        """Across the small-Mw regime the sliprate must have positive, finite area at the
        1 Hz sampling used by the Santorini config, so normalisation never produces NaN.
        This is the regression guard for the STF-sampling NaN found in the first smoke run."""
        for gcmt_half in (0.02, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0):
            for scale in (0.5, 1.0, 2.0):
                s = build_stf_sliprate(scale, dt=1.0, gcmt_half_duration=gcmt_half)
                area = np.trapz(s, dx=1.0)
                assert np.isfinite(area) and area > 0.0, (gcmt_half, scale, area)


# ===========================================================================
# Section 3: Dirac delta backward-compat path
# ===========================================================================


class TestBuildStfSliprateDirac:
    """Tests for the Dirac-delta (``stf_duration=None``) branch."""

    def test_none_produces_spike_at_zero(self):
        s = build_stf_sliprate(None, dt=0.5)
        assert s[0] == pytest.approx(1.0)

    def test_none_zeros_elsewhere(self):
        s = build_stf_sliprate(None, dt=0.5)
        assert np.all(s[1:] == 0.0)

    def test_none_min_length(self):
        s = build_stf_sliprate(None, dt=0.5)
        assert len(s) >= _MIN_STF_SAMPLES

    def test_none_returns_float_array(self):
        s = build_stf_sliprate(None, dt=0.5)
        assert s.dtype.kind == "f"

    def test_none_ignores_gcmt_half_duration(self):
        """gcmt_half_duration is irrelevant when stf_duration is None."""
        s = build_stf_sliprate(None, dt=0.5, gcmt_half_duration=5.0)
        assert s[0] == pytest.approx(1.0)
        assert np.all(s[1:] == 0.0)


# ===========================================================================
# Section 4: Dispatch — stf_duration flows through run_simulation()
# ===========================================================================


class TestStfDurationDispatch:
    """Confirm run_simulation() pops stf_duration and forwards it correctly."""

    def test_no_stf_duration_passes_none(self, mock_sim):
        """When ``stf_duration`` is absent, ``None`` is forwarded."""
        mock_sim.run_simulation(_mt_params())
        assert mock_sim.last_kwargs.get("stf_duration") is None

    def test_stf_duration_forwarded_as_kwarg(self, mock_sim):
        """When ``stf_duration`` is in source_parameters it reaches the simulator."""
        mock_sim.run_simulation(_mt_params(stf_duration=2.0))
        received = mock_sim.last_kwargs.get("stf_duration")
        assert float(np.squeeze(received)) == pytest.approx(2.0)

    def test_stf_duration_forwarded_from_array(self, mock_sim):
        """The nuisance sampler yields a 1-element array; the content must arrive."""
        params = {
            "source_location": _SOURCE_LOC,
            "moment_tensor": [1e14] * 6,
            "stf_duration": np.array([3.5]),
        }
        mock_sim.run_simulation(params)
        received = mock_sim.last_kwargs.get("stf_duration")
        assert float(np.squeeze(received)) == pytest.approx(3.5)

    def test_stf_duration_not_in_post_proc_params(self, mock_sim):
        """stf_duration must NOT appear in the post-processing nuisance dict."""
        received_nuisance: dict = {}
        original_chain = mock_sim.post_processing_chain

        class CapturingChain:
            def __call__(self_, smap, receivers, nuisance_params):
                received_nuisance.update(nuisance_params)
                return original_chain(smap, receivers, nuisance_params)

        mock_sim.post_processing_chain = CapturingChain()
        mock_sim.run_simulation(_mt_params(stf_duration=1.0))
        assert "stf_duration" not in received_nuisance

    def test_source_params_not_mutated(self, mock_sim):
        """run_simulation must not modify the caller's dict."""
        params = _mt_params(stf_duration=1.5)
        original = dict(params)
        mock_sim.run_simulation(params)
        assert params == original

    def test_forwarded_but_not_post_processed(self, mock_sim):
        """Jointly: stf_duration reaches generic_point_source_simulation AND
        is excluded from the post-processing nuisance dict."""
        received_pp: dict = {}
        original_chain = mock_sim.post_processing_chain

        class SnoopChain:
            def __call__(self_, smap, receivers, nuisance_params):
                received_pp.update(nuisance_params)
                return original_chain(smap, receivers, nuisance_params)

        mock_sim.post_processing_chain = SnoopChain()
        mock_sim.run_simulation(_mt_params(stf_duration=4.0))
        # Forwarded to simulator: yes
        assert float(np.squeeze(mock_sim.last_kwargs.get("stf_duration"))) == pytest.approx(4.0)
        # Leaked into post-processing: no
        assert "stf_duration" not in received_pp
