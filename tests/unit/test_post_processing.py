"""Tests for the post-processing effect framework.

These tests describe the expected behaviour of the new
`seismo_sbi.instaseis_simulator.post_processing` module and will initially
fail (ImportError) until that module is implemented.  Once the refactor is
complete every test here must pass to confirm the framework is correct.

Design contract
---------------
- `PostProcessingChain([])` is the identity: seismograms pass through unchanged.
- Each `SeismogramEffect` subclass extracts only its own key from `nuisance_params`
  and is a no-op when that key is absent.
- Effects compose sequentially; later effects see the output of earlier ones.
- `build_post_processing_chain(nuisance_keys)` returns a chain containing only
  effects registered in `EFFECT_REGISTRY` for the given keys; unknown keys are
  silently skipped.
"""

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers

# ---------------------------------------------------------------------------
# The module under test — will raise ImportError until implemented.
# Tests are collected but skipped gracefully via the importorskip mechanism.
# ---------------------------------------------------------------------------

post_processing = pytest.importorskip(
    "seismo_sbi.instaseis_simulator.post_processing",
    reason="post_processing module not yet implemented",
)

PostProcessingChain = post_processing.PostProcessingChain
build_post_processing_chain = post_processing.build_post_processing_chain
AmplitudeErrorEffect = post_processing.AmplitudeErrorEffect
InstrumentDropoutEffect = post_processing.InstrumentDropoutEffect
TimeShiftErrorEffect = post_processing.TimeShiftErrorEffect
ScatteringCodaEffect = post_processing.ScatteringCodaEffect
_apply_lanczos_shift = post_processing._apply_lanczos_shift
_apply_random_coda_filter = post_processing._apply_random_coda_filter
_apply_stahler_phase_filter = post_processing._apply_stahler_phase_filter
_lanczos_kernel_values = post_processing._lanczos_kernel_values


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TRACE_LEN = 40


def _make_receivers(*station_names):
    """Build a Receivers object with one Z component per station."""
    recs = [
        Receiver(float(i), float(i), "XX", name, ["Z"])
        for i, name in enumerate(station_names)
    ]
    return Receivers(receivers=recs)


def _make_seismo_map(receivers, amplitude=1.0):
    """Return a seismogram dict with constant-valued traces."""
    return {
        rec.station_name: {comp: np.full(TRACE_LEN, amplitude) for comp in rec.components}
        for rec in receivers.iterate()
    }


@pytest.fixture
def one_station():
    return _make_receivers("STA1")


@pytest.fixture
def two_stations():
    return _make_receivers("STA1", "STA2")


# ===========================================================================
# PostProcessingChain
# ===========================================================================

class TestPostProcessingChain:

    def test_empty_chain_returns_unchanged_values(self, one_station):
        chain = PostProcessingChain()
        seismo = _make_seismo_map(one_station)
        out = chain(seismo, one_station, {})
        assert np.array_equal(out["STA1"]["Z"], seismo["STA1"]["Z"])

    def test_empty_chain_returns_new_object_or_same(self, one_station):
        """The chain may return the same dict or a copy — but must not error."""
        chain = PostProcessingChain()
        seismo = _make_seismo_map(one_station)
        out = chain(seismo, one_station, {})
        # Values must be identical regardless of whether it is a copy
        assert np.allclose(out["STA1"]["Z"], 1.0)

    def test_single_effect_applied(self, one_station):
        """A chain with one amplitude effect (prob=1, fixed scale) should scale the trace."""
        chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(2.0, 2.0))])
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        out = chain(seismo, one_station, {"amplitude_error": 1.0})
        assert np.allclose(out["STA1"]["Z"], 2.0)

    def test_two_effects_compose_sequentially(self, one_station):
        """Two amplitude effects with fixed scale_ranges compose multiplicatively."""
        # scale_range=(2.0,2.0) → always applies factor 2;
        # scale_range=(3.0,3.0) → always applies factor 3
        # Combined result: 1.0 * 2.0 * 3.0 = 6.0
        effect_a = AmplitudeErrorEffect(scale_range=(2.0, 2.0))
        effect_b = AmplitudeErrorEffect(scale_range=(3.0, 3.0))
        chain = PostProcessingChain([effect_a, effect_b])
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        out = chain(seismo, one_station, {"amplitude_error": 1.0})
        assert np.allclose(out["STA1"]["Z"], 6.0)

    def test_nuisance_params_without_registered_key_are_ignored(self, one_station):
        """Extra nuisance params that no effect cares about must not raise."""
        # amplitude_error=0.0 → identity (no modulation), so the check is trivial
        chain = PostProcessingChain([AmplitudeErrorEffect()])
        seismo = _make_seismo_map(one_station)
        out = chain(seismo, one_station, {"amplitude_error": 0.0, "unknown_future_param": 99})
        assert np.allclose(out["STA1"]["Z"], 1.0)


# ===========================================================================
# AmplitudeErrorEffect
# ===========================================================================

class TestAmplitudeErrorEffect:
    """Tests for the two-stage stochastic amplitude modulation effect.

    ``amplitude_error`` is a **probability** (analogous to ``instrument_dropout``):
    each station is independently modulated with this probability.  When modulated,
    a per-station scale factor is drawn uniformly from [SCALE_LOW, SCALE_HIGH].

    Pass ``scale_range=(x, x)`` to the constructor to make the scale deterministic
    in tests that need exact arithmetic.
    """

    # ------------------------------------------------------------------
    # Identity / no-op behaviour
    # ------------------------------------------------------------------

    def test_zero_probability_is_identity(self, one_station):
        """amplitude_error=0.0 → no station is ever modulated."""
        effect = AmplitudeErrorEffect()
        seismo = _make_seismo_map(one_station, amplitude=3.7)
        out = effect(seismo, one_station, amplitude_error=0.0)
        assert np.allclose(out["STA1"]["Z"], 3.7)

    def test_absent_key_is_noop(self, one_station):
        """If amplitude_error key is absent, traces are returned unchanged."""
        effect = AmplitudeErrorEffect()
        seismo = _make_seismo_map(one_station, amplitude=2.0)
        out = effect(seismo, one_station)
        assert np.allclose(out["STA1"]["Z"], 2.0)

    def test_zero_probability_multi_station_is_identity(self, two_stations):
        """amplitude_error=0.0 with multiple stations → all unchanged."""
        effect = AmplitudeErrorEffect()
        seismo = _make_seismo_map(two_stations, amplitude=1.0)
        out = effect(seismo, two_stations, amplitude_error=0.0)
        assert np.allclose(out["STA1"]["Z"], 1.0)
        assert np.allclose(out["STA2"]["Z"], 1.0)

    # ------------------------------------------------------------------
    # Deterministic scale via scale_range=(x, x)
    # ------------------------------------------------------------------

    def test_full_probability_deterministic_scale(self, one_station):
        """amplitude_error=1.0 with a degenerate scale_range → exact known output."""
        effect = AmplitudeErrorEffect(scale_range=(3.0, 3.0))
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        out = effect(seismo, one_station, amplitude_error=1.0)
        assert np.allclose(out["STA1"]["Z"], 3.0)

    def test_full_probability_all_stations_modulated(self, two_stations):
        """amplitude_error=1.0 → every station receives a scale factor."""
        effect = AmplitudeErrorEffect(scale_range=(2.0, 2.0))
        seismo = _make_seismo_map(two_stations, amplitude=1.0)
        out = effect(seismo, two_stations, amplitude_error=1.0)
        assert np.allclose(out["STA1"]["Z"], 2.0)
        assert np.allclose(out["STA2"]["Z"], 2.0)

    # ------------------------------------------------------------------
    # Per-station independence
    # ------------------------------------------------------------------

    def test_per_station_scales_are_independent(self, two_stations):
        """With a non-degenerate range, different stations must get different scales."""
        effect = AmplitudeErrorEffect(scale_range=(0.5, 2.0))
        seismo = _make_seismo_map(two_stations, amplitude=1.0)
        # With many trials at least one pair of stations will differ
        all_same = True
        for seed in range(20):
            np.random.seed(seed)
            out = effect(seismo, two_stations, amplitude_error=1.0)
            if not np.allclose(out["STA1"]["Z"], out["STA2"]["Z"]):
                all_same = False
                break
        assert not all_same, (
            "Per-station scales must be drawn independently; "
            "over 20 seeds at least one pair should differ"
        )

    def test_scale_stays_within_configured_range(self, one_station):
        """The applied scale factor must fall within the configured range."""
        low, high = 0.3, 1.7
        effect = AmplitudeErrorEffect(scale_range=(low, high))
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        for seed in range(30):
            np.random.seed(seed)
            out = effect(seismo, one_station, amplitude_error=1.0)
            scale = float(out["STA1"]["Z"][0])  # input was 1.0
            assert low <= scale <= high, (
                f"seed={seed}: scale {scale:.4f} outside [{low}, {high}]"
            )

    # ------------------------------------------------------------------
    # Non-mutation and dtype
    # ------------------------------------------------------------------

    def test_does_not_modify_input_in_place(self, one_station):
        """AmplitudeErrorEffect must return a new map, not mutate the input."""
        effect = AmplitudeErrorEffect(scale_range=(5.0, 5.0))
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        original = seismo["STA1"]["Z"].copy()
        effect(seismo, one_station, amplitude_error=1.0)
        assert np.allclose(seismo["STA1"]["Z"], original), (
            "AmplitudeErrorEffect must not modify the input seismogram map in place"
        )

    def test_output_dtype_is_float(self, one_station):
        """Output traces must be floating-point."""
        effect = AmplitudeErrorEffect(scale_range=(2.0, 2.0))
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        out = effect(seismo, one_station, amplitude_error=1.0)
        assert np.issubdtype(out["STA1"]["Z"].dtype, np.floating)


# ===========================================================================
# InstrumentDropoutEffect
# ===========================================================================

class TestInstrumentDropoutEffect:

    def test_absent_key_is_noop(self, one_station):
        """If instrument_dropout key is absent, traces are returned unchanged."""
        effect = InstrumentDropoutEffect()
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        out = effect(seismo, one_station)
        assert np.allclose(out["STA1"]["Z"], 1.0)

    def test_zero_dropout_probability_no_change(self, two_stations):
        """Dropout probability 0 means no station is zeroed."""
        effect = InstrumentDropoutEffect()
        seismo = _make_seismo_map(two_stations, amplitude=1.0)
        out = effect(seismo, two_stations, instrument_dropout=0.0)
        assert np.allclose(out["STA1"]["Z"], 1.0)
        assert np.allclose(out["STA2"]["Z"], 1.0)

    def test_full_dropout_probability_zeros_all_stations(self, two_stations):
        """Dropout probability 1 must zero every station's traces."""
        effect = InstrumentDropoutEffect()
        seismo = _make_seismo_map(two_stations, amplitude=1.0)
        out = effect(seismo, two_stations, instrument_dropout=1.0)
        assert np.allclose(out["STA1"]["Z"], 0.0)
        assert np.allclose(out["STA2"]["Z"], 0.0)

    def test_does_not_modify_input_in_place(self, two_stations):
        effect = InstrumentDropoutEffect()
        seismo = _make_seismo_map(two_stations, amplitude=1.0)
        original_sta1 = seismo["STA1"]["Z"].copy()
        effect(seismo, two_stations, instrument_dropout=1.0)
        assert np.allclose(seismo["STA1"]["Z"], original_sta1), (
            "InstrumentDropoutEffect must not modify the input seismogram map in place"
        )

    def test_output_structure_unchanged(self, two_stations):
        """Output must have the same station/component keys as input."""
        effect = InstrumentDropoutEffect()
        seismo = _make_seismo_map(two_stations)
        out = effect(seismo, two_stations, instrument_dropout=0.5)
        assert set(out.keys()) == set(seismo.keys())
        for station in seismo:
            assert set(out[station].keys()) == set(seismo[station].keys())


# ===========================================================================
# build_post_processing_chain factory
# ===========================================================================

class TestBuildPostProcessingChain:

    def test_empty_keys_returns_empty_chain(self):
        chain = build_post_processing_chain([])
        assert isinstance(chain, PostProcessingChain)
        assert len(chain.effects) == 0

    def test_known_key_amplitude_error_included(self):
        chain = build_post_processing_chain(["amplitude_error"])
        assert any(isinstance(e, AmplitudeErrorEffect) for e in chain.effects)

    def test_known_key_instrument_dropout_included(self):
        chain = build_post_processing_chain(["instrument_dropout"])
        assert any(isinstance(e, InstrumentDropoutEffect) for e in chain.effects)

    def test_unknown_keys_silently_ignored(self):
        """Non-effect nuisance keys (e.g. source_location) must not raise."""
        chain = build_post_processing_chain(["source_location", "velocity_model"])
        assert len(chain.effects) == 0

    def test_mixed_known_and_unknown_keys(self):
        chain = build_post_processing_chain(["source_location", "amplitude_error"])
        assert len(chain.effects) == 1
        assert isinstance(chain.effects[0], AmplitudeErrorEffect)

    def test_returns_post_processing_chain_instance(self):
        chain = build_post_processing_chain(["amplitude_error"])
        assert isinstance(chain, PostProcessingChain)

    def test_chain_functional_after_build(self, one_station=None):
        """End-to-end: build chain, apply it, verify output changes."""
        receivers = _make_receivers("STA1")
        seismo = _make_seismo_map(receivers, amplitude=1.0)
        chain = build_post_processing_chain(["amplitude_error"])
        # amplitude_error=1.0 → probability=1 → modulation always applied
        np.random.seed(0)
        out = chain(seismo, receivers, {"amplitude_error": 1.0})
        # Output must differ from the input (scale ≠ 1 with overwhelming probability)
        assert not np.allclose(out["STA1"]["Z"], 1.0), (
            "amplitude_error=1.0 must produce a different output (modulation always applied)"
        )
        # Scale must be within the default range [0.5, 2.0]
        scale = float(out["STA1"]["Z"][0])
        assert AmplitudeErrorEffect.DEFAULT_SCALE_LOW <= scale <= AmplitudeErrorEffect.DEFAULT_SCALE_HIGH

    def test_effect_config_scale_range_forwarded(self):
        """build_post_processing_chain should forward scale_range to AmplitudeErrorEffect."""
        chain = build_post_processing_chain(
            ["amplitude_error"],
            effect_configs={"amplitude_error": {"scale_range": (3.0, 3.0)}},
        )
        assert len(chain.effects) == 1
        effect = chain.effects[0]
        assert isinstance(effect, AmplitudeErrorEffect)
        # Degenerate range → scale is deterministically 3.0
        receivers = _make_receivers("STA1")
        seismo = _make_seismo_map(receivers, amplitude=1.0)
        out = effect(seismo, receivers, amplitude_error=1.0)
        assert np.allclose(out["STA1"]["Z"], 3.0)


# ===========================================================================
# Lanczos interpolation helpers
# ===========================================================================


class TestLanczosKernelValues:

    def test_at_zero_returns_one(self):
        vals = _lanczos_kernel_values(np.array([0.0]), order=5)
        assert np.isclose(vals[0], 1.0)

    def test_at_integer_nonzero_returns_zero(self):
        """sinc(n) = 0 for non-zero integers, so L(n) = 0 for n ≠ 0."""
        for n in [1, -1, 2, -2, 3]:
            vals = _lanczos_kernel_values(np.array([float(n)]), order=5)
            assert np.isclose(vals[0], 0.0), f"L({n}) should be 0, got {vals[0]}"

    def test_outside_support_returns_zero(self):
        vals = _lanczos_kernel_values(np.array([5.0, -5.0, 6.0]), order=5)
        assert np.allclose(vals, 0.0)

    def test_symmetric(self):
        xs = np.linspace(-4.9, 4.9, 50)
        assert np.allclose(
            _lanczos_kernel_values(xs, order=5),
            _lanczos_kernel_values(-xs, order=5),
        )


class TestApplyLanczosShift:

    def test_zero_shift_is_identity(self):
        trace = np.random.default_rng(0).standard_normal(100)
        out = _apply_lanczos_shift(trace, 0.0)
        assert np.allclose(out, trace)

    def test_zero_shift_returns_copy(self):
        trace = np.ones(50)
        out = _apply_lanczos_shift(trace, 0.0)
        out[0] = 99.0
        assert trace[0] != 99.0, "zero shift should return a copy, not a view"

    def test_integer_shift_moves_impulse_exactly(self):
        """A pure integer shift should move an impulse to the exact new position."""
        trace = np.zeros(80)
        trace[20] = 1.0
        for shift in [3, 5, -4]:
            out = _apply_lanczos_shift(trace, float(shift))
            expected_pos = 20 + shift
            if 0 <= expected_pos < 80:
                # The impulse should be at expected_pos, neighbours ≈ 0
                assert abs(out[expected_pos] - 1.0) < 1e-6, (
                    f"shift={shift}: impulse should be at {expected_pos}, got max at "
                    f"{np.argmax(np.abs(out))}"
                )
                out_copy = out.copy()
                out_copy[expected_pos] = 0.0
                assert np.max(np.abs(out_copy)) < 1e-6

    def test_positive_shift_delays_trace(self):
        """Positive tau_samples → event moves to larger indices."""
        trace = np.zeros(60)
        trace[10] = 1.0
        out = _apply_lanczos_shift(trace, 5.0)
        assert abs(out[15] - 1.0) < 1e-6

    def test_output_length_unchanged(self):
        for n in [30, 100, 256]:
            trace = np.random.default_rng(n).standard_normal(n)
            out = _apply_lanczos_shift(trace, 3.7)
            assert len(out) == n

    def test_output_dtype_float64(self):
        trace = np.ones(40, dtype=np.float32)
        out = _apply_lanczos_shift(trace, 1.5)
        assert out.dtype == np.float64

    def test_fractional_shift_preserves_energy_approx(self):
        """For a smooth band-limited signal a sub-sample shift should preserve energy."""
        t = np.linspace(0, 10, 512)
        trace = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sinusoid
        out = _apply_lanczos_shift(trace, 2.3, order=8)
        # Energy ratio within 2 % (boundary effects cause minor loss)
        ratio = np.sum(out ** 2) / np.sum(trace ** 2)
        assert 0.95 < ratio < 1.05, f"Energy ratio {ratio:.3f} outside [0.95, 1.05]"

    def test_fractional_shift_of_sinusoid_matches_analytic(self):
        """Lanczos shift of a 1 Hz sinusoid should match the analytic phase shift.

        The margin must be large enough to skip:
        - The leading zero-padded region (samples 0 … ceil(tau_samples))
        - The Lanczos kernel boundary artefacts (order samples on each side)
        """
        sr = 100.0  # samples per second
        duration = 5.0
        t = np.arange(0, duration, 1 / sr)
        freq = 1.0  # Hz — well within Lanczos bandwidth
        trace = np.sin(2 * np.pi * freq * t)

        tau_s = 0.13  # fractional-sample shift: 13 samples (not integer)
        tau_samples = tau_s * sr
        order = 8

        out = _apply_lanczos_shift(trace, tau_samples, order=order)
        expected = np.sin(2 * np.pi * freq * (t - tau_s))

        # Skip the leading zero-pad region plus kernel width on each side
        margin = int(np.ceil(abs(tau_samples))) + order + 2
        assert np.allclose(out[margin:-margin], expected[margin:-margin], atol=1e-3), (
            "Lanczos shift of a 1 Hz sinusoid should match analytic phase shift to 1e-3"
        )


# ===========================================================================
# TimeShiftErrorEffect
# ===========================================================================


class TestTimeShiftErrorEffect:
    """Tests for the two-stage stochastic per-station time shift effect.

    ``time_shift_error`` is a **probability**: each station is independently
    shifted with this probability.  When shifted, the magnitude is drawn from
    N(0, gaussian_sigma).  Lanczos interpolation is used for sub-sample accuracy.
    """

    SR = 10.0  # low sampling rate keeps traces short in tests

    def _effect(self, sigma=0.5, order=5):
        return TimeShiftErrorEffect(
            sampling_rate=self.SR,
            gaussian_sigma=sigma,
            lanczos_order=order,
        )

    # ------------------------------------------------------------------
    # Identity / no-op
    # ------------------------------------------------------------------

    def test_absent_key_is_noop(self, one_station):
        effect = self._effect()
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        out = effect(seismo, one_station)
        assert np.allclose(out["STA1"]["Z"], 1.0)

    def test_zero_probability_is_identity(self, one_station):
        """time_shift_error=0.0 → no station is ever shifted."""
        effect = self._effect()
        seismo = _make_seismo_map(one_station, amplitude=2.0)
        for seed in range(10):
            np.random.seed(seed)
            out = effect(seismo, one_station, time_shift_error=0.0)
            assert np.allclose(out["STA1"]["Z"], 2.0), f"seed={seed} produced change"

    def test_zero_probability_multi_station_is_identity(self, two_stations):
        effect = self._effect()
        seismo = _make_seismo_map(two_stations, amplitude=1.0)
        out = effect(seismo, two_stations, time_shift_error=0.0)
        assert np.allclose(out["STA1"]["Z"], 1.0)
        assert np.allclose(out["STA2"]["Z"], 1.0)

    # ------------------------------------------------------------------
    # Modulation behaviour at prob=1
    # ------------------------------------------------------------------

    def test_full_probability_changes_non_constant_trace(self, one_station):
        """With prob=1 a non-constant trace must change (the shift moves samples)."""
        effect = self._effect(sigma=1.0)
        # Use a ramp so any shift produces a different trace
        trace = np.arange(TRACE_LEN, dtype=float)
        seismo = {"STA1": {"Z": trace}}

        changed = False
        for seed in range(20):
            np.random.seed(seed)
            out = effect(seismo, one_station, time_shift_error=1.0)
            if not np.allclose(out["STA1"]["Z"], trace):
                changed = True
                break
        assert changed, "prob=1 with a non-constant trace must produce a change over 20 seeds"

    def test_zero_gaussian_sigma_is_identity(self, one_station):
        """sigma=0 → all shifts are exactly 0.0 s → trace unchanged."""
        effect = TimeShiftErrorEffect(sampling_rate=self.SR, gaussian_sigma=1e-30)
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        np.random.seed(0)
        out = effect(seismo, one_station, time_shift_error=1.0)
        assert np.allclose(out["STA1"]["Z"], 1.0)

    # ------------------------------------------------------------------
    # Per-station independence
    # ------------------------------------------------------------------

    def test_per_station_shifts_are_independent(self, two_stations):
        """Different stations must receive independent shift draws."""
        effect = self._effect(sigma=2.0)
        trace = np.arange(TRACE_LEN, dtype=float)
        seismo = {
            "STA1": {"Z": trace.copy()},
            "STA2": {"Z": trace.copy()},
        }

        found_different = False
        for seed in range(30):
            np.random.seed(seed)
            out = effect(seismo, two_stations, time_shift_error=1.0)
            if not np.allclose(out["STA1"]["Z"], out["STA2"]["Z"]):
                found_different = True
                break
        assert found_different, (
            "Per-station time shifts must be drawn independently; "
            "over 30 seeds at least one pair should differ"
        )

    # ------------------------------------------------------------------
    # Non-mutation and dtype
    # ------------------------------------------------------------------

    def test_does_not_modify_input_in_place(self, one_station):
        effect = self._effect(sigma=1.0)
        trace = np.arange(TRACE_LEN, dtype=float)
        seismo = {"STA1": {"Z": trace.copy()}}
        original = trace.copy()
        np.random.seed(42)
        effect(seismo, one_station, time_shift_error=1.0)
        assert np.allclose(seismo["STA1"]["Z"], original), (
            "TimeShiftErrorEffect must not modify the input seismogram map in place"
        )

    def test_output_dtype_is_float64(self, one_station):
        effect = self._effect()
        seismo = {"STA1": {"Z": np.ones(TRACE_LEN, dtype=np.float32)}}
        np.random.seed(0)
        out = effect(seismo, one_station, time_shift_error=1.0)
        assert out["STA1"]["Z"].dtype == np.float64

    def test_output_length_unchanged(self, one_station):
        effect = self._effect(sigma=1.0)
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        np.random.seed(0)
        out = effect(seismo, one_station, time_shift_error=1.0)
        assert len(out["STA1"]["Z"]) == TRACE_LEN

    # ------------------------------------------------------------------
    # Registry integration
    # ------------------------------------------------------------------

    def test_time_shift_error_in_registry(self):
        from seismo_sbi.instaseis_simulator.post_processing import EFFECT_REGISTRY
        assert "time_shift_error" in EFFECT_REGISTRY

    def test_build_chain_with_sampling_rate_config(self):
        """build_post_processing_chain forwards sampling_rate via effect_configs."""
        chain = build_post_processing_chain(
            ["time_shift_error"],
            effect_configs={"time_shift_error": {"sampling_rate": self.SR, "gaussian_sigma": 0.5}},
        )
        assert len(chain.effects) == 1
        assert isinstance(chain.effects[0], TimeShiftErrorEffect)


# ===========================================================================
# _apply_random_coda_filter helper
# ===========================================================================


class TestApplyRandomCodaFilter:

    def test_alpha_zero_is_identity(self):
        """alpha=0 → kernel is a unit spike → filter is identity."""
        trace = np.random.default_rng(7).standard_normal(128)
        out = _apply_random_coda_filter(trace, alpha=0.0)
        assert np.allclose(out, trace, atol=1e-10)

    def test_output_real_and_same_length(self):
        for n in [64, 100, 128, 200]:
            trace = np.random.default_rng(n).standard_normal(n)
            out = _apply_random_coda_filter(trace, alpha=0.4)
            assert len(out) == n
            assert np.issubdtype(out.dtype, np.floating)

    def test_causal_no_wrap_around(self):
        """Energy near the right edge must NOT bleed into the start of the trace.

        Regression for the circular-convolution wrap-around of the old spectral
        all-pass implementation.
        """
        n = 512
        t = np.arange(n)
        # A localised wavelet right against the right-hand edge.
        x = (t - 480) / 8.0
        trace = (1 - x ** 2) * np.exp(-(x ** 2) / 2)
        np.random.seed(1)
        out = _apply_random_coda_filter(trace, alpha=0.6)
        # Nothing should appear in the first half — coda only trails (and is
        # truncated at the window edge), it cannot wrap to the start.
        assert np.allclose(out[:256], 0.0, atol=1e-10), (
            "Coda must not wrap around to the start of the window"
        )

    def test_no_bulk_time_shift(self):
        """The direct-arrival onset must stay put (no alpha-dependent bulk delay).

        Regression for the positive average group delay of any causal all-pass.
        """
        n = 512
        t = np.arange(n)
        x = (t - 256) / 8.0
        trace = (1 - x ** 2) * np.exp(-(x ** 2) / 2)
        lags = []
        for seed in range(50):
            np.random.seed(seed)
            out = _apply_random_coda_filter(trace, alpha=0.4)
            lags.append(np.argmax(np.correlate(out, trace, "full")) - (n - 1))
        # Cross-correlation peak should remain at ~0 lag (was ~25 samples for
        # the spectral all-pass).
        assert abs(np.mean(lags)) < 2.0, (
            f"scattering_coda introduced a bulk time shift of {np.mean(lags):.1f} samples"
        )

    def test_energy_approximately_preserved(self):
        """L2-normalised kernel conserves energy on average for a white trace."""
        trace = np.random.default_rng(99).standard_normal(2000)
        np.random.seed(1)
        out = _apply_random_coda_filter(trace, alpha=0.9)
        assert np.isclose(np.sum(out ** 2), np.sum(trace ** 2), rtol=0.1)

    def test_larger_alpha_spreads_more_energy_into_coda(self):
        """Stronger perturbation should redistribute more energy into the tail."""
        # Start with an impulse-like trace (energy concentrated near start)
        trace = np.zeros(256)
        trace[10] = 1.0

        np.random.seed(5)
        out_weak = _apply_random_coda_filter(trace.copy(), alpha=0.05)
        np.random.seed(5)
        out_strong = _apply_random_coda_filter(trace.copy(), alpha=0.9)

        # Coda window: samples past the direct arrival
        coda_energy_weak = np.sum(out_weak[30:] ** 2)
        coda_energy_strong = np.sum(out_strong[30:] ** 2)
        assert coda_energy_strong > coda_energy_weak, (
            "Stronger alpha should spread more energy into the coda tail"
        )

    def test_output_dtype_float64(self):
        trace = np.ones(64, dtype=np.float32)
        out = _apply_random_coda_filter(trace, alpha=0.4)
        assert out.dtype == np.float64


# ===========================================================================
# _apply_stahler_phase_filter helper (paper-exact, Eq. 17)
# ===========================================================================


class TestApplyStahlerPhaseFilter:

    def test_alpha_zero_is_identity(self):
        """alpha=0 → φ≡0 → transfer function is a unit spike → identity."""
        trace = np.random.default_rng(7).standard_normal(128)
        out = _apply_stahler_phase_filter(trace, alpha=0.0)
        assert np.allclose(out, trace, atol=1e-10)

    def test_output_real_and_same_length(self):
        for n in [64, 100, 128, 200]:
            trace = np.random.default_rng(n).standard_normal(n)
            out = _apply_stahler_phase_filter(trace, alpha=0.4)
            assert len(out) == n
            assert np.issubdtype(out.dtype, np.floating)

    def test_transfer_function_is_unit_amplitude(self):
        """The modelling-error filter has a unit amplitude spectrum (all-pass).

        Reconstruct the filter the helper builds and check |rfft(h)| ≡ 1.
        """
        n = 512
        coda_len = max(2, int(round(0.25 * n)))
        n_bins = coda_len // 2 + 1
        np.random.seed(0)
        phi = np.random.uniform(0.0, 0.6 * np.pi / 2.0, size=n_bins)
        phi[0] = 0.0
        if coda_len % 2 == 0:
            phi[-1] = 0.0
        h = np.fft.irfft(np.exp(1j * phi), n=coda_len)
        assert np.allclose(np.abs(np.fft.rfft(h)), 1.0, atol=1e-9), (
            "Stähler transfer function must have a unit amplitude spectrum"
        )

    def test_causal_quiet_regions_stay_quiet(self):
        """A localised arrival must not leak energy into the leading zero region.

        This is the paper's signature behaviour (and the fix for the circular
        wrap-around / acausal smearing of multiplying the trace's own spectrum).
        """
        n = 512
        t = np.arange(n)
        x = (t - 256) / 6.0
        trace = (1 - x ** 2) * np.exp(-(x ** 2) / 2)  # wavelet at sample 256
        for seed in range(20):
            np.random.seed(seed)
            out = _apply_stahler_phase_filter(trace, alpha=0.9)
            assert np.allclose(out[:150], 0.0, atol=1e-12), (
                f"seed={seed}: energy leaked into the pre-arrival zero region"
            )

    def test_energy_preserved_broadband(self):
        """Unit-amplitude filter conserves energy for a broadband signal."""
        rng = np.random.default_rng(3)
        trace = np.zeros(512)
        trace[200:312] = rng.standard_normal(112) * np.hanning(112)
        np.random.seed(1)
        out = _apply_stahler_phase_filter(trace, alpha=0.6)
        assert np.isclose(np.sum(out ** 2), np.sum(trace ** 2), rtol=0.2)


# ===========================================================================
# ScatteringCodaEffect
# ===========================================================================


class TestScatteringCodaEffect:

    def test_absent_key_is_noop(self, one_station):
        effect = ScatteringCodaEffect()
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        out = effect(seismo, one_station)
        assert np.allclose(out["STA1"]["Z"], 1.0)

    def test_zero_probability_is_identity(self, one_station):
        """scattering_coda=0.0 → no station is ever perturbed."""
        effect = ScatteringCodaEffect(alpha=0.9)
        seismo = _make_seismo_map(one_station, amplitude=2.0)
        for seed in range(10):
            np.random.seed(seed)
            out = effect(seismo, one_station, scattering_coda=0.0)
            assert np.allclose(out["STA1"]["Z"], 2.0), f"seed={seed} produced change at prob=0"

    def test_prob1_changes_non_constant_trace(self, one_station):
        """scattering_coda=1.0 with alpha>0 must change a non-constant trace."""
        effect = ScatteringCodaEffect(alpha=0.4)
        trace = np.sin(np.linspace(0, 2 * np.pi, TRACE_LEN))
        seismo = {"STA1": {"Z": trace}}
        changed = False
        for seed in range(20):
            np.random.seed(seed)
            out = effect(seismo, one_station, scattering_coda=1.0)
            if not np.allclose(out["STA1"]["Z"], trace):
                changed = True
                break
        assert changed, "scattering_coda=1.0 with alpha=0.4 must change a sinusoidal trace"

    def test_causal_no_wrap_around_prob1(self, one_station):
        """Coda from a late arrival must not wrap to the start of the trace."""
        effect = ScatteringCodaEffect(alpha=0.9)
        trace = np.zeros(TRACE_LEN)
        trace[TRACE_LEN - 2] = 1.0  # impulse hard against the right edge
        seismo = {"STA1": {"Z": trace}}
        np.random.seed(7)
        out = effect(seismo, one_station, scattering_coda=1.0)
        assert np.allclose(out["STA1"]["Z"][: TRACE_LEN // 2], 0.0, atol=1e-10), (
            "ScatteringCodaEffect must not wrap coda around to the start of the trace"
        )

    def test_per_station_independence(self, two_stations):
        """Different stations must receive independent phase draws."""
        effect = ScatteringCodaEffect(alpha=0.4)
        trace = np.sin(np.linspace(0, 4 * np.pi, TRACE_LEN))
        seismo = {
            "STA1": {"Z": trace.copy()},
            "STA2": {"Z": trace.copy()},
        }
        found_different = False
        for seed in range(30):
            np.random.seed(seed)
            out = effect(seismo, two_stations, scattering_coda=1.0)
            if not np.allclose(out["STA1"]["Z"], out["STA2"]["Z"]):
                found_different = True
                break
        assert found_different, (
            "Per-station phase draws must be independent; "
            "STA1 and STA2 should differ over 30 seeds"
        )

    def test_does_not_mutate_input(self, one_station):
        effect = ScatteringCodaEffect(alpha=0.9)
        trace = np.sin(np.linspace(0, np.pi, TRACE_LEN))
        seismo = {"STA1": {"Z": trace.copy()}}
        original = trace.copy()
        np.random.seed(0)
        effect(seismo, one_station, scattering_coda=1.0)
        assert np.allclose(seismo["STA1"]["Z"], original), (
            "ScatteringCodaEffect must not mutate the input seismogram map"
        )

    def test_output_dtype_float64(self, one_station):
        effect = ScatteringCodaEffect()
        seismo = {"STA1": {"Z": np.ones(TRACE_LEN, dtype=np.float32)}}
        np.random.seed(0)
        out = effect(seismo, one_station, scattering_coda=1.0)
        assert out["STA1"]["Z"].dtype == np.float64

    def test_output_length_unchanged(self, one_station):
        effect = ScatteringCodaEffect(alpha=0.4)
        seismo = _make_seismo_map(one_station, amplitude=1.0)
        np.random.seed(0)
        out = effect(seismo, one_station, scattering_coda=1.0)
        assert len(out["STA1"]["Z"]) == TRACE_LEN

    def test_registry_membership(self):
        from seismo_sbi.instaseis_simulator.post_processing import EFFECT_REGISTRY
        assert "scattering_coda" in EFFECT_REGISTRY
        assert EFFECT_REGISTRY["scattering_coda"] is ScatteringCodaEffect

    def test_build_via_effect_configs(self):
        """build_post_processing_chain forwards alpha via effect_configs."""
        chain = build_post_processing_chain(
            ["scattering_coda"],
            effect_configs={"scattering_coda": {"alpha": 0.9}},
        )
        assert len(chain.effects) == 1
        effect = chain.effects[0]
        assert isinstance(effect, ScatteringCodaEffect)
        assert effect._alpha == 0.9

    def test_default_alpha(self):
        """Constructing without alpha uses the class default."""
        effect = ScatteringCodaEffect()
        assert effect._alpha == ScatteringCodaEffect.DEFAULT_ALPHA

    def test_default_mode_is_causal(self):
        assert ScatteringCodaEffect()._mode == "causal"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ScatteringCodaEffect(mode="nonsense")

    def test_stahler_mode_is_causal(self, one_station):
        """mode='stahler' convolves with a causal filter → no leakage before an arrival."""
        effect = ScatteringCodaEffect(alpha=0.9, mode="stahler")
        trace = np.zeros(TRACE_LEN)
        trace[TRACE_LEN - 2] = 1.0  # impulse hard against the right edge
        seismo = {"STA1": {"Z": trace}}
        np.random.seed(7)
        out = effect(seismo, one_station, scattering_coda=1.0)
        assert np.allclose(out["STA1"]["Z"][: TRACE_LEN // 2], 0.0, atol=1e-12), (
            "mode='stahler' must not leak coda to the start of the trace"
        )

    def test_stahler_mode_changes_trace(self, one_station):
        """mode='stahler' with alpha>0 must perturb a non-constant trace."""
        effect = ScatteringCodaEffect(alpha=0.6, mode="stahler")
        trace = np.sin(np.linspace(0, 4 * np.pi, TRACE_LEN))
        seismo = {"STA1": {"Z": trace}}
        changed = False
        for seed in range(20):
            np.random.seed(seed)
            out = effect(seismo, one_station, scattering_coda=1.0)
            if not np.allclose(out["STA1"]["Z"], trace):
                changed = True
                break
        assert changed

    def test_mode_forwarded_via_effect_configs(self):
        chain = build_post_processing_chain(
            ["scattering_coda"],
            effect_configs={"scattering_coda": {"alpha": 0.9, "mode": "stahler"}},
        )
        effect = chain.effects[0]
        assert effect._mode == "stahler"
        assert effect._alpha == 0.9
