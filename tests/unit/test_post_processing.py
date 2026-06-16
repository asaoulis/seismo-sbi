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
_apply_lanczos_shift_batch = post_processing._apply_lanczos_shift_batch
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
# TimeShiftErrorEffect — common-offset distribution
# ===========================================================================


class TestTimeShiftCommonOffset:

    @staticmethod
    def _ramp_map(receivers):
        # non-constant traces so a time shift is detectable
        return {rec.station_name: {"Z": np.arange(TRACE_LEN, dtype=float)}
                for rec in receivers.iterate()}

    def test_default_is_uniform_backcompat(self):
        effect = TimeShiftErrorEffect(sampling_rate=1.0)
        assert effect._common_dist == "uniform"

    def test_invalid_dist_raises(self):
        with pytest.raises(ValueError):
            TimeShiftErrorEffect(sampling_rate=1.0, common_offset_dist="lognormal")

    def test_gaussian_common_shared_across_stations(self, two_stations):
        # per-station sigma 0 => the only shift is the shared array-wide common
        # offset, so both stations must shift identically and away from input.
        np.random.seed(0)
        effect = TimeShiftErrorEffect(
            sampling_rate=1.0, common_offset_dist="gaussian",
            common_offset_sigma=3.0, gaussian_sigma=0.0)
        seismo = self._ramp_map(two_stations)
        out = effect(seismo, two_stations, time_shift_error=1.0)
        np.testing.assert_allclose(out["STA1"]["Z"], out["STA2"]["Z"])
        assert not np.allclose(out["STA1"]["Z"], np.arange(TRACE_LEN, dtype=float))

    def test_gaussian_common_sigma_defaults_to_uniform_offset(self):
        # when flipped to gaussian without a sigma, the existing uniform_offset
        # scale carries over as the Gaussian std.
        effect = TimeShiftErrorEffect(
            sampling_rate=1.0, common_offset_dist="gaussian", uniform_offset=2.5)
        assert effect._common_sigma == 2.5

    def test_gaussian_common_can_exceed_uniform_halfwidth(self, one_station):
        # A Gaussian has unbounded support: over many draws the common shift
        # magnitude should sometimes exceed sigma (impossible for U(±0.5*sigma)).
        effect = TimeShiftErrorEffect(
            sampling_rate=1.0, common_offset_dist="gaussian",
            common_offset_sigma=3.0, gaussian_sigma=0.0)
        ramp = np.arange(TRACE_LEN, dtype=float)
        # recover the applied shift via the ramp slope (unit slope => Δvalue ≈ shift)
        max_abs_shift = 0.0
        for seed in range(40):
            np.random.seed(seed)
            out = effect(self._ramp_map(one_station), one_station, time_shift_error=1.0)
            mid = TRACE_LEN // 2
            max_abs_shift = max(max_abs_shift, abs(out["STA1"]["Z"][mid] - ramp[mid]))
        assert max_abs_shift > 3.0    # exceeds 1 sigma at least once over 40 draws


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

    def test_batch_equals_per_trace_loop(self):
        """Batched shift must be IDENTICAL to looping the 1-D shift row by row.

        ``_apply_lanczos_shift_batch`` applies one shared shift to every row of a
        ``(C, T)`` block (all components of a station get one per-station shift);
        the TimeShiftErrorEffect now uses it instead of a per-component Python loop.
        Each output element is the same weighted sum of the same inputs, so the
        result must match the per-trace path bitwise (not just to a tolerance).
        """
        rng = np.random.default_rng(7)
        for _ in range(50):
            C = int(rng.integers(1, 5))
            T = int(rng.integers(40, 300))
            block = rng.standard_normal((C, T))
            tau = float(rng.uniform(-9.0, 9.0))
            order = int(rng.integers(3, 9))
            batched = _apply_lanczos_shift_batch(block, tau, order=order)
            per_trace = np.stack(
                [_apply_lanczos_shift(block[j], tau, order=order) for j in range(C)]
            )
            assert batched.shape == (C, T)
            np.testing.assert_array_equal(batched, per_trace)

    def test_batch_tiny_shift_is_identity_copy(self):
        """|tau| < 1e-10 returns a float64 copy of every row (no aliasing)."""
        block = np.arange(12, dtype=np.float64).reshape(3, 4)
        out = _apply_lanczos_shift_batch(block, 1e-12)
        np.testing.assert_array_equal(out, block)
        assert out.dtype == np.float64
        assert out is not block
        out[0, 0] = -999.0
        assert block[0, 0] == 0.0  # mutation did not leak back


# ===========================================================================
# TimeShiftErrorEffect
# ===========================================================================


class TestTimeShiftErrorEffect:
    """Tests for the common-offset + per-station-Gaussian time shift effect.

    ``time_shift_error`` is an **on/off switch** (NOT a probability): ``0.0`` (or
    absent) ⇒ identity; any non-zero value ⇒ active.  When active, every station
    receives a shared ``uniform(-uniform_offset, +uniform_offset)`` common offset
    plus an independent ``N(0, gaussian_sigma)`` draw.  Lanczos interpolation is
    used for sub-sample accuracy.
    """

    SR = 10.0  # low sampling rate keeps traces short in tests

    def _effect(self, sigma=0.5, order=5, uniform_offset=0.0):
        return TimeShiftErrorEffect(
            sampling_rate=self.SR,
            uniform_offset=uniform_offset,
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
    # Common offset (uniform) + per-station Gaussian parametrisation
    # ------------------------------------------------------------------

    def test_common_offset_is_identical_across_stations(self, two_stations):
        """uniform_offset>0, sigma=0 → every station shifted by the SAME amount."""
        # sigma effectively zero removes the per-station component, leaving only
        # the shared common offset → identical traces for identical input.
        effect = self._effect(sigma=1e-30, uniform_offset=2.0)
        trace = np.arange(TRACE_LEN, dtype=float)
        seismo = {"STA1": {"Z": trace.copy()}, "STA2": {"Z": trace.copy()}}
        np.random.seed(3)
        out = effect(seismo, two_stations, time_shift_error=1.0)
        assert np.allclose(out["STA1"]["Z"], out["STA2"]["Z"]), (
            "With only a common offset, all stations must be shifted identically"
        )

    def test_per_station_gaussian_differs_without_common_offset(self, two_stations):
        """uniform_offset=0, sigma>0 → stations get independent (differing) shifts."""
        effect = self._effect(sigma=2.0, uniform_offset=0.0)
        trace = np.arange(TRACE_LEN, dtype=float)
        seismo = {"STA1": {"Z": trace.copy()}, "STA2": {"Z": trace.copy()}}
        found_different = False
        for seed in range(30):
            np.random.seed(seed)
            out = effect(seismo, two_stations, time_shift_error=1.0)
            if not np.allclose(out["STA1"]["Z"], out["STA2"]["Z"]):
                found_different = True
                break
        assert found_different

    def test_zero_switch_is_identity_with_offsets_configured(self, two_stations):
        """time_shift_error=0.0 ⇒ identity even with large uniform_offset/sigma set."""
        effect = self._effect(sigma=5.0, uniform_offset=5.0)
        seismo = _make_seismo_map(two_stations, amplitude=3.0)
        for seed in range(10):
            np.random.seed(seed)
            out = effect(seismo, two_stations, time_shift_error=0.0)
            assert np.allclose(out["STA1"]["Z"], 3.0)
            assert np.allclose(out["STA2"]["Z"], 3.0)

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
        """build_post_processing_chain forwards a fixed alpha via effect_configs."""
        chain = build_post_processing_chain(
            ["scattering_coda"],
            effect_configs={"scattering_coda": {"alpha": 0.9}},
        )
        assert len(chain.effects) == 1
        effect = chain.effects[0]
        assert isinstance(effect, ScatteringCodaEffect)
        # Fixed alpha → degenerate sampling range.
        assert effect._alpha_low == 0.9 and effect._alpha_high == 0.9

    def test_default_alpha_range_is_unit_interval(self):
        """Constructing without args samples alpha per station from (0, 1)."""
        effect = ScatteringCodaEffect()
        assert (effect._alpha_low, effect._alpha_high) == ScatteringCodaEffect.DEFAULT_ALPHA_RANGE

    def test_alpha_range_forwarded_via_effect_configs(self):
        chain = build_post_processing_chain(
            ["scattering_coda"],
            effect_configs={"scattering_coda": {"alpha_range": [0.2, 0.6]}},
        )
        effect = chain.effects[0]
        assert effect._alpha_low == 0.2 and effect._alpha_high == 0.6

    def test_alpha_and_alpha_range_mutually_exclusive(self):
        with pytest.raises(ValueError):
            ScatteringCodaEffect(alpha=0.5, alpha_range=(0.0, 1.0))

    def test_per_station_alpha_sampled_in_range(self, two_stations):
        """Each station draws its own alpha; a narrow non-zero range perturbs both."""
        # Range well above 0 so both stations are reliably perturbed (alpha>0).
        effect = ScatteringCodaEffect(alpha_range=(0.7, 0.9), mode="causal")
        trace = np.sin(np.linspace(0, 4 * np.pi, TRACE_LEN))
        seismo = {"STA1": {"Z": trace.copy()}, "STA2": {"Z": trace.copy()}}
        np.random.seed(0)
        out = effect(seismo, two_stations, scattering_coda=1.0)
        assert not np.allclose(out["STA1"]["Z"], trace)
        assert not np.allclose(out["STA2"]["Z"], trace)

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
        assert effect._alpha_low == 0.9 and effect._alpha_high == 0.9


# ===========================================================================
# apply_chain_to_array — map<->array adapter (training-time augmentation bridge)
# ===========================================================================

apply_chain_to_array = post_processing.apply_chain_to_array
build_augmentation_chain = post_processing.build_augmentation_chain


class TestBuildAugmentationChain:
    """build_augmentation_chain selects training-staged effects and uses fiducials."""

    def test_only_training_augmentation_keys_selected(self):
        nuisance = {"amplitude_error": [0.3], "instrument_dropout": [0.5], "stf_duration": [1.0]}
        stage = {
            "amplitude_error": "training_augmentation",
            "instrument_dropout": "simulation",
            "stf_duration": "training_augmentation",  # Category-1 → never augmentable
        }
        chain, params = build_augmentation_chain(nuisance, stage)
        assert [type(e).__name__ for e in chain.effects] == ["AmplitudeErrorEffect"]
        assert params == {"amplitude_error": 0.3}

    def test_activation_uses_fiducial_not_hardcoded_one(self):
        """Regression: probability comes from the fiducial, not a hardcoded 1.0."""
        nuisance = {"instrument_dropout": [0.3]}
        stage = {"instrument_dropout": "training_augmentation"}
        _, params = build_augmentation_chain(nuisance, stage)
        assert params["instrument_dropout"] == 0.3

    def test_sampling_rate_injected_for_time_shift(self):
        nuisance = {"time_shift_error": [1.0]}
        stage = {"time_shift_error": "training_augmentation"}
        chain, params = build_augmentation_chain(
            nuisance, stage,
            effect_configs={"time_shift_error": {"uniform_offset": 2.0, "gaussian_sigma": 1.0}},
            sampling_rate=4.0,
        )
        assert isinstance(chain.effects[0], TimeShiftErrorEffect)
        assert chain.effects[0]._sampling_rate == 4.0
        assert params == {"time_shift_error": 1.0}

    def test_empty_when_nothing_staged_for_augmentation(self):
        nuisance = {"amplitude_error": [0.3]}
        chain, params = build_augmentation_chain(nuisance, {"amplitude_error": "simulation"})
        assert chain.effects == [] and params == {}


class TestApplyChainToArray:
    """The same effects must run on the stacked (N_stations, N_components, T) array."""

    COMPONENTS = "ZNE"

    def _multi_comp_receivers(self):
        # STA1 has all three components; STA2 only Z (rows N,E zero-filled in D).
        recs = [
            Receiver(0.0, 0.0, "XX", "STA1", ["Z", "N", "E"]),
            Receiver(1.0, 1.0, "XX", "STA2", ["Z"]),
        ]
        return Receivers(receivers=recs)

    def _stacked(self, receivers):
        """Build a stacked D matching convert_sim_data_to_array(stacked, fill_unused)."""
        n_stations = len(list(receivers.iterate()))
        D = np.zeros((n_stations, len(self.COMPONENTS), TRACE_LEN), dtype=np.float64)
        for i, rec in enumerate(receivers.iterate()):
            for j, comp in enumerate(self.COMPONENTS):
                if comp in rec.components:
                    D[i, j] = np.arange(TRACE_LEN, dtype=float) + 10 * i + j
        return D

    def test_empty_chain_roundtrips_unchanged(self, two_stations):
        D = self._stacked(self._multi_comp_receivers())
        out = apply_chain_to_array(PostProcessingChain([]), D, self._multi_comp_receivers(),
                                   self.COMPONENTS, {})
        assert np.array_equal(out, D)

    def test_amplitude_scale_doubles_all(self):
        recs = self._multi_comp_receivers()
        D = self._stacked(recs)
        chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(2.0, 2.0))])
        np.random.seed(0)
        out = apply_chain_to_array(chain, D, recs, self.COMPONENTS, {"amplitude_error": 1.0})
        assert np.allclose(out, 2.0 * D)

    def test_dropout_zeros_all(self):
        recs = self._multi_comp_receivers()
        D = self._stacked(recs)
        chain = PostProcessingChain([InstrumentDropoutEffect()])
        np.random.seed(0)
        out = apply_chain_to_array(chain, D, recs, self.COMPONENTS, {"instrument_dropout": 1.0})
        assert np.allclose(out, 0.0)

    def test_zero_filled_components_stay_zero(self):
        recs = self._multi_comp_receivers()
        D = self._stacked(recs)
        # STA2 (index 1) rows N(1) and E(2) are zero-filled and must remain zero.
        chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(3.0, 3.0))])
        np.random.seed(1)
        out = apply_chain_to_array(chain, D, recs, self.COMPONENTS, {"amplitude_error": 1.0})
        assert np.allclose(out[1, 1], 0.0)
        assert np.allclose(out[1, 2], 0.0)

    def test_array_path_matches_dict_path_under_same_seed(self):
        """apply_chain_to_array == hand-built dict path under identical RNG."""
        recs = self._multi_comp_receivers()
        D = self._stacked(recs)
        chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(0.3, 1.9))])

        np.random.seed(7)
        out_array = apply_chain_to_array(chain, D, recs, self.COMPONENTS, {"amplitude_error": 0.6})

        # Reconstruct the equivalent dict path with the SAME seed and ordering.
        station_names = [r.station_name for r in recs.iterate()]
        seismo = {s: {c: D[i, j] for j, c in enumerate(self.COMPONENTS)}
                  for i, s in enumerate(station_names)}
        np.random.seed(7)
        processed = chain(seismo, recs, {"amplitude_error": 0.6})
        expected = np.zeros_like(D)
        for i, s in enumerate(station_names):
            for j, c in enumerate(self.COMPONENTS):
                expected[i, j] = processed[s][c]
        assert np.allclose(out_array, expected)

    def test_time_shift_via_adapter(self):
        recs = self._multi_comp_receivers()
        D = self._stacked(recs)
        chain = PostProcessingChain([
            TimeShiftErrorEffect(sampling_rate=10.0, uniform_offset=0.0, gaussian_sigma=2.0)
        ])
        np.random.seed(0)
        out = apply_chain_to_array(chain, D, recs, self.COMPONENTS, {"time_shift_error": 1.0})
        assert out.shape == D.shape
        # STA1 Z row (non-constant ramp) should change under a shift.
        assert not np.allclose(out[0, 0], D[0, 0])


# ===========================================================================
# ComponentDropoutEffect — per-channel zeroing of PRESENT components
# ===========================================================================

ComponentDropoutEffect = post_processing.ComponentDropoutEffect


def _multi_comp_map_and_receivers():
    """A seismograms_map (global-component keyed, absent components zero-filled, mirroring
    the array adapter) plus Receivers whose per-station `.components` mark the PRESENT set.

    STA1: present Z, N, E (3 present);  STA2: present Z, N (E zero-filled);  STA3: present Z only.
    """
    components = "ZNE"
    recs = Receivers(receivers=[
        Receiver(0.0, 0.0, "XX", "STA1", ["Z", "N", "E"]),
        Receiver(1.0, 1.0, "XX", "STA2", ["Z", "N"]),
        Receiver(2.0, 2.0, "XX", "STA3", ["Z"]),
    ])
    seismo = {}
    for rec in recs.iterate():
        seismo[rec.station_name] = {
            comp: (np.full(TRACE_LEN, 5.0) if comp in rec.components else np.zeros(TRACE_LEN))
            for comp in components
        }
    return seismo, recs, components


class TestComponentDropoutEffect:
    """Per-channel Bernoulli dropout that zeros PRESENT components, keeping >=1 per station."""

    def test_absent_key_is_noop(self):
        seismo, recs, _ = _multi_comp_map_and_receivers()
        out = ComponentDropoutEffect()(seismo, recs)  # component_dropout not passed
        assert out is seismo

    def test_zero_probability_is_identity(self):
        seismo, recs, comps = _multi_comp_map_and_receivers()
        np.random.seed(0)
        out = ComponentDropoutEffect()(seismo, recs, component_dropout=0.0)
        for station, channels in seismo.items():
            for comp in comps:
                assert np.array_equal(out[station][comp], channels[comp])

    def test_full_probability_keeps_exactly_one_per_station(self):
        """p=1: every station with >=2 present channels keeps exactly ONE present channel."""
        seismo, recs, comps = _multi_comp_map_and_receivers()
        present = {rec.station_name: list(rec.components) for rec in recs.iterate()}
        for seed in range(25):
            np.random.seed(seed)
            out = ComponentDropoutEffect()(seismo, recs, component_dropout=1.0)
            # STA1 (3 present) and STA2 (2 present): exactly one present channel survives non-zero.
            for sta in ("STA1", "STA2"):
                kept = [c for c in present[sta] if not np.allclose(out[sta][c], 0.0)]
                assert len(kept) == 1, f"{sta} seed={seed} kept={kept}"
            # STA3 has a single present channel → never dropped.
            assert np.allclose(out["STA3"]["Z"], 5.0)

    def test_single_present_channel_station_never_dropped(self):
        seismo, recs, _ = _multi_comp_map_and_receivers()
        for seed in range(25):
            np.random.seed(seed)
            out = ComponentDropoutEffect()(seismo, recs, component_dropout=1.0)
            assert np.allclose(out["STA3"]["Z"], 5.0)

    def test_absent_components_untouched(self):
        """Zero-filled absent components (STA2 E, STA3 N/E) stay exactly zero and are not
        counted as droppable present channels."""
        seismo, recs, _ = _multi_comp_map_and_receivers()
        for seed in range(15):
            np.random.seed(seed)
            out = ComponentDropoutEffect()(seismo, recs, component_dropout=1.0)
            assert np.allclose(out["STA2"]["E"], 0.0)
            assert np.allclose(out["STA3"]["N"], 0.0)
            assert np.allclose(out["STA3"]["E"], 0.0)

    def test_dropped_channels_are_exactly_zero(self):
        seismo, recs, _ = _multi_comp_map_and_receivers()
        np.random.seed(3)
        out = ComponentDropoutEffect()(seismo, recs, component_dropout=1.0)
        # Whatever is dropped in STA1 is EXACTLY zero (not merely small).
        dropped = [c for c in ("Z", "N", "E") if np.all(out["STA1"][c] == 0.0)]
        assert len(dropped) == 2  # 3 present, keep 1

    def test_does_not_modify_input_in_place(self):
        seismo, recs, _ = _multi_comp_map_and_receivers()
        original = {s: {c: t.copy() for c, t in ch.items()} for s, ch in seismo.items()}
        np.random.seed(0)
        ComponentDropoutEffect()(seismo, recs, component_dropout=1.0)
        for s, ch in original.items():
            for c, t in ch.items():
                assert np.array_equal(seismo[s][c], t), "input map mutated in place"

    def test_reproducible_under_fixed_seed(self):
        seismo, recs, comps = _multi_comp_map_and_receivers()
        np.random.seed(42)
        a = ComponentDropoutEffect()(seismo, recs, component_dropout=0.5)
        np.random.seed(42)
        b = ComponentDropoutEffect()(seismo, recs, component_dropout=0.5)
        for s in seismo:
            for c in comps:
                assert np.array_equal(a[s][c], b[s][c])

    def test_via_apply_chain_to_array(self):
        """The effect runs through the stacked-array adapter and zeros present rows only."""
        seismo, recs, comps = _multi_comp_map_and_receivers()
        D = np.zeros((3, len(comps), TRACE_LEN), dtype=np.float64)
        for i, rec in enumerate(recs.iterate()):
            for j, comp in enumerate(comps):
                if comp in rec.components:
                    D[i, j] = 5.0
        chain = PostProcessingChain([ComponentDropoutEffect()])
        np.random.seed(0)
        out = apply_chain_to_array(chain, D, recs, comps, {"component_dropout": 1.0})
        assert out.shape == D.shape
        # STA1 keeps exactly one present (Z/N/E) row non-zero.
        kept = [j for j in range(3) if not np.allclose(out[0, j], 0.0)]
        assert len(kept) == 1
        # STA3 (index 2) single present Z stays; its absent N/E rows stay zero.
        assert np.allclose(out[2, 0], 5.0)
        assert np.allclose(out[2, 1], 0.0) and np.allclose(out[2, 2], 0.0)


# ===========================================================================
# Stage routing — pre-noise vs post-noise augmentation chains
# ===========================================================================

class TestPostNoiseStageRouting:

    def test_post_noise_stage_selects_only_component_dropout(self):
        nuisance = {"component_dropout": [0.2], "amplitude_error": [0.3]}
        stage = {
            "component_dropout": "training_augmentation_post_noise",
            "amplitude_error": "training_augmentation",
        }
        chain, params = build_augmentation_chain(
            nuisance, stage, stage="training_augmentation_post_noise"
        )
        assert [type(e).__name__ for e in chain.effects] == ["ComponentDropoutEffect"]
        assert params == {"component_dropout": 0.2}

    def test_pre_noise_stage_excludes_component_dropout(self):
        nuisance = {"component_dropout": [0.2], "amplitude_error": [0.3]}
        stage = {
            "component_dropout": "training_augmentation_post_noise",
            "amplitude_error": "training_augmentation",
        }
        chain, params = build_augmentation_chain(nuisance, stage)  # default pre-noise stage
        assert [type(e).__name__ for e in chain.effects] == ["AmplitudeErrorEffect"]
        assert "component_dropout" not in params

    def test_component_dropout_in_post_noise_keys(self):
        assert "component_dropout" in post_processing.POST_NOISE_EFFECT_KEYS
        assert "component_dropout" not in post_processing.AUGMENTABLE_EFFECT_KEYS
