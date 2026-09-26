"""Effects applied to synthetic seismograms after the forward model returns.

A ``SeismogramEffect`` owns exactly one nuisance key, is a strict no-op when that key is absent
from ``nuisance_params``, and returns a new dict rather than mutating its input. Effects compose
in order through ``PostProcessingChain``, each one's output feeding the next.
``build_post_processing_chain(nuisance_keys)`` assembles a chain from ``EFFECT_REGISTRY``,
silently skipping keys that name no effect, so a caller can pass every nuisance key it has.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


# Abstract base


class SeismogramEffect(ABC):
    """One transformation of a ``{station: {component: waveform}}`` map.

    A subclass implements ``__call__``, takes its own key out of ``nuisance_params``, ignores
    every other key, and returns the input unchanged when its key is absent.
    """

    @abstractmethod
    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        **nuisance_params,
    ) -> dict:
        ...


# PostProcessingChain


class PostProcessingChain:
    """Effects applied in order, each one's output feeding the next; an empty chain is
    the identity.
    """

    def __init__(self, effects: list[SeismogramEffect] | None = None) -> None:
        self.effects: list[SeismogramEffect] = list(effects or [])

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        nuisance_params: dict,
    ) -> dict:
        """The seismogram map with every effect applied in order."""
        result = seismograms_map
        for effect in self.effects:
            result = effect(result, receivers, **nuisance_params)
        return result


# Shared per-station gate (used by amplitude / dropout / coda effects)


def _apply_per_station_gated(seismograms_map: dict, probability, transform) -> dict:
    """Apply ``transform`` to each station independently with the given probability.

    ``probability`` is clipped to ``[0, 1]``; a station that does not fire is copied through
    as float64. The gate is drawn before whatever ``transform`` draws, which fixes the order
    the random number generator is called in.
    """
    p = float(np.clip(probability, 0.0, 1.0))
    result = {}
    for station, components in seismograms_map.items():
        if np.random.uniform() < p:
            result[station] = transform(components)
        else:
            result[station] = {
                comp: trace.astype(np.float64) for comp, trace in components.items()
            }
    return result


# Concrete effects


class AmplitudeErrorEffect(SeismogramEffect):
    """Multiply a station's traces by a random amplitude factor.

    Nuisance key ``amplitude_error``: a per-station activation probability, or, under
    ``always_on``, a strength multiplier on the width (0 is the identity, 1 the configured
    width, 2 double).

    By default one factor per station is drawn from ``uniform(*scale_range)``.
    ``distribution='lognormal'`` draws ``10 ** (log_sigma_dex * N(0, 1))`` instead;
    ``per_component`` draws once per trace rather than once per station. Measured per-trace
    amplitude errors of regional records against one-dimensional synthetics are log-normal at
    about 0.3 dex and independent between a station's components, which is what those two
    switches express.
    """

    #: Default lower bound of the per-station scale factor distribution.
    DEFAULT_SCALE_LOW: float = 0.5
    #: Default upper bound of the per-station scale factor distribution.
    DEFAULT_SCALE_HIGH: float = 2.0
    #: Default log-normal width in dex; measured per-trace widths on regional broadband
    #: records run 0.31 to 0.38 dex.
    DEFAULT_LOG_SIGMA_DEX: float = 0.3
    VALID_DISTRIBUTIONS = ("uniform", "lognormal")

    def __init__(
        self,
        scale_range: tuple[float, float] | None = None,
        distribution: str = "uniform",
        log_sigma_dex: Optional[float] = None,
        per_component: bool = False,
        always_on: bool = False,
    ) -> None:
        if scale_range is not None:
            self._scale_low, self._scale_high = float(scale_range[0]), float(scale_range[1])
        else:
            self._scale_low = self.DEFAULT_SCALE_LOW
            self._scale_high = self.DEFAULT_SCALE_HIGH
        self._distribution = str(distribution).lower()
        if self._distribution not in self.VALID_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {self.VALID_DISTRIBUTIONS}; got {distribution!r}"
            )
        self._log_sigma = (
            float(log_sigma_dex) if log_sigma_dex is not None else self.DEFAULT_LOG_SIGMA_DEX
        )
        if self._log_sigma < 0.0:
            raise ValueError("log_sigma_dex must be >= 0")
        self._per_component = bool(per_component)
        self._always_on = bool(always_on)

    def _draw_scale(self, multiplier: float) -> float:
        """One amplitude scale factor; the uniform distribution ignores ``multiplier``."""
        if self._distribution == "lognormal":
            return float(10.0 ** (multiplier * self._log_sigma * np.random.normal()))
        return float(np.random.uniform(self._scale_low, self._scale_high))

    def _scale_components(self, components: dict, multiplier: float) -> dict:
        if self._per_component:
            return {
                comp: trace.astype(np.float64) * self._draw_scale(multiplier)
                for comp, trace in components.items()
            }
        scale = self._draw_scale(multiplier)
        return {comp: trace.astype(np.float64) * scale for comp, trace in components.items()}

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        amplitude_error: float | None = None,
        **_ignored,
    ) -> dict:
        if amplitude_error is None:
            return seismograms_map

        if self._always_on:
            multiplier = float(amplitude_error)
            if multiplier == 0.0:
                return seismograms_map
            return {
                station: self._scale_components(components, multiplier)
                for station, components in seismograms_map.items()
            }

        def _scale(components):
            return self._scale_components(components, 1.0)

        return _apply_per_station_gated(seismograms_map, amplitude_error, _scale)


class InstrumentDropoutEffect(SeismogramEffect):
    """Zero a whole station's traces, independently per station.

    Nuisance key ``instrument_dropout``: the probability in ``[0, 1]`` that a station is
    zeroed. Absent or zero is the identity.
    """

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        instrument_dropout: float | None = None,
        **_ignored,
    ) -> dict:
        if instrument_dropout is None:
            return seismograms_map

        def _zero(components):
            return {comp: np.zeros_like(trace, dtype=np.float64) for comp, trace in components.items()}

        return _apply_per_station_gated(seismograms_map, instrument_dropout, _zero)


class ComponentDropoutEffect(SeismogramEffect):
    """Zero individual present channels of a station, modelling an event missing a subset
    of them.

    Nuisance key ``component_dropout``: the probability each present channel is dropped,
    drawn independently per channel in ``receivers.iterate()`` order. At least one channel
    per station is always kept, a whole absent station being
    :class:`InstrumentDropoutEffect`'s business.

    It must run after sensor noise is added, so a dropped channel is exactly zero as a
    genuinely absent one is, which is why it is staged post-noise and never baked into a
    simulation.
    """

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        component_dropout: float | None = None,
        **_ignored,
    ) -> dict:
        if component_dropout is None:
            return seismograms_map

        p = float(np.clip(component_dropout, 0.0, 1.0))
        # From the receivers, not the map keys, which also carry zero-filled absent
        # components the adapter inserts.
        present_by_station = {rec.station_name: list(rec.components) for rec in receivers.iterate()}

        result = {}
        for station, components in seismograms_map.items():
            new_components = {comp: trace.astype(np.float64) for comp, trace in components.items()}
            present = [c for c in present_by_station.get(station, []) if c in new_components]
            if len(present) >= 2:
                drop = [c for c in present if np.random.uniform() < p]
                # One channel is restored at random if every one was selected, so the
                # station never becomes all-zero.
                if len(drop) == len(present):
                    keep = present[np.random.randint(len(present))]
                    drop = [c for c in drop if c != keep]
                for c in drop:
                    new_components[c] = np.zeros_like(new_components[c], dtype=np.float64)
            result[station] = new_components
        return result


# Lanczos interpolation helpers (used by TimeShiftErrorEffect)


def _lanczos_kernel_values(x: np.ndarray, order: int) -> np.ndarray:
    """The Lanczos kernel ``sinc(x) * sinc(x / order)``, zero outside ``|x| < order``."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_x = np.pi * x
        sinc_x = np.where(np.abs(x) < 1e-10, 1.0, np.sin(pi_x) / pi_x)
        pi_xa = np.pi * x / order
        sinc_xa = np.where(np.abs(x) < 1e-10, 1.0, np.sin(pi_xa) / pi_xa)
    kernel = sinc_x * sinc_xa
    kernel[np.abs(x) >= order] = 0.0
    return kernel


def _apply_lanczos_shift_batch(
    traces: np.ndarray,
    tau_samples: float,
    order: int = 5,
) -> np.ndarray:
    """Every row of ``(n_traces, n_samples)`` shifted by the same ``tau_samples``.

    The kernel weights depend only on the shift, so it is built once for all the rows; each
    output sample is the same weighted sum as the per-trace form.
    """
    traces_f = np.asarray(traces, dtype=np.float64)
    n = traces_f.shape[-1]
    if abs(tau_samples) < 1e-10:
        return traces_f.copy()

    result = np.zeros_like(traces_f)

    tau_floor = int(np.floor(tau_samples))
    tau_frac = tau_samples - tau_floor

    offsets = np.arange(-order + 1, order + 1, dtype=np.float64)
    weights = _lanczos_kernel_values(offsets - tau_frac, order)
    w_sum = weights.sum()
    if abs(w_sum) > 1e-10:
        weights /= w_sum

    for k_int, w in zip(offsets.astype(int), weights):
        if abs(w) < 1e-12:
            continue
        shift = tau_floor + k_int
        dst_lo = max(0, shift)
        dst_hi = min(n, n + shift)
        src_lo = max(0, -shift)
        src_hi = min(n, n - shift)
        if dst_lo < dst_hi and src_lo < src_hi:
            result[:, dst_lo:dst_hi] += w * traces_f[:, src_lo:src_hi]

    return result


def _apply_lanczos_shift(
    trace: np.ndarray,
    tau_samples: float,
    order: int = 5,
) -> np.ndarray:
    """``trace`` shifted by ``tau_samples``, positive delaying, by Lanczos interpolation.

    ``order`` is the kernel half-width in samples, typically 3 to 8; a higher order leaks
    less spectrally and costs more. The weights are normalised to unit gain at any fractional
    shift, samples outside the trace contribute zero, and the length is unchanged.
    """
    return _apply_lanczos_shift_batch(np.asarray(trace)[np.newaxis, :], tau_samples, order)[0]


# TimeShiftErrorEffect


class TimeShiftErrorEffect(SeismogramEffect):
    """Shift a station's traces in time by a sub-sample amount, via Lanczos interpolation.

    The shift in s is an array-wide common offset plus an independent per-station draw from
    ``N(0, gaussian_sigma)``. The common offset models a constant velocity or source-time
    bias and is drawn once per call, from ``uniform(-uniform_offset, uniform_offset)`` or,
    under ``common_offset_dist='gaussian'``, from ``N(0, common_offset_sigma)``; measured
    array-wide offsets are peaked at zero rather than flat. Positive shifts delay.

    ``sigma_per_1000km`` and ``distance_cap_km`` grow the per-station width with path length
    and need the source location; zero keeps it flat. ``sampling_rate`` in samples per second
    converts the shift to samples and is injected by the caller rather than configured.

    Nuisance key ``time_shift_error``: a switch, not a probability. Absent or zero is the
    identity; any other value makes the effect active, with the magnitude set entirely by the
    configuration.
    """

    #: Default half-width of the array-wide common-offset uniform distribution (s).
    DEFAULT_UNIFORM_OFFSET: float = 0.0
    #: Default Gaussian standard deviation (seconds).
    DEFAULT_GAUSSIAN_SIGMA: float = 1.0
    #: Default Lanczos kernel order.
    DEFAULT_LANCZOS_ORDER: int = 5
    #: Default common-offset distribution ("uniform" for back-compat).
    DEFAULT_COMMON_OFFSET_DIST: str = "uniform"

    def __init__(
        self,
        sampling_rate: float,
        uniform_offset: Optional[float] = None,
        gaussian_sigma: Optional[float] = None,
        lanczos_order: Optional[int] = None,
        common_offset_dist: Optional[str] = None,
        common_offset_sigma: Optional[float] = None,
        sigma_per_1000km: float = 0.0,
        distance_cap_km: Optional[float] = None,
        source_latitude: Optional[float] = None,
        source_longitude: Optional[float] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
        # Timing error of one-dimensional synthetics grows with path length.
        self._sigma_per_1000km = float(sigma_per_1000km)
        if self._sigma_per_1000km < 0.0:
            raise ValueError("sigma_per_1000km must be >= 0")
        self._distance_cap = None if distance_cap_km is None else float(distance_cap_km)
        self._src = (None if source_latitude is None or source_longitude is None
                     else (float(source_latitude), float(source_longitude)))
        self._uniform_offset = (
            float(uniform_offset)
            if uniform_offset is not None
            else self.DEFAULT_UNIFORM_OFFSET
        )
        self._sigma = (
            float(gaussian_sigma)
            if gaussian_sigma is not None
            else self.DEFAULT_GAUSSIAN_SIGMA
        )
        self._order = (
            int(lanczos_order)
            if lanczos_order is not None
            else self.DEFAULT_LANCZOS_ORDER
        )
        self._common_dist = (
            str(common_offset_dist).lower()
            if common_offset_dist is not None
            else self.DEFAULT_COMMON_OFFSET_DIST
        )
        if self._common_dist not in ("uniform", "gaussian"):
            raise ValueError(
                "common_offset_dist must be 'uniform' or 'gaussian'; got "
                f"{common_offset_dist!r}"
            )
        # Defaults to uniform_offset so an existing scale carries over when the distribution
        # is switched.
        self._common_sigma = (
            float(common_offset_sigma)
            if common_offset_sigma is not None
            else self._uniform_offset
        )

    def station_sigmas(self, receivers, source_location=None) -> dict:
        """``{station: width in s}`` of the per-station Gaussian, distance-scaled when
        ``sigma_per_1000km`` is positive and flat otherwise."""
        if self._sigma_per_1000km <= 0.0:
            return {}
        src = (tuple(np.asarray(source_location, dtype=np.float64).ravel()[:2])
               if source_location is not None else self._src)
        if src is None:
            raise ValueError(
                "TimeShiftErrorEffect(sigma_per_1000km > 0) is active but no source location "
                "is available (pass source_location in nuisance_params or "
                "source_latitude/longitude in the effect config)")
        out = {}
        for r in receivers.iterate():
            _, dist = _bearing_and_distance_km(src[0], src[1], r.latitude, r.longitude)
            d = dist if self._distance_cap is None else min(dist, self._distance_cap)
            out[r.station_name] = self._sigma + self._sigma_per_1000km * d / 1000.0
        return out

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        time_shift_error: Optional[float] = None,
        source_location=None,
        **_ignored,
    ) -> dict:
        if time_shift_error is None or float(time_shift_error) == 0.0:
            return seismograms_map

        station_sigma = self.station_sigmas(receivers, source_location)

        if self._common_dist == "gaussian":
            common_offset_s = (
                np.random.normal(0.0, self._common_sigma)
                if self._common_sigma > 0.0
                else 0.0
            )
        else:
            common_offset_s = (
                np.random.uniform(-self._uniform_offset, self._uniform_offset)
                if self._uniform_offset > 0.0
                else 0.0
            )
        result = {}
        for station, components in seismograms_map.items():
            station_shift_s = common_offset_s + np.random.normal(
                0.0, station_sigma.get(station, self._sigma))
            shift_samples = station_shift_s * self._sampling_rate
            comps = list(components)
            if not comps:
                result[station] = {}
                continue
            # All components of a station share the shift, so the kernel is built once.
            traces = np.stack([components[c] for c in comps])
            shifted = _apply_lanczos_shift_batch(traces, shift_samples, self._order)
            result[station] = {c: shifted[j] for j, c in enumerate(comps)}
        return result


# ScatteringCodaEffect helpers and implementation


#: Default coda-tail length as a fraction of the trace length (at ``alpha = 1``).
DEFAULT_CODA_FRACTION: float = 0.25


def _apply_stahler_phase_filter(
    trace: np.ndarray,
    alpha: float,
    coda_fraction: float = DEFAULT_CODA_FRACTION,
) -> np.ndarray:
    """``trace`` convolved with the Stahler and Sigloch (2016) modelling-error filter.

    The filter is a compact FIR of length ``round(coda_fraction * len(trace))`` with a unit
    amplitude spectrum and a phase drawn uniformly on ``[0, alpha * pi / 2]``, so ``alpha``
    sets the coda strength and zero is the identity. Being compact and linearly convolved, it
    leaves quiet stretches quiet and can place no energy before an arrival. The phase is
    pinned to zero at DC and Nyquist, where ``irfft`` discards the imaginary part and a
    non-zero phase would break the unit amplitude.
    """
    n = len(trace)
    trace_f = trace.astype(np.float64)
    if alpha <= 0.0:
        return trace_f.copy()

    coda_len = max(2, int(round(coda_fraction * n)))
    n_bins = coda_len // 2 + 1
    phi = np.random.uniform(0.0, alpha * np.pi / 2.0, size=n_bins)
    phi[0] = 0.0
    if coda_len % 2 == 0:
        phi[-1] = 0.0
    transfer_function = np.fft.irfft(np.exp(1j * phi), n=coda_len)

    return np.convolve(trace_f, transfer_function)[:n]


def _apply_random_coda_filter(
    trace: np.ndarray,
    alpha: float,
    max_coda_fraction: float = DEFAULT_CODA_FRACTION,
) -> np.ndarray:
    """``trace`` convolved with a causal random coda kernel, modelling scattered energy
    trailing the direct arrival.

    The kernel is a unit spike at lag zero followed by an exponentially decaying tail of
    independent taps drawn on ``[-1, 1]``, normalised so total energy is conserved on average.
    ``alpha`` in ``[0, 1]`` scales both the tail amplitude and its length, which is
    ``round(alpha * max_coda_fraction * len(trace))``; zero is the identity. The spike pins
    the onset, so the filter adds no bulk delay, and linear convolution cannot wrap coda back
    to the start of the window as a circular all-pass filter would.
    """
    n = len(trace)
    trace_f = trace.astype(np.float64)
    if alpha <= 0.0:
        return trace_f.copy()

    coda_len = max(1, int(round(alpha * max_coda_fraction * n)))
    taps = np.arange(coda_len + 1, dtype=np.float64)
    envelope = np.exp(-taps / max(coda_len / 3.0, 1.0))
    tail = np.random.uniform(-1.0, 1.0, size=coda_len + 1) * envelope

    kernel = alpha * tail
    kernel[0] = 1.0
    kernel /= np.linalg.norm(kernel)

    return np.convolve(trace_f, kernel)[:n]



# Distance-scaled scattering (far-path decoherence + incoherent coda energy)

def distance_scaled_alpha(
    dist_km,
    alpha_intercept: float = 0.0,
    alpha_per_1000km: float = 0.0,
    distance_cap_km: Optional[float] = None,
):
    """Coda strength in ``[0, 1]`` at source-station distances ``dist_km``.

    Linear in path length up to ``distance_cap_km``, which is the first-order scattering
    expectation: the scattered energy fraction accumulates as distance over mean free path
    (Sato, Fehler and Maeda 2012).
    """
    d = np.asarray(dist_km, dtype=np.float64)
    if distance_cap_km is not None:
        d = np.minimum(d, float(distance_cap_km))
    a = float(alpha_intercept) + float(alpha_per_1000km) * d / 1000.0
    return np.clip(a, 0.0, 1.0)


def distance_tail_energy(
    dist_km,
    excess_dex_per_1000km: float = 0.0,
    distance_cap_km: Optional[float] = None,
):
    """Coda energy relative to the direct arrival at distances ``dist_km``.

    Inverts a linear ramp in root-mean-square excess, in dex per 1000 km, into the tail
    energy the kernel needs. This is the kernel-level target; the excess measured in a
    group-velocity window also depends on the trace's own time structure, so the ramp must be
    calibrated against the same window diagnostic used on the data.
    """
    d = np.asarray(dist_km, dtype=np.float64)
    if distance_cap_km is not None:
        d = np.minimum(d, float(distance_cap_km))
    e = 10.0 ** (2.0 * float(excess_dex_per_1000km) * d / 1000.0) - 1.0
    return np.maximum(e, 0.0)


def _apply_distance_coda_kernel(
    trace: np.ndarray,
    alpha: float,
    tail_energy: float,
    max_coda_fraction: float = DEFAULT_CODA_FRACTION,
) -> np.ndarray:
    """``trace`` convolved with a delayed-replica coda kernel of energy ``tail_energy``.

    Built as :func:`_apply_random_coda_filter` but with the tail scaled to the given energy
    and the kernel left un-normalised, so convolution both decorrelates the waveform and adds
    incoherent energy behind every arrival, which are the two signatures a long path leaves.
    A non-positive ``tail_energy`` or ``alpha`` is the identity.
    """
    n = len(trace)
    trace_f = trace.astype(np.float64)
    if alpha <= 0.0 or tail_energy <= 0.0:
        return trace_f.copy()
    coda_len = max(1, int(round(alpha * max_coda_fraction * n)))
    taps = np.arange(coda_len + 1, dtype=np.float64)
    envelope = np.exp(-taps / max(coda_len / 3.0, 1.0))
    tail = np.random.uniform(-1.0, 1.0, size=coda_len + 1) * envelope
    tail[0] = 0.0
    norm = np.linalg.norm(tail)
    if norm <= 0.0:
        return trace_f.copy()
    tail *= np.sqrt(float(tail_energy)) / norm
    kernel = tail
    kernel[0] = 1.0
    return np.convolve(trace_f, kernel)[:n]


class ScatteringCodaEffect(SeismogramEffect):
    """Add scattered coda trailing the direct arrivals, per station.

    Nuisance key ``scattering_coda``: a per-station activation probability, or, in distance
    mode, a strength multiplier on the ramps below.

    By default a gated station draws its own strength from ``uniform(*alpha_range)``, shared
    across its components, or uses a fixed ``alpha``. ``mode='causal'`` convolves with the
    spike-plus-decaying-tail kernel of :func:`_apply_random_coda_filter`; ``mode='stahler'``
    convolves with the unit-amplitude random-phase filter of
    :func:`_apply_stahler_phase_filter`. Both are causal and leave quiet stretches quiet.

    ``distance_mode`` replaces the gate and the uniform draw with ramps in source-station
    distance: the strength from :func:`distance_scaled_alpha`, jittered by ``alpha_jitter``,
    and, in causal mode, the tail energy from :func:`distance_tail_energy`. It models the way
    one-dimensional Green's functions decohere with path length while the observed
    surface-wave energy exceeds their prediction, neither of which the distance-blind gate can
    express. The ramp defaults are inert, the calibrated slopes being configuration. The
    source location comes from the nuisance dict or the constructor, and distance mode raises
    without one.
    """

    #: Default per-station ``alpha`` sampling range (uniform).
    DEFAULT_ALPHA_RANGE = (0.0, 1.0)
    VALID_MODES = ("causal", "stahler")

    def __init__(
        self,
        alpha: Optional[float] = None,
        alpha_range: Optional[tuple] = None,
        mode: str = "causal",
        coda_fraction: Optional[float] = None,
        distance_mode: bool = False,
        alpha_intercept: float = 0.0,
        alpha_per_1000km: float = 0.0,
        alpha_jitter: float = 0.0,
        excess_dex_per_1000km: float = 0.0,
        distance_cap_km: Optional[float] = None,
        source_latitude: Optional[float] = None,
        source_longitude: Optional[float] = None,
    ) -> None:
        if alpha is not None and alpha_range is not None:
            raise ValueError(
                "ScatteringCodaEffect: specify only one of `alpha` (fixed) or "
                "`alpha_range` (per-station sampled)."
            )
        self._distance_mode = bool(distance_mode)
        self._alpha_intercept = float(alpha_intercept)
        self._alpha_per_1000km = float(alpha_per_1000km)
        self._alpha_jitter = float(alpha_jitter)
        self._excess_dex = float(excess_dex_per_1000km)
        self._distance_cap = None if distance_cap_km is None else float(distance_cap_km)
        self._src = (None if source_latitude is None or source_longitude is None
                     else (float(source_latitude), float(source_longitude)))
        if self._distance_mode:
            if not (0.0 <= self._alpha_jitter < 1.0):
                raise ValueError("ScatteringCodaEffect: alpha_jitter must be in [0, 1)")
            if self._alpha_per_1000km < 0.0 or self._excess_dex < 0.0:
                raise ValueError(
                    "ScatteringCodaEffect: alpha_per_1000km and excess_dex_per_1000km must be >= 0"
                )
        if alpha is not None:
            self._alpha_low = self._alpha_high = float(alpha)
        else:
            rng = alpha_range if alpha_range is not None else self.DEFAULT_ALPHA_RANGE
            self._alpha_low, self._alpha_high = float(rng[0]), float(rng[1])
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"ScatteringCodaEffect mode must be one of {self.VALID_MODES}, got {mode!r}"
            )
        self._mode = mode
        self._coda_fraction = (
            float(coda_fraction)
            if coda_fraction is not None
            else DEFAULT_CODA_FRACTION
        )

    def _filter_trace(self, trace: np.ndarray, alpha: float) -> np.ndarray:
        if self._mode == "stahler":
            return _apply_stahler_phase_filter(trace, alpha, self._coda_fraction)
        return _apply_random_coda_filter(trace, alpha, self._coda_fraction)

    @property
    def distance_mode(self) -> bool:
        return self._distance_mode

    def _resolve_source(self, source_location):
        src = (tuple(np.asarray(source_location, dtype=np.float64).ravel()[:2])
               if source_location is not None else self._src)
        if src is None:
            raise ValueError(
                "ScatteringCodaEffect(distance_mode=True) is active but no source location "
                "is available (pass source_location in nuisance_params or "
                "source_latitude/longitude in the effect config)")
        return src

    def station_scattering_params(self, receivers, source_latlon, multiplier: float = 1.0) -> dict:
        """``{station: (distance in km, strength, tail energy)}``, the strength being the
        ramp value before the per-station jitter draw.
        """
        src_lat, src_lon = source_latlon
        m = float(multiplier)
        out = {}
        for r in receivers.iterate():
            _, dist = _bearing_and_distance_km(src_lat, src_lon, r.latitude, r.longitude)
            a = float(np.clip(m * distance_scaled_alpha(
                dist, self._alpha_intercept, self._alpha_per_1000km, self._distance_cap), 0.0, 1.0))
            e = float(distance_tail_energy(dist, m * self._excess_dex, self._distance_cap))
            out[r.station_name] = (float(dist), a, e)
        return out

    def _apply_distance_mode(self, seismograms_map, receivers, multiplier, source_location):
        m = float(multiplier)
        if m == 0.0:
            return seismograms_map
        src = self._resolve_source(source_location)
        params = self.station_scattering_params(receivers, src, m)
        result = {}
        for station, components in seismograms_map.items():
            if station not in params:
                raise KeyError(
                    f"ScatteringCodaEffect(distance_mode=True): station {station!r} has no "
                    "receiver coordinates")
            dist, a_nom, e_tail = params[station]
            jitter = (np.random.uniform(-self._alpha_jitter, self._alpha_jitter)
                      if self._alpha_jitter > 0.0 else 0.0)
            alpha = float(np.clip(a_nom * (1.0 + jitter), 0.0, 1.0))
            if self._mode == "stahler":
                result[station] = {
                    comp: _apply_stahler_phase_filter(trace, alpha, self._coda_fraction)
                    for comp, trace in components.items()}
            else:
                result[station] = {
                    comp: _apply_distance_coda_kernel(trace, alpha, e_tail, self._coda_fraction)
                    for comp, trace in components.items()}
        return result

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        scattering_coda: Optional[float] = None,
        source_location=None,
        **_ignored,
    ) -> dict:
        if scattering_coda is None:
            return seismograms_map
        if self._distance_mode:
            return self._apply_distance_mode(seismograms_map, receivers,
                                             scattering_coda, source_location)

        def _coda(components):
            alpha = np.random.uniform(self._alpha_low, self._alpha_high)
            return {comp: self._filter_trace(trace, alpha) for comp, trace in components.items()}

        return _apply_per_station_gated(seismograms_map, scattering_coda, _coda)


# Anisotropy injection effects


def _bearing_and_distance_km(src_lat, src_lon, sta_lat, sta_lon):
    """``(azimuth in deg clockwise from north, distance in km)`` on a spherical Earth."""
    la1, lo1 = np.deg2rad(src_lat), np.deg2rad(src_lon)
    la2, lo2 = np.deg2rad(sta_lat), np.deg2rad(sta_lon)
    dlon = lo2 - lo1
    az = np.arctan2(
        np.sin(dlon) * np.cos(la2),
        np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon))
    hav = (np.sin((la2 - la1) / 2.0) ** 2
           + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2.0) ** 2)
    dist = 2.0 * 6371.0 * np.arcsin(np.sqrt(hav))
    return np.rad2deg(az) % 360.0, dist


class AzimuthalAnisotropyEffect(SeismogramEffect):
    """Delay each station by the travel-time anomaly weak azimuthal anisotropy gives it.

    With the speed varying as ``V(phi) = V0 (1 + A cos 2(phi - phi_fast))``, a path of length
    D at source-station azimuth phi accumulates ``-(D / V0) A cos(2 (phi - phi_fast))``, so
    the fast azimuth arrives early. The delay grows with path length and is coherent across
    the array, unlike the independent per-station shifts of :class:`TimeShiftErrorEffect`. All
    components of a station share it, applied by Lanczos interpolation; there is no randomness.

    Nuisance key ``azimuthal_anisotropy``: a strength multiplier on ``aniso_fraction``, absent
    or zero being the identity. The source location comes from the nuisance dict or the
    constructor, and an active effect raises without one rather than silently doing nothing.
    """

    DEFAULT_REF_VELOCITY_KMS: float = 3.5
    DEFAULT_LANCZOS_ORDER: int = 5

    def __init__(
        self,
        sampling_rate: float,
        fast_azimuth_deg: float = 45.0,
        aniso_fraction: float = 0.0125,
        ref_velocity_kms: Optional[float] = None,
        source_latitude: Optional[float] = None,
        source_longitude: Optional[float] = None,
        lanczos_order: Optional[int] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
        self._fast_az = float(fast_azimuth_deg)
        self._fraction = float(aniso_fraction)
        self._v0 = float(ref_velocity_kms if ref_velocity_kms is not None
                         else self.DEFAULT_REF_VELOCITY_KMS)
        self._src = (None if source_latitude is None or source_longitude is None
                     else (float(source_latitude), float(source_longitude)))
        self._order = int(lanczos_order if lanczos_order is not None
                          else self.DEFAULT_LANCZOS_ORDER)

    def station_delays(self, receivers, source_latlon) -> dict:
        """``{station: delay in s}`` at multiplier one."""
        src_lat, src_lon = source_latlon
        out = {}
        for r in receivers.iterate():
            az, dist = _bearing_and_distance_km(src_lat, src_lon,
                                                r.latitude, r.longitude)
            out[r.station_name] = float(
                -(dist / self._v0) * self._fraction
                * np.cos(2.0 * np.deg2rad(az - self._fast_az)))
        return out

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        azimuthal_anisotropy: Optional[float] = None,
        source_location=None,
        **_ignored,
    ) -> dict:
        if azimuthal_anisotropy is None or float(azimuthal_anisotropy) == 0.0:
            return seismograms_map
        mult = float(azimuthal_anisotropy)
        src = (tuple(np.asarray(source_location, float)[:2])
               if source_location is not None else self._src)
        if src is None:
            raise ValueError(
                "AzimuthalAnisotropyEffect is active but no source location is "
                "available (pass source_location in nuisance_params or "
                "source_latitude/longitude in the effect config)")
        delays = self.station_delays(receivers, src)
        result = {}
        for station, components in seismograms_map.items():
            comps = list(components)
            if station not in delays or not comps:
                result[station] = dict(components)
                continue
            shift_samples = mult * delays[station] * self._sampling_rate
            traces = np.stack([components[c] for c in comps])
            shifted = _apply_lanczos_shift_batch(traces, shift_samples, self._order)
            result[station] = {c: shifted[j] for j, c in enumerate(comps)}
        return result


class ShearSplittingEffect(SeismogramEffect):
    """Split the horizontals of each station, the Silver and Chan (1991) operator.

    Station-side and in the geographic frame: rotate north and east into the fast and slow
    frame given by ``fast_azimuth_deg``, delay the slow trace by ``delay_s``, rotate back. The
    vertical is untouched, a station missing either horizontal passes through, and there is no
    randomness.

    Nuisance key ``shear_wave_splitting``: a multiplier on ``delay_s``, absent or zero being
    the identity.
    """

    DEFAULT_LANCZOS_ORDER: int = 5

    def __init__(
        self,
        sampling_rate: float,
        fast_azimuth_deg: float = 45.0,
        delay_s: float = 0.149,
        lanczos_order: Optional[int] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
        self._fast_az = float(fast_azimuth_deg)
        self._delay = float(delay_s)
        self._order = int(lanczos_order if lanczos_order is not None
                          else self.DEFAULT_LANCZOS_ORDER)

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        shear_wave_splitting: Optional[float] = None,
        **_ignored,
    ) -> dict:
        if shear_wave_splitting is None or float(shear_wave_splitting) == 0.0:
            return seismograms_map
        delay_samples = (float(shear_wave_splitting) * self._delay
                         * self._sampling_rate)
        phi = np.deg2rad(self._fast_az)
        c, s = np.cos(phi), np.sin(phi)
        result = {}
        for station, components in seismograms_map.items():
            if "E" not in components or "N" not in components:
                result[station] = dict(components)
                continue
            north = np.asarray(components["N"], float)
            east = np.asarray(components["E"], float)
            fast = c * north + s * east
            slow = -s * north + c * east
            slow = _apply_lanczos_shift_batch(slow[None, :], delay_samples,
                                              self._order)[0]
            new = dict(components)
            new["N"] = c * fast - s * slow
            new["E"] = s * fast + c * slow
            result[station] = new
        return result


# Registry and factory

#: Nuisance parameter key to effect class; a new effect is registered here.

class DispersionSpreadEffect(SeismogramEffect):
    """Delay each station by a frequency-dependent travel-time error: a pure phase delay
    ``u'(f) = u(f) exp(-2 pi i f tau(f))``.

    The delay is given at the octave centres ``octave_centres_s``, interpolated in log period
    between them and held constant outside. Per station and octave it is the product of the
    nuisance multiplier, a standard normal and a width that grows linearly with source-station
    distance from ``sigma_intercept_s`` at ``sigma_per_1000km_s`` per 1000 km.

    The correlation of those normals is the physics: a crust that is too fast delays every
    octave of the path the same way, so ``octave_correlation`` shares one draw across octaves
    by default, and ``common_fraction`` puts that share of the variance into a single
    array-wide draw, which is the coherent same-sign far-station delay independent per-station
    sampling can never produce. A systematic bias belongs in the reference model, not here.

    Nuisance key ``dispersion_spread``: a strength multiplier. Needs the source location, so
    it is a simulation-stage effect; ``sampling_rate`` is injected by the caller.
    """

    DEFAULT_OCTAVE_CENTRES_S = (12.5, 17.5, 25.0, 40.0)

    def __init__(
        self,
        sampling_rate: float,
        octave_centres_s=None,
        sigma_intercept_s=None,
        sigma_per_1000km_s=None,
        distance_cap_km: Optional[float] = None,
        octave_correlation: float = 1.0,
        common_fraction: float = 0.0,
        source_latitude: Optional[float] = None,
        source_longitude: Optional[float] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
        self._T = np.array(octave_centres_s if octave_centres_s is not None
                           else self.DEFAULT_OCTAVE_CENTRES_S, dtype=np.float64)
        n = len(self._T)
        if n < 1 or np.any(np.diff(self._T) <= 0):
            raise ValueError("octave_centres_s must be strictly increasing periods (s)")
        self._a = np.zeros(n) if sigma_intercept_s is None else np.asarray(sigma_intercept_s, float)
        self._b = np.zeros(n) if sigma_per_1000km_s is None else np.asarray(sigma_per_1000km_s, float)
        if self._a.shape != (n,) or self._b.shape != (n,):
            raise ValueError("sigma_intercept_s / sigma_per_1000km_s must have one entry per octave centre")
        if np.any(self._a < 0) or np.any(self._b < 0):
            raise ValueError("dispersion sigmas must be >= 0")
        self._cap = None if distance_cap_km is None else float(distance_cap_km)
        self._rho = float(octave_correlation)
        self._common = float(common_fraction)
        if not (0.0 <= self._rho <= 1.0) or not (0.0 <= self._common <= 1.0):
            raise ValueError("octave_correlation and common_fraction must lie in [0, 1]")
        self._src = (None if source_latitude is None or source_longitude is None
                     else (float(source_latitude), float(source_longitude)))

    def station_sigmas(self, receivers, source_location=None) -> dict:
        """``{station: width per octave}`` in s."""
        src = (tuple(np.asarray(source_location, dtype=np.float64).ravel()[:2])
               if source_location is not None else self._src)
        if src is None:
            raise ValueError(
                "DispersionSpreadEffect is active but no source location is available "
                "(pass source_location in nuisance_params or source_latitude/longitude "
                "in the effect config)")
        out = {}
        for r in receivers.iterate():
            _, dist = _bearing_and_distance_km(src[0], src[1], r.latitude, r.longitude)
            d = dist if self._cap is None else min(dist, self._cap)
            out[r.station_name] = self._a + self._b * d / 1000.0
        return out

    def tau_of_freq(self, freqs: np.ndarray, tau_octaves: np.ndarray) -> np.ndarray:
        """Per-octave delays in s interpolated onto ``freqs`` in log period, held constant
        outside the octave centres."""
        with np.errstate(divide="ignore"):
            per = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), self._T[-1])
        per = np.clip(per, self._T[0], self._T[-1])
        if len(self._T) == 1:
            return np.full_like(freqs, tau_octaves[0], dtype=np.float64)
        return np.interp(np.log(per), np.log(self._T), tau_octaves)

    @staticmethod
    def _delay(trace: np.ndarray, tau_f: np.ndarray, dt: float) -> np.ndarray:
        n = len(trace)
        fr = np.fft.rfftfreq(n, d=dt)
        return np.fft.irfft(np.fft.rfft(trace) * np.exp(-2j * np.pi * fr * tau_f), n=n)

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        dispersion_spread: Optional[float] = None,
        source_location=None,
        **_ignored,
    ) -> dict:
        if dispersion_spread is None or float(dispersion_spread) == 0.0:
            return seismograms_map
        m = float(dispersion_spread)
        sig = self.station_sigmas(receivers, source_location)
        n_oct = len(self._T)
        z_common = np.random.normal(size=n_oct)
        z_common_shared = np.random.normal()
        dt = 1.0 / self._sampling_rate
        result = {}
        for station, components in seismograms_map.items():
            z_shared = np.random.normal()
            z_ind = np.random.normal(size=n_oct)
            z_sta = np.sqrt(self._rho) * z_shared + np.sqrt(1.0 - self._rho) * z_ind
            z_com = np.sqrt(self._rho) * z_common_shared + np.sqrt(1.0 - self._rho) * z_common
            z = np.sqrt(self._common) * z_com + np.sqrt(1.0 - self._common) * z_sta
            tau_oct = m * z * sig.get(station, np.zeros(n_oct))
            comps = list(components)
            if not comps or not np.any(tau_oct):
                result[station] = {c: components[c].astype(np.float64) for c in comps}
                continue
            n = len(components[comps[0]])
            fr = np.fft.rfftfreq(n, d=dt)
            tau_f = self.tau_of_freq(fr, tau_oct)
            result[station] = {c: self._delay(components[c].astype(np.float64), tau_f, dt)
                               for c in comps}
        return result


EFFECT_REGISTRY: dict[str, type[SeismogramEffect]] = {
    "amplitude_error": AmplitudeErrorEffect,
    "instrument_dropout": InstrumentDropoutEffect,
    "time_shift_error": TimeShiftErrorEffect,
    "scattering_coda": ScatteringCodaEffect,
    "component_dropout": ComponentDropoutEffect,
    "azimuthal_anisotropy": AzimuthalAnisotropyEffect,
    "shear_wave_splitting": ShearSplittingEffect,
    "dispersion_spread": DispersionSpreadEffect,
}


#: Nuisance keys eligible for pre-noise training augmentation, folded into the clean signal.
AUGMENTABLE_EFFECT_KEYS: tuple[str, ...] = (
    "amplitude_error",
    "instrument_dropout",
    "time_shift_error",
    "scattering_coda",
)


#: Nuisance keys eligible for post-noise augmentation, applied to the data plus noise.
#: ``component_dropout`` must run there so a dropped channel is exactly zero.
POST_NOISE_EFFECT_KEYS: tuple[str, ...] = (
    "component_dropout",
)


#: Nuisance keys that augment the source-location conditioning vector rather than the
#: waveform, so they have no effect class and the dataloader applies them.
CONDITIONING_AUGMENTABLE_KEYS: tuple[str, ...] = (
    "source_location_error",
)


#: The effect keys eligible at each augmentation stage.
_STAGE_EFFECT_KEYS: dict[str, tuple[str, ...]] = {
    "training_augmentation": AUGMENTABLE_EFFECT_KEYS,
    "training_augmentation_post_noise": POST_NOISE_EFFECT_KEYS,
}


# Map <-> stacked-array adapter (lets the SAME effects run in the dataloader)


def _array_to_map(D: np.ndarray, receivers, components):
    """``(seismograms map, station names)`` from a ``(n_stations, n_components, n_samples)``
    array, in receiver order and loader component order.
    """
    station_names = [rec.station_name for rec in receivers.iterate()]
    seismograms_map = {
        station: {comp: D[i, j] for j, comp in enumerate(components)}
        for i, station in enumerate(station_names)
    }
    return seismograms_map, station_names


def _map_to_array(seismograms_map: dict, station_names, components) -> np.ndarray:
    """The inverse of :func:`_array_to_map`: back to ``(n_stations, n_components, n_samples)``."""
    return np.array(
        [[seismograms_map[station][comp] for comp in components] for station in station_names],
        dtype=np.float64,
    )


def apply_chain_to_array(
    chain: PostProcessingChain,
    D: np.ndarray,
    receivers,
    components,
    nuisance_params: dict,
) -> np.ndarray:
    """``D``, shaped ``(n_stations, n_components, n_samples)``, with ``chain`` applied.

    Lets the same effect classes serve as training-time augmentation on the dataloader's
    stacked array. ``receivers`` and ``components`` give the station and component orders that
    array is in. An empty chain returns ``D`` unchanged.
    """
    if not chain.effects:
        return D
    D = np.asarray(D)
    # A silent mismatch would scramble stations rather than raise.
    n_stations = len(list(receivers.iterate()))
    if D.ndim != 3 or D.shape[0] != n_stations or D.shape[1] != len(components):
        raise ValueError(
            f"apply_chain_to_array: D shape {D.shape} is incompatible with "
            f"{n_stations} receivers x {len(components)} components "
            f"(expected ({n_stations}, {len(components)}, T))."
        )
    seismograms_map, station_names = _array_to_map(D, receivers, components)
    processed = chain(seismograms_map, receivers, nuisance_params)
    return _map_to_array(processed, station_names, components)


def _fiducial_scalar(value) -> float:
    """The activation scalar of a nuisance fiducial entry, so ``[0.3]`` gives 0.3."""
    arr = np.ravel(value)
    return float(arr[0])


def build_augmentation_chain(
    nuisance: dict,
    nuisance_stage: dict,
    effect_configs: Optional[dict] = None,
    sampling_rate: Optional[float] = None,
    stage: str = "training_augmentation",
) -> Tuple[PostProcessingChain, dict]:
    """``(chain, nuisance_params)`` for the training-time augmentation at ``stage``.

    ``nuisance`` is ``{key: fiducial values}`` and ``nuisance_stage`` is ``{key: stage}``, a
    key absent from it defaulting to the simulation stage and so not augmented. Only keys
    staged at ``stage`` and eligible there are included. Each key's activation value in
    ``nuisance_params`` is its configured fiducial scalar; the magnitudes live in
    ``effect_configs``, and the effects draw their own randomness per call. ``sampling_rate``
    in samples per second is injected into the shift effect. An empty selection gives an empty
    chain, which the dataloader treats as no augmentation.
    """
    configs = dict(effect_configs or {})
    eligible = _STAGE_EFFECT_KEYS.get(stage, ())
    aug_keys = [
        key for key in nuisance
        if key in eligible
        and nuisance_stage.get(key, "simulation") == stage
    ]
    if "time_shift_error" in aug_keys and sampling_rate is not None:
        configs["time_shift_error"] = dict(configs.get("time_shift_error", {}))
        configs["time_shift_error"]["sampling_rate"] = sampling_rate

    chain = build_post_processing_chain(aug_keys, configs)
    # The configured fiducial value, not a hardcoded 1.0, which would perturb every station.
    nuisance_params = {key: _fiducial_scalar(nuisance[key]) for key in aug_keys}
    return chain, nuisance_params


def build_augmentation_chain_from_parameters(parameters, sampling_rate=None,
                                             stage="training_augmentation"):
    """``(chain, nuisance_params)`` as :func:`build_augmentation_chain`, unpacking the
    nuisance blocks from a parsed ``ModelParameters``.
    """
    return build_augmentation_chain(
        parameters.nuisance,
        getattr(parameters, "nuisance_stage", {}),
        getattr(parameters, "nuisance_effect_config", {}),
        sampling_rate=sampling_rate,
        stage=stage,
    )


def build_post_processing_chain(
    nuisance_keys,
    effect_configs: Optional[dict] = None,
) -> PostProcessingChain:
    """A chain of the effects :data:`EFFECT_REGISTRY` has for ``nuisance_keys``.

    A key naming no effect is skipped, so a caller can pass every nuisance key it has.
    ``effect_configs`` is ``{nuisance key: constructor keyword arguments}``.
    """
    configs = effect_configs or {}
    effects = [
        EFFECT_REGISTRY[key](**configs.get(key, {}))
        for key in nuisance_keys
        if key in EFFECT_REGISTRY
    ]
    return PostProcessingChain(effects)
