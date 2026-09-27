"""Scattered coda behind the direct arrivals, gated per station or ramped with distance.

The coda filters (causal random tail, Stahler and Sigloch random phase, distance-scaled
delayed replicas) and the distance ramps are module functions; ``ScatteringCodaEffect``
chooses and applies them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from seismo_sbi.nuisance_effects.seismogram_effect import SeismogramEffect, _bearing_and_distance_km


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
        self._src = self._configured_source(source_latitude, source_longitude)
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

        return self._apply_per_station_gated(seismograms_map, scattering_coda, _coda)
