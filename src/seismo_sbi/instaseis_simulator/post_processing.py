"""Post-processing effects applied to synthetic seismograms after simulation.

This module provides a composable chain of ``SeismogramEffect`` callables that
can be wired into ``Simulator.run_simulation()`` to add nuisance-parameter-driven
modifications to synthetic waveforms *after* the forward model returns.

Design goals
------------
- Each effect is responsible for exactly one nuisance key.
- An effect that does not find its key in ``nuisance_params`` is a strict no-op.
- Effects do **not** mutate the input ``seismograms_map``; they return a new dict.
- Effects compose sequentially through ``PostProcessingChain``; the output of one
  effect is the input to the next.
- ``build_post_processing_chain(nuisance_keys)`` constructs a chain from the
  ``EFFECT_REGISTRY`` — unknown keys (e.g. ``source_location``) are silently
  skipped, so callers can pass the full ``parameters.nuisance.keys()`` list.

Usage example
-------------
::

    from seismo_sbi.instaseis_simulator.post_processing import build_post_processing_chain

    chain = build_post_processing_chain(parameters.nuisance.keys())
    processed = chain(seismograms_map, receivers, nuisance_params_dict)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SeismogramEffect(ABC):
    """A single post-processing transformation applied to a seismogram map.

    Subclasses implement ``__call__`` and extract their own key(s) from
    ``nuisance_params``.  If the relevant key is absent the method must return
    the input map unchanged (identity behaviour).

    Parameters
    ----------
    seismograms_map:
        ``{station_name: {component: np.ndarray}}`` — the raw simulator output.
    receivers:
        ``Receivers`` object (provides station ordering and metadata).
    **nuisance_params:
        Key-value pairs from the sampled nuisance parameter dict; effects
        select their own key and ignore everything else.

    Returns
    -------
    dict
        A new (or the same) ``{station_name: {component: np.ndarray}}`` dict
        with the effect applied.
    """

    @abstractmethod
    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        **nuisance_params,
    ) -> dict:
        ...


# ---------------------------------------------------------------------------
# PostProcessingChain
# ---------------------------------------------------------------------------


class PostProcessingChain:
    """Sequential composition of zero or more :class:`SeismogramEffect` objects.

    An empty chain is the identity transformation.

    Parameters
    ----------
    effects:
        Ordered list of :class:`SeismogramEffect` instances to apply in
        sequence.  Defaults to an empty list.
    """

    def __init__(self, effects: list[SeismogramEffect] | None = None) -> None:
        self.effects: list[SeismogramEffect] = list(effects or [])

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        nuisance_params: dict,
    ) -> dict:
        """Apply all effects in order.

        Parameters
        ----------
        seismograms_map:
            ``{station_name: {component: np.ndarray}}``
        receivers:
            ``Receivers`` instance.
        nuisance_params:
            Full nuisance parameter dict — each effect extracts its own key.

        Returns
        -------
        dict
            Post-processed seismogram map.
        """
        result = seismograms_map
        for effect in self.effects:
            result = effect(result, receivers, **nuisance_params)
        return result


# ---------------------------------------------------------------------------
# Shared per-station gate (used by amplitude / dropout / coda effects)
# ---------------------------------------------------------------------------


def _apply_per_station_gated(seismograms_map: dict, probability, transform) -> dict:
    """Apply ``transform`` to each station independently with the given probability.

    For each station a Bernoulli gate is drawn (``np.random.uniform() < p``); if it
    fires, ``transform(components)`` produces the new ``{component: trace}`` dict for
    that station, otherwise the station's traces are passed through (copied to
    ``float64``).  This is the common skeleton of the probability-gated effects; the
    RNG call order is **gate draw first, then whatever ``transform`` draws** — matching
    the original per-effect loops so seeded behaviour is unchanged.

    Parameters
    ----------
    seismograms_map:
        ``{station: {component: np.ndarray}}``.
    probability:
        Per-station activation probability, clipped to ``[0, 1]``.
    transform:
        ``components_dict -> components_dict`` applied to a selected station.
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


# ---------------------------------------------------------------------------
# Concrete effects
# ---------------------------------------------------------------------------


class AmplitudeErrorEffect(SeismogramEffect):
    """Per-station stochastic amplitude modulation.

    Two-stage model for each station, applied independently:

    1. **Dropout stage** — decide whether to apply modulation at all.
       The probability is ``amplitude_error`` (a value in ``[0, 1]``).
       ``amplitude_error = 0.0`` → no station is ever modulated (identity).
       ``amplitude_error = 1.0`` → every station is modulated.

    2. **Scale stage** — if the station is selected, multiply all its
       component traces by a scale factor drawn uniformly from
       ``[scale_low, scale_high]``.  The default range is
       ``[DEFAULT_SCALE_LOW, DEFAULT_SCALE_HIGH]``.  Each station gets an
       **independent** draw, so different stations can have different scales
       within the same simulation.

    Nuisance key: ``amplitude_error`` — a probability in ``[0, 1]``.

    If ``amplitude_error`` is absent from ``nuisance_params`` the input map is
    returned unchanged.

    Parameters
    ----------
    scale_range:
        Optional ``(low, high)`` tuple overriding the default scale factor
        range.  Useful in tests to make the output deterministic (e.g.
        ``scale_range=(2.0, 2.0)`` always applies a factor of exactly 2).
        Defaults to ``(DEFAULT_SCALE_LOW, DEFAULT_SCALE_HIGH)``.

    Note
    ----
    This effect is stochastic; different calls with the same inputs may
    produce different outputs.  Use ``numpy.random.seed`` in tests that
    require reproducibility.
    """

    #: Default lower bound of the per-station scale factor distribution.
    DEFAULT_SCALE_LOW: float = 0.5
    #: Default upper bound of the per-station scale factor distribution.
    DEFAULT_SCALE_HIGH: float = 2.0

    def __init__(
        self,
        scale_range: tuple[float, float] | None = None,
    ) -> None:
        if scale_range is not None:
            self._scale_low, self._scale_high = float(scale_range[0]), float(scale_range[1])
        else:
            self._scale_low = self.DEFAULT_SCALE_LOW
            self._scale_high = self.DEFAULT_SCALE_HIGH

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

        def _scale(components):
            # Independent per-station scale factor
            scale = np.random.uniform(self._scale_low, self._scale_high)
            return {comp: trace.astype(np.float64) * scale for comp, trace in components.items()}

        return _apply_per_station_gated(seismograms_map, amplitude_error, _scale)


class InstrumentDropoutEffect(SeismogramEffect):
    """Randomly zero-out entire stations with a given probability.

    Nuisance key: ``instrument_dropout`` — a probability in ``[0, 1]`` that
    each station's traces are replaced with zeros.  Each station is sampled
    independently.

    Special cases:
    - ``instrument_dropout = 0.0`` → no station is zeroed (identity).
    - ``instrument_dropout = 1.0`` → all stations are zeroed.
    - Key absent → input map returned unchanged.

    Note
    ----
    This effect is stochastic; different calls with the same inputs may produce
    different outputs.  To fix the random state in tests use
    ``numpy.random.seed``.
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
            # zero out all components for this station
            return {comp: np.zeros_like(trace, dtype=np.float64) for comp, trace in components.items()}

        return _apply_per_station_gated(seismograms_map, instrument_dropout, _zero)


class ComponentDropoutEffect(SeismogramEffect):
    """Randomly zero individual *present* components (channels) per station.

    Models events that are missing a subset of channels (different from the fixed
    ``components.json`` pattern a model trains on).  A missing component is represented
    everywhere as an **exactly-zero** channel, so this effect simply zeros selected
    present channels.

    Nuisance key: ``component_dropout`` — a per-channel drop probability ``p`` in
    ``[0, 1]``.  For each station, every *present* component (``receiver.components``)
    is independently dropped with probability ``p`` (Bernoulli).  At least one present
    component is always kept (a station never becomes all-zero — full-station absence is
    the domain of :class:`InstrumentDropoutEffect` / variable-station masking), so
    stations with a single present component are never touched.

    Special cases:
    - ``component_dropout = 0.0`` → identity.
    - ``component_dropout = 1.0`` → all-but-one present channel zeroed, per station.
    - Key absent → input map returned unchanged.

    **Ordering contract (critical):** this effect must be applied to the data *after*
    sensor noise has been added, so a dropped channel is exactly zero (matching a
    genuinely-absent channel).  Applying it before noise would leave ``0 + noise``.  It
    is therefore staged ``training_augmentation_post_noise`` (see
    :data:`POST_NOISE_EFFECT_KEYS`), never baked into a simulation.

    Note
    ----
    Stochastic; seed ``numpy.random`` in tests for reproducibility.  Draws proceed in
    ``receivers.iterate()`` order, one Bernoulli per present channel.
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
        # Present components per station come from the receivers (NOT the map keys, which
        # also carry zero-filled *absent* components the adapter inserts).
        present_by_station = {rec.station_name: list(rec.components) for rec in receivers.iterate()}

        result = {}
        for station, components in seismograms_map.items():
            new_components = {comp: trace.astype(np.float64) for comp, trace in components.items()}
            present = [c for c in present_by_station.get(station, []) if c in new_components]
            if len(present) >= 2:
                drop = [c for c in present if np.random.uniform() < p]
                # Keep >= 1 present channel: if every present channel was selected, restore
                # one at random so the station never becomes all-zero.
                if len(drop) == len(present):
                    keep = present[np.random.randint(len(present))]
                    drop = [c for c in drop if c != keep]
                for c in drop:
                    new_components[c] = np.zeros_like(new_components[c], dtype=np.float64)
            result[station] = new_components
        return result


# ---------------------------------------------------------------------------
# Lanczos interpolation helpers (used by TimeShiftErrorEffect)
# ---------------------------------------------------------------------------


def _lanczos_kernel_values(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate the Lanczos kernel L(x) = sinc(x) * sinc(x/a) element-wise.

    The kernel is zero outside the support ``|x| >= order``.  At ``x = 0``
    the limit value 1.0 is returned.

    Parameters
    ----------
    x:
        Argument values (any shape).
    order:
        Kernel half-width / number of lobes (``a`` in the literature).

    Returns
    -------
    np.ndarray
        Kernel values with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_x = np.pi * x
        sinc_x = np.where(np.abs(x) < 1e-10, 1.0, np.sin(pi_x) / pi_x)
        pi_xa = np.pi * x / order
        sinc_xa = np.where(np.abs(x) < 1e-10, 1.0, np.sin(pi_xa) / pi_xa)
    kernel = sinc_x * sinc_xa
    kernel[np.abs(x) >= order] = 0.0
    return kernel


def _apply_lanczos_shift(
    trace: np.ndarray,
    tau_samples: float,
    order: int = 5,
) -> np.ndarray:
    """Shift a 1-D trace by ``tau_samples`` using Lanczos interpolation.

    Implements y[n] = x[n − τ] by convolving with a Lanczos kernel of the
    given order.  The kernel weights are normalised so the total gain is
    exactly 1.0 for any fractional shift.

    Boundary handling: samples that fall outside the original trace contribute
    zero (zero-padding).  The output length equals the input length.

    Parameters
    ----------
    trace:
        Input signal (1-D array).
    tau_samples:
        Shift in samples.  **Positive values delay** the trace (event moves
        later); negative values advance it.
    order:
        Lanczos kernel order (half-width in samples).  Typical values: 3–8.
        Higher order reduces spectral leakage at the cost of more computation.

    Returns
    -------
    np.ndarray
        Shifted trace of the same length, dtype ``float64``.
    """
    n = len(trace)
    if abs(tau_samples) < 1e-10:
        return trace.astype(np.float64).copy()

    trace_f = trace.astype(np.float64)
    result = np.zeros(n, dtype=np.float64)

    tau_floor = int(np.floor(tau_samples))
    tau_frac = tau_samples - tau_floor  # in [0, 1)

    # Kernel support: 2*order taps centred at the fractional shift
    offsets = np.arange(-order + 1, order + 1, dtype=np.float64)
    weights = _lanczos_kernel_values(offsets - tau_frac, order)
    w_sum = weights.sum()
    if abs(w_sum) > 1e-10:
        weights /= w_sum  # normalise to unit gain

    for k_int, w in zip(offsets.astype(int), weights):
        if abs(w) < 1e-12:
            continue
        shift = tau_floor + k_int  # total integer displacement for this tap
        # y[i] += w * x[i - shift] for valid i
        # dst range: max(0, shift) … min(n, n+shift)
        # src range: max(0,-shift) … min(n, n-shift)
        dst_lo = max(0, shift)
        dst_hi = min(n, n + shift)
        src_lo = max(0, -shift)
        src_hi = min(n, n - shift)
        if dst_lo < dst_hi and src_lo < src_hi:
            result[dst_lo:dst_hi] += w * trace_f[src_lo:src_hi]

    return result


# ---------------------------------------------------------------------------
# TimeShiftErrorEffect
# ---------------------------------------------------------------------------


class TimeShiftErrorEffect(SeismogramEffect):
    """Sub-sample time shift via Lanczos interpolation: common offset + per-station Gaussian.

    The total time shift (seconds) applied to a station is the sum of two
    components, mirroring the legacy ``random_shift_distribution`` pattern but
    flipped (uniform common offset + per-station Gaussian):

    1. **Common offset** (array-wide) — a single value drawn once per call from
       ``uniform(-uniform_offset, +uniform_offset)`` and applied identically to
       every station.  Models a constant velocity / source-time bias.  With the
       default ``uniform_offset = 0.0`` this component is always zero.

    2. **Per-station Gaussian** — each station additionally gets an independent
       draw from ``N(0, gaussian_sigma)`` seconds.

    The total per-station shift ``common_offset + station_gaussian`` is converted
    to samples (``* sampling_rate``) and applied via Lanczos interpolation, so
    fractional-sample accuracy is preserved.  Positive shifts delay the trace.

    Nuisance key: ``time_shift_error`` — an **on/off switch**, NOT a probability.
    ``0.0`` (or absent) ⇒ identity (no shift); any non-zero value ⇒ the effect is
    active and the shift magnitude is governed entirely by ``uniform_offset`` and
    ``gaussian_sigma``.  (There is **no** per-station probability gate — this
    replaces the earlier gated model; see the task log for the rationale.)

    Parameters
    ----------
    sampling_rate:
        Samples per second of the synthetic traces.  Required to convert the
        time shift from seconds to samples before Lanczos interpolation.
        Injected automatically from ``SimulationParameters`` by
        ``GeneralSimulatorWrapper`` / the augmentation chain builder and does
        **not** need to appear in the YAML config.
    uniform_offset:
        Half-width (seconds) of the array-wide common-offset uniform
        distribution.  Defaults to ``DEFAULT_UNIFORM_OFFSET`` (0.0 → no common
        offset).  Set via the YAML key ``uniform_offset``.
    gaussian_sigma:
        Standard deviation of the per-station Gaussian time-shift distribution
        in seconds.  Defaults to ``DEFAULT_GAUSSIAN_SIGMA`` (1.0 s).  Set via
        the YAML key ``gaussian_sigma``.
    lanczos_order:
        Lanczos kernel order.  Higher values are more accurate but slower.
        Defaults to ``DEFAULT_LANCZOS_ORDER`` (5).  Set via YAML key
        ``lanczos_order`` if needed.

    YAML example
    ------------
    .. code-block:: yaml

        parameters:
          nuisance:
            time_shift_error:
              fiducial: [1.0]        # >0 ⇒ active (0.0 ⇒ identity)
              bounds:   [0.0, 1.0]
              uniform_offset: 1.5    # seconds; array-wide common shift
              gaussian_sigma: 2.0    # std dev in seconds; omit to use default 1.0

    Note
    ----
    This effect is stochastic; use ``numpy.random.seed`` in tests that require
    reproducibility.
    """

    #: Default half-width of the array-wide common-offset uniform distribution (s).
    DEFAULT_UNIFORM_OFFSET: float = 0.0
    #: Default Gaussian standard deviation (seconds).
    DEFAULT_GAUSSIAN_SIGMA: float = 1.0
    #: Default Lanczos kernel order.
    DEFAULT_LANCZOS_ORDER: int = 5

    def __init__(
        self,
        sampling_rate: float,
        uniform_offset: Optional[float] = None,
        gaussian_sigma: Optional[float] = None,
        lanczos_order: Optional[int] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
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

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        time_shift_error: Optional[float] = None,
        **_ignored,
    ) -> dict:
        # On/off switch: absent or 0.0 ⇒ identity (back-compat).
        if time_shift_error is None or float(time_shift_error) == 0.0:
            return seismograms_map

        # One array-wide common offset for this call.
        common_offset_s = (
            np.random.uniform(-self._uniform_offset, self._uniform_offset)
            if self._uniform_offset > 0.0
            else 0.0
        )
        result = {}
        for station, components in seismograms_map.items():
            station_shift_s = common_offset_s + np.random.normal(0.0, self._sigma)
            shift_samples = station_shift_s * self._sampling_rate
            result[station] = {
                comp: _apply_lanczos_shift(trace, shift_samples, self._order)
                for comp, trace in components.items()
            }
        return result


# ---------------------------------------------------------------------------
# ScatteringCodaEffect helpers and implementation
# ---------------------------------------------------------------------------


#: Default coda-tail length as a fraction of the trace length (at ``alpha = 1``).
DEFAULT_CODA_FRACTION: float = 0.25


def _apply_stahler_phase_filter(
    trace: np.ndarray,
    alpha: float,
    coda_fraction: float = DEFAULT_CODA_FRACTION,
) -> np.ndarray:
    """Convolve trace with the Stähler & Sigloch (2016) modelling-error filter.

    Implements ``u_me = u_i^c ∗ T_error,i`` (Eq. 17): the synthetic is *convolved*
    with a modelling-error transfer function ``T_error,i`` that has a **unit
    amplitude spectrum** and a **random phase spectrum in ``[0, α·π/2]``**.  The
    transfer function is built explicitly (not by multiplying the synthetic's own
    spectrum), so the operation is a genuine convolution with a compact filter
    rather than a circular spectral product.

    Construction of ``T_error,i``
    -----------------------------
    1. Take a **compact** filter of length ``L = round(coda_fraction · n)`` — this
       is the support of the coda the filter can add.  A compact filter is what
       keeps the effect *local*: stretches of the synthetic that are zero and lie
       further than ``L`` samples from any arrival stay zero, matching the paper
       (its Fig. 2 coda decays and quiet windows are unaffected).
    2. Draw a random phase ``φ[k] ~ U(0, α·π/2)`` on the filter's rfft bins; set a
       unit amplitude, ``H[k] = exp(i·φ[k])``.
    3. ``irfft`` → a real, length-``L`` FIR filter ``h`` with ``|rfft(h)| ≡ 1``
       (all-pass / unit amplitude, exactly as specified).
    4. Convolve the trace with ``h`` **linearly** and truncate to the input
       length.  Linear convolution with a causal FIR cannot wrap energy to the
       start of the window and cannot place energy *before* an arrival.

    α regulates the perturbing effect: ``α = 0`` → ``φ ≡ 0`` → ``h = δ`` → identity;
    larger α → stronger phase scrambling → more coda.

    On phase pinning
    ----------------
    DC (``φ[0]``) and — for even ``L`` — Nyquist (``φ[-1]``) are pinned to zero.
    This is required, not incidental: ``irfft`` discards the imaginary part of the
    DC and Nyquist bins, so a non-zero phase there would pull ``|H|`` below unity
    at those bins and break the unit-amplitude property (and bias the filter gain).
    Pinning them keeps ``|H| ≡ 1`` and yields a real filter.

    For broadband signals the unit-amplitude filter conserves energy; narrowband
    inputs see band-dependent gain (the filter response is only flat on its own
    ``L``-point grid).
    """
    n = len(trace)
    trace_f = trace.astype(np.float64)
    if alpha <= 0.0:
        return trace_f.copy()

    coda_len = max(2, int(round(coda_fraction * n)))
    n_bins = coda_len // 2 + 1
    phi = np.random.uniform(0.0, alpha * np.pi / 2.0, size=n_bins)
    phi[0] = 0.0  # DC must stay real → keeps |H| = 1 at DC
    if coda_len % 2 == 0:
        phi[-1] = 0.0  # Nyquist must stay real → keeps |H| = 1 at Nyquist
    transfer_function = np.fft.irfft(np.exp(1j * phi), n=coda_len)

    return np.convolve(trace_f, transfer_function)[:n]


def _apply_random_coda_filter(
    trace: np.ndarray,
    alpha: float,
    max_coda_fraction: float = DEFAULT_CODA_FRACTION,
) -> np.ndarray:
    """Convolve trace with a causal, energy-conserving random coda kernel.

    Models the waveform modelling error of Stähler & Sigloch (2016), §2.3–2.4:
    scattering adds oscillatory coda that *trails* the direct arrival.  The
    kernel is

        h = [1, alpha·r₁·e^{-1/τ}, alpha·r₂·e^{-2/τ}, …]   (then L2-normalised)

    i.e. a unit spike at lag 0 followed by an exponentially decaying tail of
    i.i.d. random taps ``rₖ ~ U(-1, 1)``.  It is applied by **linear**
    convolution and truncated to the input length.

    Why not the spectral all-pass of the original plan
    --------------------------------------------------
    A circular FFT all-pass filter has two unavoidable, unphysical artifacts:

    1. **Wrap-around** — coda from energy near the right edge of the window
       wraps back to the start (circular convolution).
    2. **Bulk time shift** — any non-trivial *causal* all-pass filter has a
       strictly positive average group delay, so the whole trace (including the
       direct arrival) is delayed by an ``alpha``-dependent amount.

    A causal time-domain kernel with a unit spike at lag 0 avoids both: linear
    convolution + truncation cannot wrap (artifact 1), and the lag-0 spike pins
    the direct-arrival onset in place so only the tail is added (artifact 2).
    The amplitude spectrum is no longer flat — but exact all-pass, strict
    causality, and zero bulk delay are mutually exclusive (only the identity
    satisfies all three), and physical scattering does ripple the spectrum.
    The L2-normalised kernel conserves total energy on average.

    Parameters
    ----------
    trace:
        Input signal (1-D array).
    alpha:
        Coda strength in ``[0, 1]``.  Scales both the tail amplitude (relative
        to the unit direct spike) and the tail length.  ``alpha <= 0`` is the
        identity.
    max_coda_fraction:
        Tail length at ``alpha = 1`` as a fraction of ``len(trace)``.  The
        actual tail length is ``round(alpha · max_coda_fraction · n)``.

    Returns
    -------
    np.ndarray
        Filtered trace of the same length, dtype ``float64``.
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
    kernel[0] = 1.0  # unit spike at lag 0 → direct arrival onset preserved
    kernel /= np.linalg.norm(kernel)  # ~unit gain (energy conserved on average)

    return np.convolve(trace_f, kernel)[:n]


class ScatteringCodaEffect(SeismogramEffect):
    """Per-station stochastic coda filter (waveform modelling error).

    Models the waveform modelling error T_model,i from Stähler & Sigloch
    (2016), §2.3–2.4: scattering / unmodelled 3-D structure adds oscillatory
    coda that *trails* the direct arrivals.  Each selected station's component
    traces are convolved with a causal random coda kernel
    (:func:`_apply_random_coda_filter`).

    Two-stage model applied independently per station:

    1. **Probability gate** — station is perturbed with probability
       ``scattering_coda`` (a value in ``[0, 1]``).
       ``scattering_coda = 0.0`` → no station is ever perturbed (identity).

    2. **Coda perturbation** — if selected, each component trace is filtered
       according to ``mode``:

       - ``'causal'`` (default) — convolution with a causal kernel: a unit spike
         at lag 0 (so the direct-arrival onset is unchanged — no bulk time shift)
         followed by an exponentially decaying random tail.  Linear convolution
         truncated to the window means the coda can never wrap back to the start.
       - ``'stahler'`` — convolution with the Stähler & Sigloch (2016) modelling-
         error transfer function: a compact, unit-amplitude, random-phase FIR
         filter (Eq. 17, ``u_me = u ∗ T_error``).  See
         :func:`_apply_stahler_phase_filter`.

    Both modes are causal and leave quiet (zero) stretches of the synthetic
    untouched; they differ in the coda envelope (``'stahler'``: unit-amplitude
    all-pass filter; ``'causal'``: explicit spike + exponentially decaying tail).

    Nuisance key: ``scattering_coda`` — a probability in ``[0, 1]``.

    Coda strength ``alpha``
    -----------------------
    The coda strength is **drawn independently per station** from
    ``uniform(alpha_range[0], alpha_range[1])`` (default ``(0.0, 1.0)``), so each
    station gets its own scattering strength every realisation.  All components of
    a given station share that station's ``alpha`` draw.  ``alpha`` controls the
    coda: in ``'causal'`` mode it scales the tail amplitude (relative to the unit
    direct spike) and tail length; in ``'stahler'`` mode it is the upper bound of
    the per-bin random phase.  ``alpha = 0`` is the identity.

    Parameters
    ----------
    alpha_range:
        ``(low, high)`` bounds of the per-station uniform ``alpha`` distribution.
        Defaults to :data:`DEFAULT_ALPHA_RANGE` = ``(0.0, 1.0)``.  Set via the YAML
        key ``alpha_range`` inside the ``scattering_coda`` nuisance block.
    alpha:
        Optional **fixed** coda strength.  If given, every station uses exactly this
        value (equivalent to ``alpha_range=(alpha, alpha)``) — back-compatible with
        the previous deterministic behaviour.  Mutually exclusive with ``alpha_range``.
    mode:
        ``'causal'`` (default, physical) or ``'stahler'`` (paper-exact).
    coda_fraction:
        Coda length as a fraction of the trace length (``'causal'``: tail length
        at ``alpha = 1``; ``'stahler'``: transfer-function FIR length).  Defaults
        to :data:`DEFAULT_CODA_FRACTION`.

    Note
    ----
    This effect is stochastic.  Use ``numpy.random.seed`` in tests that
    require reproducibility.
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
    ) -> None:
        if alpha is not None and alpha_range is not None:
            raise ValueError(
                "ScatteringCodaEffect: specify only one of `alpha` (fixed) or "
                "`alpha_range` (per-station sampled)."
            )
        if alpha is not None:
            # Fixed alpha: degenerate range so every station gets exactly `alpha`.
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

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        scattering_coda: Optional[float] = None,
        **_ignored,
    ) -> dict:
        if scattering_coda is None:
            return seismograms_map

        def _coda(components):
            # One alpha per station, shared across its components.
            alpha = np.random.uniform(self._alpha_low, self._alpha_high)
            return {comp: self._filter_trace(trace, alpha) for comp, trace in components.items()}

        return _apply_per_station_gated(seismograms_map, scattering_coda, _coda)


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

#: Maps nuisance parameter key → effect class.  Register new effects here.
EFFECT_REGISTRY: dict[str, type[SeismogramEffect]] = {
    "amplitude_error": AmplitudeErrorEffect,
    "instrument_dropout": InstrumentDropoutEffect,
    "time_shift_error": TimeShiftErrorEffect,
    "scattering_coda": ScatteringCodaEffect,
    "component_dropout": ComponentDropoutEffect,
}


#: Category-2 (post-processing) nuisance keys eligible for **pre-noise** training-time
#: augmentation (folded into the clean signal before sensor noise is added).
AUGMENTABLE_EFFECT_KEYS: tuple[str, ...] = (
    "amplitude_error",
    "instrument_dropout",
    "time_shift_error",
    "scattering_coda",
)


#: Nuisance keys eligible for **post-noise** training-time augmentation (applied to the
#: noisy data ``x = D + noise``).  ``component_dropout`` zeros present channels and must run
#: after noise so a dropped channel is exactly zero (matching a genuinely-absent channel);
#: a pre-noise zeroing would leave ``0 + noise`` instead.
POST_NOISE_EFFECT_KEYS: tuple[str, ...] = (
    "component_dropout",
)


#: Maps an augmentation stage value → the effect keys eligible at that stage.
_STAGE_EFFECT_KEYS: dict[str, tuple[str, ...]] = {
    "training_augmentation": AUGMENTABLE_EFFECT_KEYS,
    "training_augmentation_post_noise": POST_NOISE_EFFECT_KEYS,
}


# ---------------------------------------------------------------------------
# Map <-> stacked-array adapter (lets the SAME effects run in the dataloader)
# ---------------------------------------------------------------------------


def _array_to_map(D: np.ndarray, receivers, components):
    """Convert a stacked ``(n_stations, n_components, T)`` array → seismograms map.

    Mirrors the station/component ordering of
    :meth:`SimulationDataLoader.convert_sim_data_to_array` (station order from
    ``receivers.iterate()``, component order from the loader ``components`` string,
    with zero-filled unused components carried verbatim).

    Returns ``(seismograms_map, station_names)`` where ``seismograms_map`` is
    ``{station_name: {component: np.ndarray}}``.
    """
    station_names = [rec.station_name for rec in receivers.iterate()]
    seismograms_map = {
        station: {comp: D[i, j] for j, comp in enumerate(components)}
        for i, station in enumerate(station_names)
    }
    return seismograms_map, station_names


def _map_to_array(seismograms_map: dict, station_names, components, trace_length: int) -> np.ndarray:
    """Inverse of :func:`_array_to_map` — stack back to ``(n_stations, n_components, T)``."""
    D = np.zeros((len(station_names), len(components), trace_length), dtype=np.float64)
    for i, station in enumerate(station_names):
        comps = seismograms_map[station]
        for j, comp in enumerate(components):
            D[i, j] = comps[comp]
    return D


def apply_chain_to_array(
    chain: PostProcessingChain,
    D: np.ndarray,
    receivers,
    components,
    nuisance_params: dict,
) -> np.ndarray:
    """Apply a :class:`PostProcessingChain` to a stacked data array.

    Bridges the dict-based :class:`SeismogramEffect` API (used per-simulation) to
    the stacked ``(n_stations, n_components, T)`` array produced by the dataloader,
    so the SAME effect classes can be reused as training-time augmentation with no
    duplicated shift/amplitude/dropout logic.

    An empty chain returns ``D`` unchanged (lossless round-trip).

    Parameters
    ----------
    chain:
        The :class:`PostProcessingChain` of training-augmentation effects.
    D:
        Stacked data array, shape ``(n_stations, n_components, T)`` — the same
        layout as ``convert_sim_data_to_array(..., stacked=True, fill_unused=True)``.
    receivers:
        ``Receivers`` instance (provides station ordering, matching ``D``).
    components:
        The loader ``components`` string/sequence (provides the component axis
        ordering, matching ``D``'s second axis).
    nuisance_params:
        Nuisance parameter dict forwarded to each effect.

    Returns
    -------
    np.ndarray
        Augmented array of the same shape as ``D`` (dtype ``float64``).
    """
    if not chain.effects:
        return D
    D = np.asarray(D)
    # Fail loudly if the array layout does not match the receiver/component ordering
    # this adapter assumes (axis 0 = receivers.iterate(), axis 1 = `components`).
    # A silent mismatch would scramble stations/components instead of erroring.
    n_stations = len(list(receivers.iterate()))
    if D.ndim != 3 or D.shape[0] != n_stations or D.shape[1] != len(components):
        raise ValueError(
            f"apply_chain_to_array: D shape {D.shape} is incompatible with "
            f"{n_stations} receivers x {len(components)} components "
            f"(expected ({n_stations}, {len(components)}, T))."
        )
    trace_length = D.shape[-1]
    seismograms_map, station_names = _array_to_map(D, receivers, components)
    processed = chain(seismograms_map, receivers, nuisance_params)
    return _map_to_array(processed, station_names, components, trace_length)


def _fiducial_scalar(value) -> float:
    """Extract the activation scalar from a nuisance fiducial entry (e.g. ``[0.3]`` → 0.3)."""
    arr = np.ravel(value)
    return float(arr[0])


def build_augmentation_chain(
    nuisance: dict,
    nuisance_stage: dict,
    effect_configs: Optional[dict] = None,
    sampling_rate: Optional[float] = None,
    stage: str = "training_augmentation",
) -> Tuple[PostProcessingChain, dict]:
    """Build the training-time augmentation chain + its nuisance-param dict.

    Selects only the nuisance keys staged ``training_augmentation`` that are
    Category-2 post-processing effects, builds a :class:`PostProcessingChain` from
    them (injecting ``sampling_rate`` for ``time_shift_error``), and returns
    ``(chain, nuisance_params)`` where ``nuisance_params`` maps each augmented key
    to its **configured fiducial value** — i.e. the per-station probability for
    ``amplitude_error`` / ``instrument_dropout`` / ``scattering_coda`` (a fiducial of
    ``[0.3]`` ⇒ 30% per-station probability), or the on/off switch for
    ``time_shift_error``.  The effects draw their own random offsets/scales/decisions
    internally per call; the shift/coda *magnitudes* live in ``effect_configs``
    (``uniform_offset``, ``gaussian_sigma``, ``alpha``, ...), not here.

    An empty selection yields ``(PostProcessingChain([]), {})`` — a no-op that the
    dataloader treats as "no augmentation" (back-compat).

    Parameters
    ----------
    nuisance:
        ``{key: fiducial_values}`` (``ModelParameters.nuisance``).  The fiducial
        scalar of each augmented key becomes its activation probability/switch.
    nuisance_stage:
        ``{key: "simulation" | "training_augmentation" | "training_augmentation_post_noise"}``
        (``ModelParameters.nuisance_stage``).  Keys absent from this map default to
        ``"simulation"`` (not augmented).
    effect_configs:
        Optional ``{key: {ctor kwarg: value}}`` (``ModelParameters.nuisance_effect_config``).
    sampling_rate:
        Synthetic sampling rate (samples/s), injected into ``time_shift_error``'s
        effect config so the Lanczos shift can convert seconds → samples.
    stage:
        Which augmentation stage to build for — ``"training_augmentation"`` (pre-noise,
        default) or ``"training_augmentation_post_noise"`` (applied to the noisy data, e.g.
        ``component_dropout``).  Only keys eligible for that stage (see
        :data:`_STAGE_EFFECT_KEYS`) and staged accordingly are included.
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
    # Activation scalar for each effect comes from its configured fiducial value
    # (per-station probability, or on/off switch for time_shift_error) — NOT a
    # hardcoded 1.0, which would force every station to be dropped/perturbed.
    nuisance_params = {key: _fiducial_scalar(nuisance[key]) for key in aug_keys}
    return chain, nuisance_params


def build_augmentation_chain_from_parameters(parameters, sampling_rate=None,
                                             stage="training_augmentation"):
    """Convenience wrapper: build the augmentation chain straight from a ModelParameters.

    Unpacks ``nuisance`` / ``nuisance_stage`` / ``nuisance_effect_config`` from a parsed
    :class:`ModelParameters` and forwards to :func:`build_augmentation_chain`.  Used by the
    training (`train_NPE`) and evaluation (`_eval_inference`) entrypoints so the unpacking
    isn't duplicated.  ``stage`` selects the pre-noise (default) or
    ``training_augmentation_post_noise`` chain.  Returns ``(chain, nuisance_params)``.
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
    """Build a :class:`PostProcessingChain` from a collection of nuisance keys.

    Only keys present in :data:`EFFECT_REGISTRY` result in an effect being
    added.  Unknown keys (e.g. ``source_location``) are silently skipped, so
    callers can safely pass the full ``parameters.nuisance.keys()`` list.

    Parameters
    ----------
    nuisance_keys:
        Iterable of nuisance parameter names (strings).
    effect_configs:
        Optional mapping of ``nuisance_key → dict`` of constructor keyword
        arguments forwarded to each effect's ``__init__``.  Missing entries
        default to ``{}``.  For example::

            effect_configs = {
                "amplitude_error": {"scale_range": (0.5, 2.0)},
                "time_shift_error": {"sampling_rate": 4.0, "gaussian_sigma": 2.0},
            }

    Returns
    -------
    PostProcessingChain
        A chain containing exactly the effects whose keys appear in
        ``EFFECT_REGISTRY``.
    """
    configs = effect_configs or {}
    effects = [
        EFFECT_REGISTRY[key](**configs.get(key, {}))
        for key in nuisance_keys
        if key in EFFECT_REGISTRY
    ]
    return PostProcessingChain(effects)
