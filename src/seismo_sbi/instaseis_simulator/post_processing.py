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
    """Stochastic amplitude modulation — per-station gated (legacy) or per-trace always-on.

    **Legacy model** (``distribution='uniform'``, ``per_component=False``,
    ``always_on=False`` — the defaults; RNG call order byte-identical to the original):

    1. **Dropout stage** — decide whether to apply modulation at all.
       The probability is ``amplitude_error`` (a value in ``[0, 1]``).
       ``amplitude_error = 0.0`` → no station is ever modulated (identity).
       ``amplitude_error = 1.0`` → every station is modulated.
    2. **Scale stage** — if the station is selected, multiply all its component
       traces by ONE scale factor drawn uniformly from ``[scale_low, scale_high]``.

    **Recalibrated model** (Japan forensics N14 §4 / N17a, 2026-08-25): the measured
    per-trace amplitude error of real regional records against 1-D synthetics is
    log-normal with σ ≈ 0.3 dex, *independent between the components of a station*
    (σ(R−Z) 0.16 dex, σ(T−Z) 0.30 dex) and present on every trace — while the legacy
    model applies one flat-in-frequency factor per station, identical on all
    components, to a Bernoulli-gated minority of stations.  On a frozen network the
    per-trace structure at σ = 0.35 dex reproduces ~37 % of the observed ISO
    displacement and half the posterior-width excess; the per-station structure at
    the same σ reproduces none of it.  Three switches express the measured structure:

    * ``distribution='lognormal'`` — ``g = 10 ** (σ · N(0, 1))`` with
      ``σ = log_sigma_dex`` (default :data:`DEFAULT_LOG_SIGMA_DEX`) instead of
      ``U(scale_low, scale_high)``.
    * ``per_component=True`` — an independent draw per (station, component) trace
      instead of one draw per station.
    * ``always_on=True`` — no Bernoulli gate: every station is modulated, and the
      nuisance value ``amplitude_error`` becomes a **strength multiplier** on σ
      (``0`` → identity, ``1`` → the configured σ, ``2`` → double), the same
      convention as :class:`ScatteringCodaEffect` in distance mode.  With the
      uniform distribution the multiplier only switches the effect on/off.

    Nuisance key: ``amplitude_error`` — a probability in ``[0, 1]`` (legacy) or a
    strength multiplier (``always_on``).  If ``amplitude_error`` is absent from
    ``nuisance_params`` the input map is returned unchanged.

    Parameters
    ----------
    scale_range:
        Optional ``(low, high)`` tuple overriding the default uniform scale range.
        Useful in tests to make the output deterministic (e.g.
        ``scale_range=(2.0, 2.0)`` always applies a factor of exactly 2).
        Defaults to ``(DEFAULT_SCALE_LOW, DEFAULT_SCALE_HIGH)``.
    distribution:
        ``'uniform'`` (default, legacy) or ``'lognormal'``.
    log_sigma_dex:
        Width of the log-normal in dex (``distribution='lognormal'`` only).
    per_component:
        Draw independently per component trace instead of once per station.
    always_on:
        Disable the per-station Bernoulli gate; nuisance value = strength multiplier.

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
    #: Default log-normal width in dex (``distribution='lognormal'``): the measured
    #: per-trace σ on Japan F-net records is 0.31–0.38 dex (N14 §4).
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
        """One scale factor.  Uniform ignores ``multiplier`` (legacy RNG call preserved)."""
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
            # Independent per-station (or per-trace) scale factor
            return self._scale_components(components, 1.0)

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


def _apply_lanczos_shift_batch(
    traces: np.ndarray,
    tau_samples: float,
    order: int = 5,
) -> np.ndarray:
    """Shift every row of a ``(n_traces, T)`` array by the SAME ``tau_samples``.

    Vectorised form of :func:`_apply_lanczos_shift`: the Lanczos kernel weights
    depend only on ``tau_samples`` (not on the trace), so for a set of traces that
    share a shift (e.g. all components of one station, which get one per-station
    shift) the kernel is built **once** and the tap-additions are applied to all
    rows at once.  Each output element is the same weighted sum of the same input
    samples as the per-trace loop, so the result is numerically identical (no
    reassociation across rows).

    Parameters
    ----------
    traces:
        ``(n_traces, T)`` array of input signals sharing one shift.
    tau_samples, order:
        As in :func:`_apply_lanczos_shift`.

    Returns
    -------
    np.ndarray
        ``(n_traces, T)`` shifted traces, dtype ``float64``.
    """
    traces_f = np.asarray(traces, dtype=np.float64)
    n = traces_f.shape[-1]
    if abs(tau_samples) < 1e-10:
        return traces_f.copy()

    result = np.zeros_like(traces_f)

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
        # y[:, i] += w * x[:, i - shift] for valid i
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
    """Shift a 1-D trace by ``tau_samples`` using Lanczos interpolation.

    Implements y[n] = x[n − τ] by convolving with a Lanczos kernel of the
    given order.  The kernel weights are normalised so the total gain is
    exactly 1.0 for any fractional shift.

    Boundary handling: samples that fall outside the original trace contribute
    zero (zero-padding).  The output length equals the input length.

    Thin wrapper over :func:`_apply_lanczos_shift_batch` for a single trace.

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
    return _apply_lanczos_shift_batch(np.asarray(trace)[np.newaxis, :], tau_samples, order)[0]


# ---------------------------------------------------------------------------
# TimeShiftErrorEffect
# ---------------------------------------------------------------------------


class TimeShiftErrorEffect(SeismogramEffect):
    """Sub-sample time shift via Lanczos interpolation: common offset + per-station Gaussian.

    The total time shift (seconds) applied to a station is the sum of two
    components, mirroring the legacy ``random_shift_distribution`` pattern but
    flipped (uniform common offset + per-station Gaussian):

    1. **Common offset** (array-wide) — a single value drawn once per call and
       applied identically to every station.  Models a constant velocity /
       source-time bias.  Its distribution is selected by ``common_offset_dist``:

       * ``"uniform"`` (default, back-compat) — ``uniform(-uniform_offset,
         +uniform_offset)``; with ``uniform_offset = 0.0`` this component is zero.
       * ``"gaussian"`` — ``N(0, common_offset_sigma)``.  Calibrated array-wide
         offsets are peaked at zero (not flat), so a Gaussian fits them better
         than a uniform.

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
    sigma_per_1000km, distance_cap_km:
        Distance scaling of the per-station sigma (seconds per 1000 km, optional cap in km);
        ``0`` (default) keeps the flat legacy sigma.  Requires the source location (nuisance
        dict ``source_location`` at the simulation stage, or ``source_latitude`` /
        ``source_longitude`` in the effect config).
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
        # Distance-scaled per-station sigma (Japan forensics N9/N11/N14, 2026-08-25): the
        # measured per-station timing error of 1-D synthetics grows with path length —
        # σ ≈ 4 s inside 400 km rising to ≈ 15 s beyond 1200 km at 20–30 s — so the
        # per-station Gaussian width is sigma(D) = gaussian_sigma + sigma_per_1000km * min(D, cap)/1000.
        # Inert by default (0 slope → legacy behaviour and RNG order). Needs the source
        # location (nuisance dict ``source_location`` — forwarded at the simulation stage —
        # or the constructor), exactly like ScatteringCodaEffect(distance_mode=True).
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
        # Std of the array-wide common offset when common_offset_dist='gaussian'.
        # Defaults to uniform_offset (so an existing scale carries over if the dist
        # is flipped without specifying a new sigma).
        self._common_sigma = (
            float(common_offset_sigma)
            if common_offset_sigma is not None
            else self._uniform_offset
        )

    def station_sigmas(self, receivers, source_location=None) -> dict:
        """``{station_name: sigma_s}`` — the per-station Gaussian width, distance-scaled if
        ``sigma_per_1000km > 0`` (otherwise the flat ``gaussian_sigma`` for every station)."""
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
        # On/off switch: absent or 0.0 ⇒ identity (back-compat).
        if time_shift_error is None or float(time_shift_error) == 0.0:
            return seismograms_map

        station_sigma = self.station_sigmas(receivers, source_location)

        # One array-wide common offset for this call, from the chosen distribution.
        if self._common_dist == "gaussian":
            common_offset_s = (
                np.random.normal(0.0, self._common_sigma)
                if self._common_sigma > 0.0
                else 0.0
            )
        else:  # uniform (back-compat default)
            common_offset_s = (
                np.random.uniform(-self._uniform_offset, self._uniform_offset)
                if self._uniform_offset > 0.0
                else 0.0
            )
        result = {}
        for station, components in seismograms_map.items():
            # One per-station Normal draw, in map order (RNG order preserved).
            station_shift_s = common_offset_s + np.random.normal(
                0.0, station_sigma.get(station, self._sigma))
            shift_samples = station_shift_s * self._sampling_rate
            comps = list(components)
            if not comps:
                result[station] = {}
                continue
            # All components of a station share this shift, so build the Lanczos
            # kernel once and apply it to the stacked (C, T) traces at once.
            traces = np.stack([components[c] for c in comps])
            shifted = _apply_lanczos_shift_batch(traces, shift_samples, self._order)
            result[station] = {c: shifted[j] for j, c in enumerate(comps)}
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



# ---------------------------------------------------------------------------
# Distance-scaled scattering (far-path decoherence + incoherent coda energy)
# ---------------------------------------------------------------------------

def distance_scaled_alpha(
    dist_km,
    alpha_intercept: float = 0.0,
    alpha_per_1000km: float = 0.0,
    distance_cap_km: Optional[float] = None,
):
    """Coda strength as a function of source–station distance.

    ``alpha(D) = alpha_intercept + alpha_per_1000km * min(D, cap) / 1000``, clipped to
    ``[0, 1]``.  A linear growth with path length is the first-order scattering-theory
    expectation: the scattered (coda) energy fraction accumulates as ``D / l`` with ``l``
    the mean free path (Sato, Fehler & Maeda 2012, ch. 3), so the strength of the
    delayed-replica kernel grows with the path.  The cap lets the far-field saturate.
    Vectorised in ``dist_km``.
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
    """Target coda (tail) energy relative to the direct arrival, from a distance ramp.

    The kernel of :func:`_apply_distance_coda_kernel` is ``[1, tail]`` with
    ``||tail||^2 = E_tail``; for a broadband input the output energy is ``(1 + E_tail)``
    times the input energy, i.e. an RMS excess of ``0.5 log10(1 + E_tail)`` dex.  Inverting
    a linear RMS-dex ramp ``excess_dex_per_1000km * min(D, cap) / 1000`` gives

        E_tail(D) = 10^(2 * excess_dex_per_1000km * min(D, cap) / 1000) - 1 .

    This is the *kernel-level* target; the excess measured in a group-velocity window
    of a real seismogram also depends on the trace's own time structure, so the
    per-1000 km value must be calibrated against the same window diagnostics used on
    the data (not read off this formula).  Vectorised in ``dist_km``.
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
    """Delayed-replica (multipath) coda kernel with prescribed tail energy.

    Same construction as :func:`_apply_random_coda_filter` — a unit spike at lag 0 (direct
    arrival pinned: no bulk time shift) followed by an exponentially decaying tail of
    i.i.d. ``U(-1, 1)`` taps whose length is ``round(alpha * max_coda_fraction * n)`` —
    but the tail is scaled to ``||tail||^2 = tail_energy`` and the kernel is **not**
    re-normalised.  Convolution therefore adds a superposition of delayed, randomly
    weighted replicas of the signal (single-scattering / multipathing picture), which
    both decorrelates the waveform from the unperturbed one (coherence loss growing with
    ``alpha`` and ``tail_energy``) and *adds* incoherent energy behind every arrival —
    the two far-path signatures measured on F-net data (cross-correlation with the 1-D
    synthetic falling with distance while the surface-wave-window energy exceeds the
    prediction).  ``tail_energy <= 0`` or ``alpha <= 0`` is the identity.

    RNG: exactly one ``np.random.uniform`` call of size ``coda_len + 1`` (as the legacy
    kernel), so per-station seeding behaves identically.
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
    kernel[0] = 1.0  # unit spike at lag 0 -> direct arrival preserved, energy ADDED by the tail
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

    Distance mode (``distance_mode=True``) — far-path scattering emulation
    ---------------------------------------------------------------------
    The legacy model above is per-station i.i.d. and **distance-blind**: 60 % of stations
    are untouched whatever their path length and the strength never grows with distance.
    On F-net regional data (paths 100–1500 km, 10–50 s) the 1-D Green's functions
    decohere with path length (best cross-correlation 0.86 at < 200 km falling to ~0.4
    beyond 1200 km) while the surface-wave-window energy exceeds the 1-D prediction by a
    frequency-independent ~+0.25 dex per 1000 km — a scattering/3-D-path regime the
    legacy operator cannot express.  Distance mode replaces the Bernoulli gate and the
    uniform ``alpha`` draw with a **deterministic distance ramp**, applied to every
    station:

        alpha_s      = clip(m * alpha(D_s) * (1 + U(-alpha_jitter, +alpha_jitter)), 0, 1)
        alpha(D)     = alpha_intercept + alpha_per_1000km * min(D, cap) / 1000
        E_tail(D_s)  = 10^(2 * m * excess_dex_per_1000km * min(D, cap) / 1000) - 1

    with ``D_s`` the source–station distance (km) and ``m`` the nuisance value
    ``scattering_coda``, which in distance mode is a **strength multiplier** (0 or absent
    → identity, 1 → the configured ramp, 2 → double), not a gate probability — the same
    convention as :class:`AzimuthalAnisotropyEffect`.  ``mode='causal'`` uses
    :func:`_apply_distance_coda_kernel` (delayed-replica kernel with tail energy
    ``E_tail``: decoherence *and* added incoherent energy); ``mode='stahler'`` uses the
    unit-amplitude phase filter at ``alpha_s`` (decoherence only, energy conserved —
    ``excess_dex_per_1000km`` is then ignored).  The path-length scaling follows the
    scattering-theory expectation that coda energy accumulates as ``D / l`` (mean free
    path ``l``; Sato, Fehler & Maeda 2012), on top of the Stähler & Sigloch (2016)
    modelling-error operator.  All ramp defaults are **inert** (0 slope, 0 excess): the
    calibrated values are config, not code, and are set from the data-side coherence and
    window-energy curves (see ``station_scattering_params`` for provenance).

    The source location is taken from the nuisance dict (``source_location``, first two
    entries = lat, lon — forwarded by ``Simulator.run_simulation``) or from the
    constructor; if distance mode is active and neither is available the effect raises.
    ``training_augmentation`` does not yet forward the source location, so distance mode
    is a simulation-stage / injection-harness feature for now.

    Note
    ----
    This effect is stochastic.  Use ``numpy.random.seed`` in tests that
    require reproducibility.  With ``distance_mode=False`` (default) the behaviour and
    the RNG call order are exactly those of the legacy model.
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

    # ------------------------------------------------------------------ distance mode
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
        """``{station_name: (dist_km, alpha_nominal, tail_energy)}`` for provenance.

        ``alpha_nominal`` is the ramp value *before* the per-station jitter draw.
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
            # RNG order per station: one jitter draw, then one tail draw per component.
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
            # One alpha per station, shared across its components.
            alpha = np.random.uniform(self._alpha_low, self._alpha_high)
            return {comp: self._filter_trace(trace, alpha) for comp, trace in components.items()}

        return _apply_per_station_gated(seismograms_map, scattering_coda, _coda)


# ---------------------------------------------------------------------------
# Anisotropy injection effects
# ---------------------------------------------------------------------------


def _bearing_and_distance_km(src_lat, src_lon, sta_lat, sta_lon):
    """Source→station azimuth (deg, clockwise from north) and distance (km).

    Spherical-Earth formulas (R = 6371 km); numpy-only so the effect stays
    dependency-free.  Accuracy is far beyond what a cos 2φ anomaly needs.
    """
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
    """Coherent azimuth-dependent travel-time anomaly (weak-anisotropy cos 2φ).

    Models the P-delay signature of weak azimuthal anisotropy (Backus 1965
    parameterisation, as used operationally by Silver & Chan 1991): the
    velocity varies as ``V(φ) = V₀(1 + A cos 2(φ − φ_fast))``, so a path of
    length ``D`` at source→station azimuth ``φ`` accumulates a delay

        Δt(φ) = −(D / V₀) · A · cos(2(φ − φ_fast))

    (fast azimuth ⇒ early arrival ⇒ negative delay).  The shift grows with
    path length and is **coherent across the array by construction** — this is
    the causal variable the injection experiment isolates, in contrast to the
    random per-station shifts of :class:`TimeShiftErrorEffect`.

    All components of a station share the shift (applied via Lanczos, as in
    :class:`TimeShiftErrorEffect`).  Deterministic — no RNG.

    Nuisance key: ``azimuthal_anisotropy`` — 0.0/absent ⇒ identity; otherwise
    the value is a **strength multiplier** on ``aniso_fraction`` (so arms ×1,
    ×5, ×10 are driven through the nuisance value with fixed configs).

    The source location is required to compute azimuths.  It is taken from the
    nuisance dict (``source_location = (lat, lon)``-like) if present, else from
    the constructor.  If the effect is ACTIVE and no source location is
    available it raises rather than silently no-oping.
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
        """``{station_name: delay_seconds}`` at multiplier 1 (for provenance)."""
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
    """Shear-wave splitting via the Silver & Chan (1991) operator.

    Station-side, geographic frame, horizontals only: rotate (N, E) into the
    (fast, slow) frame defined by the fast-polarisation azimuth ``φ_fast``,
    delay the SLOW trace by ``δt`` (Lanczos), rotate back.  Z is untouched.
    Deterministic — no RNG.  Stations missing either horizontal are passed
    through unchanged.

    Nuisance key: ``shear_wave_splitting`` — 0.0/absent ⇒ identity; otherwise
    a **multiplier** on ``delay_s`` (arms ×1/×5/×10).

    Operator form confirmed against Silver & Chan (1991) eqs. 1–13 (see the
    anisotropy-robustness task, artifacts/lit/03).
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


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

#: Maps nuisance parameter key → effect class.  Register new effects here.

class DispersionSpreadEffect(SeismogramEffect):
    """Frequency-dependent travel-time (dispersion) error — a pure phase delay per station.

    Ported from the validated N11 injection operator (Japan forensics, 2026-08-24/25):

        u'(f) = u(f) · exp(−2πi f τ(f))

    with τ(f) interpolated in log-period between octave centres ``octave_centres_s`` and held
    constant outside them.  Per station the octave delays are

        τ_o = m · z_o · σ_o(D),   σ_o(D) = sigma_intercept_s[o] + sigma_per_1000km_s[o] · min(D, cap) / 1000

    where ``m`` is the nuisance value (strength multiplier; ``0`` → identity) and the ``z_o`` are
    standard normals.  Their correlation structure is the physics: a too-fast (or too-slow)
    crust delays *every* octave of a path in the same sense, so ``z_o`` is by default ONE draw
    per station shared across octaves (``octave_correlation=1``); ``octave_correlation=0``
    reproduces N11's independent-per-octave form, and intermediate values mix the two.
    ``common_fraction`` puts that share of the variance into ONE array-wide draw (the coherent,
    same-sign far-station delay the measured data contain and per-station-independent sampling
    can never produce).

    The measured Japan calibration (N11 §2b, F-net reference, 61-member CPS spread): ensemble
    phase-velocity σ_t ≈ 4.7 / 3.7 / 2.2 / 0.8 s inside 400 km and 23.5 / 18.3 / 10.9 / 4.0 s beyond
    1200 km at 12.5 / 17.5 / 25 / 40 s — i.e. ``sigma_intercept_s ≈ [0, 0, 0, 0]`` and
    ``sigma_per_1000km_s ≈ [19, 15, 9, 3]``.  The measured *bias* (+14–16 s at 10–15 s beyond
    800 km) is NOT part of this effect: a bias belongs in the fiducial model, not in a nuisance.

    Nuisance key: ``dispersion_spread`` — strength multiplier.  Simulation stage only (needs the
    source location, forwarded by ``Simulator.run_simulation``, or ``source_latitude`` /
    ``source_longitude`` in the effect config).  ``sampling_rate`` is injected automatically.
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
        """``{station_name: sigma_per_octave (array)}`` in seconds."""
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
        """Interpolate per-octave delays (s) onto a frequency axis, log-period, clamped."""
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
        # ONE array-wide draw (shared by every station), then per-station draws.
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


#: Nuisance keys that augment the **source-location CONDITIONING vector** in the ML dataloader
#: (NOT the waveform).  These are deliberately *absent* from :data:`EFFECT_REGISTRY` — they have
#: no ``SeismogramEffect`` and are skipped by :func:`build_post_processing_chain`; the dataloader
#: applies them to ``source_vec`` instead.  ``source_location_error`` perturbs the conditioned
#: source location with per-coordinate Gaussian noise so the model learns to tolerate the
#: catalogue location error it sees at inference.  Stage-eligible like ``training_augmentation``.
CONDITIONING_AUGMENTABLE_KEYS: tuple[str, ...] = (
    "source_location_error",
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


def _map_to_array(seismograms_map: dict, station_names, components) -> np.ndarray:
    """Inverse of :func:`_array_to_map` — stack back to ``(n_stations, n_components, T)``.

    Builds the array in one allocation from the nested map rather than pre-zeroing
    and copying cell by cell; the shape follows from the traces.
    """
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
    seismograms_map, station_names = _array_to_map(D, receivers, components)
    processed = chain(seismograms_map, receivers, nuisance_params)
    return _map_to_array(processed, station_names, components)


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
    training (`train_NPE`) and evaluation (`seismo_sbi.evaluation`) entrypoints so the
    unpacking isn't duplicated.  ``stage`` selects the pre-noise (default) or
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
