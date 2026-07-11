"""Per-trace and per-station waveform-fit metrics for data QA.

Pure functions + small frozen dataclasses operating on plain numpy arrays — no
pipeline construction, no file I/O, no plotting. A reference synthetic is compared
against an observed waveform to decide, per station, whether to keep / time-shift /
drop it (see :mod:`seismo_sbi.data_quality.policy`).

Two families of metrics live here:

* the *classical* alignment/amplitude metrics ported verbatim from the Santorini
  ``qa_forward_check.py`` (cross-correlation lag, variance reduction, peak amplitude
  ratio), and
* *posterior-predictive-check*-derived per-station fidelity metrics
  (``correlation_misfit``, ``envelope_misfit``, ``station_reduced_chi2``) whose maths
  mirror :mod:`seismo_sbi.plotting.posterior_predictive_checks` so the two stay
  consistent.

The shift primitive is the canonical
:func:`seismo_sbi.instaseis_simulator.utils.shift_1d_with_padding` (``+lag`` delays
the synthetic) — it is mathematically identical to the old ``np.roll``-and-zero helper
but is the single source of truth used everywhere else in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from obspy.geodetics.base import gps2dist_azimuth

from seismo_sbi.instaseis_simulator.utils import shift_1d_with_padding

# Optional: Hilbert envelope (mirrors the guard in posterior_predictive_checks).
try:
    from scipy.signal import hilbert
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    hilbert = None


@dataclass(frozen=True)
class TraceDescriptor:
    """Metadata for one (station, component) trace; the QA library's notion of a
    receiver, decoupled from the simulator's :class:`Receivers`."""

    station: str
    component: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class TraceMetrics:
    """Waveform-fit metrics for a single observed-vs-synthetic trace."""

    station: str
    component: str
    dist_km: float
    azimuth: float
    vr: float
    aligned_vr: float
    max_xcorr: float
    best_lag_samples: int
    amp_ratio_obs_syn: float
    obs_peak: float
    syn_peak: float

    def to_row(self) -> dict:
        """Flat dict in the legacy CSV column order (for ``*_metrics.csv``)."""
        return dict(
            station=self.station, component=self.component,
            dist_km=self.dist_km, azimuth=self.azimuth,
            vr=self.vr, aligned_vr=self.aligned_vr,
            max_xcorr=self.max_xcorr, best_lag_samples=self.best_lag_samples,
            amp_ratio_obs_syn=self.amp_ratio_obs_syn,
            obs_peak=self.obs_peak, syn_peak=self.syn_peak)


@dataclass(frozen=True)
class SNRMetrics:
    """Pre-event-noise signal-to-noise metrics for one (station, component) trace.

    Built from the pre-event noise std ``sigma`` (= sqrt of the h5 ``/misc`` lag-0
    autocorrelation, i.e. the pre-event ``mean(x^2)``).  All SNRs are RMS-based, so a
    pure-noise trace has ``snr_obs ~ 1`` exactly (the observation *contains* the noise:
    ``E[obs^2] = signal^2 + sigma^2``).  The signal window is the SYNTHETIC's 5-95%
    cumulative-energy interval (noise-free, hence well defined), so far-station coda does
    not dilute the comparison.  See :func:`snr_metrics` and the policy SNR gates.
    """

    station: str
    component: str
    sigma: float            # pre-event noise std
    snr_obs: float          # RMS_W(obs) / sigma   (>= ~1 floor)
    snr_syn: float          # RMS_W(syn) / sigma   (predicted detectability; noise-free)
    snr_sig: float          # sqrt(max(0, snr_obs^2 - 1)) — noise-DEBIASED observed signal SNR
    snr_obs_full: float     # whole-window RMS(obs) / sigma  (for the excess gate)
    snr_syn_full: float     # whole-window RMS(syn) / sigma


def _rms(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def signal_window(syn1d, quantiles=(0.05, 0.95)):
    """``(lo, hi)`` sample indices bracketing the synthetic's ``quantiles`` cumulative
    energy; the FULL window if the synthetic is degenerate (all-zero / no energy)."""
    e = np.cumsum(np.asarray(syn1d, float) ** 2)
    tot = float(e[-1]) if e.size else 0.0
    if tot <= 0:
        return 0, len(syn1d)
    lo = int(np.searchsorted(e, quantiles[0] * tot))
    hi = int(np.searchsorted(e, quantiles[1] * tot))
    if hi <= lo:
        return 0, len(syn1d)
    return lo, min(hi + 1, len(syn1d))


def snr_metrics(obs2d, syn2d, traces, sigma_map, *, quantiles=(0.05, 0.95),
                sigma_floor: float = 0.0, align: bool = True, max_lag: int = 60) -> List["SNRMetrics"]:
    """Per-trace pre-event-noise SNR for a flattened event (``obs2d``/``syn2d`` are
    ``(n_traces, T)`` in the same order as ``traces``).

    ``sigma_map`` maps ``(station, component) -> noise std sigma`` — the CALLER keys it with
    the SAME component names as ``traces`` (so it must apply any E->1 / N->2 renaming itself).
    A trace with a missing / non-finite / <= ``sigma_floor`` sigma is emitted with
    ``sigma=nan`` and all SNRs 0, which the policy routes straight to a dead-channel drop.

    ``align`` (default True) cross-correlates obs vs syn (±``max_lag`` samples) and shifts the
    synthetic onto the observation before windowing. A 1-D fiducial model mis-times arrivals by
    several seconds vs the real Earth, so an UNALIGNED synthetic signal window misses the
    observed signal and understates ``snr_sig`` — making healthy but time-shifted stations look
    dead. Alignment removes that confound (the signal-window SNR is a fit-quality-free amplitude
    measure once aligned).
    """
    obs2d = np.asarray(obs2d, float)
    syn2d = np.asarray(syn2d, float)
    out: List[SNRMetrics] = []
    for i, tr in enumerate(traces):
        sigma = sigma_map.get((tr.station, tr.component))
        if sigma is None or not np.isfinite(sigma) or sigma <= sigma_floor:
            out.append(SNRMetrics(tr.station, tr.component,
                                  float(sigma) if sigma is not None else float("nan"),
                                  0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        o, s = obs2d[i], syn2d[i]
        if align:
            _, lag = align_best_lag(o, s, max_lag)          # +lag delays the synthetic
            s = shift_1d_with_padding(s, lag)
        lo, hi = signal_window(s, quantiles)
        snr_obs = _rms(o[lo:hi]) / sigma
        snr_syn = _rms(s[lo:hi]) / sigma
        snr_sig = float(np.sqrt(max(0.0, snr_obs ** 2 - 1.0)))
        out.append(SNRMetrics(tr.station, tr.component, float(sigma),
                              snr_obs, snr_syn, snr_sig,
                              _rms(o) / sigma, _rms(s) / sigma))
    return out


# --------------------------------------------------------------------------- maths
def align_best_lag(obs: np.ndarray, syn: np.ndarray, max_lag: int) -> tuple:
    """Return ``(max_normalised_xcorr, best_lag)`` aligning ``syn`` to ``obs``.

    A positive lag means the synthetic must be *delayed* by that many samples to
    match the observed. Degenerate (zero-variance) traces return ``(0.0, 0)``.
    """
    o = obs - obs.mean()
    s = syn - syn.mean()
    denom = np.sqrt(np.sum(o ** 2) * np.sum(s ** 2))
    if denom == 0:
        return 0.0, 0
    best_c, best_l = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        c = np.sum(o * shift_1d_with_padding(s, lag)) / denom
        if c > best_c:
            best_c, best_l = c, lag
    return float(best_c), int(best_l)


def variance_reduction(obs: np.ndarray, syn: np.ndarray) -> float:
    """``1 - ||obs - syn||^2 / ||obs||^2``; NaN if the observed is all-zero."""
    denom = np.sum(obs ** 2)
    return float(1.0 - np.sum((obs - syn) ** 2) / denom) if denom > 0 else float("nan")


def aligned_variance_reduction(obs: np.ndarray, syn: np.ndarray, lag: int) -> float:
    """Variance reduction after delaying ``syn`` by ``lag`` samples."""
    return variance_reduction(obs, shift_1d_with_padding(syn, lag))


def peak_amplitude_ratio(obs: np.ndarray, syn: np.ndarray) -> float:
    """``max|obs| / max|syn|`` (``inf`` if the synthetic peak is zero)."""
    oamp, samp = float(np.max(np.abs(obs))), float(np.max(np.abs(syn)))
    return oamp / samp if samp > 0 else float("inf")


def compute_trace_metrics(
    obs2d: np.ndarray,
    syn2d: np.ndarray,
    traces: List[TraceDescriptor],
    src_lat: float,
    src_lon: float,
    max_lag: int,
) -> List[TraceMetrics]:
    """Compute :class:`TraceMetrics` for every trace.

    ``obs2d`` / ``syn2d`` have shape ``(n_traces, trace_length)`` and align row-for-row
    with ``traces``.
    """
    out: List[TraceMetrics] = []
    for i, d in enumerate(traces):
        o, s = obs2d[i], syn2d[i]
        maxc, lag = align_best_lag(o, s, max_lag)
        dist_m, az, _ = gps2dist_azimuth(src_lat, src_lon, d.latitude, d.longitude)
        oamp, samp = float(np.max(np.abs(o))), float(np.max(np.abs(s)))
        out.append(TraceMetrics(
            station=d.station, component=d.component,
            dist_km=dist_m / 1000.0, azimuth=az,
            vr=variance_reduction(o, s),
            aligned_vr=aligned_variance_reduction(o, s, lag),
            max_xcorr=maxc, best_lag_samples=lag,
            amp_ratio_obs_syn=oamp / samp if samp > 0 else float("inf"),
            obs_peak=oamp, syn_peak=samp))
    return out


def traces_from_receivers(receivers) -> List[TraceDescriptor]:
    """Build the per-trace descriptor list from a simulator ``Receivers`` object,
    in the same (receiver, component) order the simulator flattens its output."""
    return [TraceDescriptor(rec.station_name, comp, rec.latitude, rec.longitude)
            for rec in receivers.iterate() for comp in rec.components]


# ------------------------------------------------- PPC-derived per-station fidelity
# These mirror the maths in plotting.posterior_predictive_checks so the QA module and
# the PPC diagnostics agree. They operate on a single station's stacked traces
# (shape ``(n_components, trace_length)``) and reduce to one scalar per station.
def correlation_misfit(obs2d_sta: np.ndarray, syn2d_sta: np.ndarray) -> float:
    """Mean over the station's components of ``1 - Pearson(obs, syn)`` at zero lag.

    Mirrors ``PosteriorPredictiveChecks._metric_corr_misfit`` (per-trace correlation,
    then averaged). Unlike ``max_xcorr`` this is *not* lag-optimised, so it penalises
    timing errors as well as shape errors.
    """
    x = obs2d_sta - obs2d_sta.mean(axis=1, keepdims=True)
    y = syn2d_sta - syn2d_sta.mean(axis=1, keepdims=True)
    xnorm = np.linalg.norm(x, axis=1) + 1e-12
    ynorm = np.linalg.norm(y, axis=1) + 1e-12
    dots = np.sum(x * y, axis=1)
    corr = np.where((xnorm * ynorm) == 0.0, 0.0, dots / (xnorm * ynorm))
    return float(np.mean(1.0 - corr))


def envelope_misfit(obs2d_sta: np.ndarray, syn2d_sta: np.ndarray) -> float:
    """Mean relative L1 difference of the Hilbert envelopes, averaged over components.

    Mirrors ``PosteriorPredictiveChecks._metric_envelope_misfit``. Computed per trace
    (not on the concatenation) to avoid Hilbert edge artefacts at trace boundaries.
    Returns NaN if scipy is unavailable (matching the PPC guard).
    """
    if hilbert is None:
        return float("nan")
    vals = []
    for o, s in zip(obs2d_sta, syn2d_sta):
        env_o = np.abs(hilbert(o))
        env_s = np.abs(hilbert(s))
        vals.append(float(np.mean(np.abs(env_o - env_s) / (env_o + 1e-12))))
    return float(np.mean(vals))


def station_reduced_chi2(
    obs_flat_sta: np.ndarray,
    syn_flat_sta: np.ndarray,
    covariance: Optional[object] = None,
    dof: Optional[int] = None,
) -> float:
    """Reduced chi-square of a station's residual.

    Mirrors ``PosteriorPredictiveChecks._metric_reduced_chi2``: with a covariance
    exposing ``compute_loss(residual, reduce=True)`` the Mahalanobis form
    ``chi2 = -2 * compute_loss`` is used (the covariance must be sized to this
    station's residual); otherwise it falls back to ``r·r``. ``dof`` defaults to the
    residual length.
    """
    r = np.asarray(obs_flat_sta).ravel() - np.asarray(syn_flat_sta).ravel()
    if dof is None:
        dof = r.size if r.size > 0 else 1
    if covariance is None:
        chi2 = float(np.dot(r, r))
    else:
        chi2 = -2.0 * covariance.compute_loss(r, reduce=True)
    return float(chi2 / dof)
