"""Posterior predictive misfit metrics: one score per synthetic against the observation.

Each metric is ``f(obs, synthetics, meta) -> (n_synthetics,)`` for an observation of shape
``(data_len,)`` and synthetics ``(n_synthetics, data_len)``. Lower is a better fit. ``meta`` holds
``sample_rate``, ``n_traces``, ``covariance_matrix``, ``dof_override``, ``max_shift_samples`` and
``autocorr_maxlag``. :class:`~seismo_sbi.plotting.posterior_predictive_checks.PosteriorPredictiveChecks`
registers them by default.
"""

from typing import Optional

import numpy as np
from scipy.signal import hilbert

from seismo_sbi.data_quality.metrics import correlation_misfit, station_reduced_chi2


def reduced_chi2_misfit(obs: np.ndarray, synthetics: np.ndarray, meta: dict):
    """Reduced chi-square of each synthetic under ``meta["covariance_matrix"]``, with ``dof_override``
    degrees of freedom or the data length."""
    n_samples = synthetics.shape[0]
    obs_len = obs.size
    dof = meta.get("dof_override", None)
    if dof is None:
        dof = obs_len if obs_len > 0 else 1

    chi2s = np.empty((n_samples,), dtype=float)
    for i in range(n_samples):
        chi2s[i] = station_reduced_chi2(obs, synthetics[i], meta.get("covariance_matrix"), dof)
    return chi2s


def traces_of(vec: np.ndarray, n_traces: int):
    """``vec`` as ``(n_traces, trace_len)``, or None when ``n_traces`` is unknown or does not divide it."""
    vec = np.asarray(vec).ravel()
    if n_traces is None or n_traces <= 0:
        return None
    if len(vec) % n_traces != 0:
        return None
    tlen = len(vec) // n_traces
    return vec.reshape((n_traces, tlen))


def correlation_misfit_per_sample(obs: np.ndarray, synthetics: np.ndarray, meta: dict):
    """
    Per-sample misfit = mean over traces of (1 - Pearson corr) if trace shape known,
    else whole-vector (1 - corr) per sample.
    """
    n_samples = synthetics.shape[0]
    n_traces = meta.get("n_traces")

    obs_mat = traces_of(obs, n_traces)
    if obs_mat is None:
        x = obs - obs.mean()
        xnorm = np.linalg.norm(x) + 1e-12
        vals = []
        for s in synthetics:
            y = s - s.mean()
            denom = xnorm * (np.linalg.norm(y) + 1e-12)
            corr = 0.0 if denom == 0.0 else float(np.dot(x, y) / denom)
            vals.append(1.0 - corr)
        return np.asarray(vals, dtype=float)

    tlen = obs_mat.shape[1]
    try:
        syn_mat = synthetics.reshape((n_samples, obs_mat.shape[0], tlen))
    except Exception:
        return correlation_misfit_per_sample(obs, synthetics, {**meta, "n_traces": None})

    vals = np.empty((n_samples,), dtype=float)
    for i in range(n_samples):
        vals[i] = correlation_misfit(obs_mat, syn_mat[i])
    return vals


def power_misfit(obs: np.ndarray, synthetics: np.ndarray, meta: dict):
    """``|P_obs - P_syn| / P_obs`` with ``P`` the summed squared amplitude."""
    p_obs = float(np.dot(obs, obs)) + 1e-12
    p_syn = np.sum(synthetics ** 2, axis=1)
    vals = np.abs(p_obs - p_syn) / p_obs
    return vals


def envelope_misfit(obs: np.ndarray, synthetics: np.ndarray, meta: dict):
    """
    Envelope-based misfit: mean absolute relative difference of envelope over time.
    """
    env_obs = np.abs(hilbert(obs))
    vals = np.empty((synthetics.shape[0],), dtype=float)
    for i, s in enumerate(synthetics):
        env_s = np.abs(hilbert(s))
        denom = env_obs + 1e-12
        vals[i] = float(np.mean(np.abs(env_obs - env_s) / denom))
    return vals


def mse(obs: np.ndarray, synthetics: np.ndarray, meta: dict):
    """Mean squared difference of each synthetic from the observation."""
    return np.mean((synthetics - obs) ** 2, axis=1)


def autocorrelation_matrix(mat: np.ndarray, maxlag: int) -> np.ndarray:
    """Normalised autocorrelation at lags 1..maxlag for each row of ``mat``."""
    n = mat.shape[1]
    x = mat - mat.mean(axis=1, keepdims=True)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(x, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, : maxlag + 1]
    return ac[:, 1:] / (ac[:, :1] + 1e-12)


def autocorrelation_misfit(obs: np.ndarray, synthetics: np.ndarray, meta: dict, maxlag: Optional[int] = None):
    """Difference in short-lag autocorrelation SHAPE, per trace.

    ``mean_{lag=1..maxlag} |acf_obs(lag) - acf_syn(lag)|``, averaged over traces, with ``maxlag``
    from ``meta["autocorr_maxlag"]`` (50 samples when absent).

    The autocorrelation of a trace is invariant to a time shift of that trace, so this
    scores pulse shape, duration and frequency content while ignoring arrival time entirely:
    the complement to the zero-lag correlation, which cannot separate the two.

    Computed PER TRACE (falling back to the whole vector only when the trace layout is
    unknown): run over concatenated traces the lag products would straddle trace
    boundaries and measure the packing order rather than the waveforms.

    .. warning::
       Blind to polarity by construction: ``ACF(-x) == ACF(x)``, so an inverted
       waveform scores exactly 0.  Never use this metric alone to argue about the sign
       of the isotropic component. Pair it with ``Shifted corr misfit`` (which keeps
       polarity as long as the lag budget stays under half a dominant period), the
       zero-lag correlation, or the first-motion polarity analysis.
    """
    maxlag = int(meta.get("autocorr_maxlag", 50) if maxlag is None else maxlag)
    n_samples = synthetics.shape[0]
    n_traces = meta.get("n_traces")
    obs_mat = traces_of(np.asarray(obs).ravel(), n_traces)
    if obs_mat is None:
        obs_mat = np.asarray(obs).ravel()[None, :]
        syn_mat = np.asarray(synthetics).reshape(n_samples, 1, -1)
    else:
        try:
            syn_mat = synthetics.reshape((n_samples,) + obs_mat.shape)
        except Exception:  # noqa: BLE001
            obs_mat = np.asarray(obs).ravel()[None, :]
            syn_mat = np.asarray(synthetics).reshape(n_samples, 1, -1)
    maxlag = max(1, min(maxlag, obs_mat.shape[1] - 1))
    acf_obs = autocorrelation_matrix(obs_mat, maxlag)
    vals = np.empty((n_samples,), dtype=float)
    for i in range(n_samples):
        vals[i] = float(np.mean(np.abs(acf_obs - autocorrelation_matrix(syn_mat[i], maxlag))))
    return vals


def shifted_correlation_misfit(obs: np.ndarray, synthetics: np.ndarray, meta: dict):
    """``1 - max_{|lag| <= L} corr(obs, syn)``, per trace, averaged over traces.

    The zero-lag correlation misfit charges a mechanism for travel-time error it did not
    cause: a 1-D velocity model and a catalogue hypocentre both translate whole traces,
    which decorrelates them without saying anything about the source. Maximising over a
    bounded lag lets the arrival time float and scores waveform shape and polarity alone.

    ``L`` = ``meta["max_shift_samples"]`` or 10 % of the trace length. The lag band
    is bounded on purpose: an unbounded search would align a P arrival onto an S.
    """
    n_samples = synthetics.shape[0]
    n_traces = meta.get("n_traces")
    obs_mat = traces_of(np.asarray(obs).ravel(), n_traces)
    if obs_mat is None:
        return correlation_misfit_per_sample(obs, synthetics, meta)
    try:
        syn_mat = synthetics.reshape((n_samples,) + obs_mat.shape)
    except Exception:  # noqa: BLE001
        return correlation_misfit_per_sample(obs, synthetics, meta)

    tlen = obs_mat.shape[1]
    L = meta.get("max_shift_samples")
    L = int(L) if L else max(1, int(round(0.10 * tlen)))
    L = max(1, min(L, tlen - 1))

    nfft = int(2 ** np.ceil(np.log2(2 * tlen)))
    x = obs_mat - obs_mat.mean(axis=1, keepdims=True)
    xn = np.linalg.norm(x, axis=1) + 1e-12
    FX = np.conj(np.fft.rfft(x, n=nfft, axis=1))
    vals = np.empty((n_samples,), dtype=float)
    for i in range(n_samples):
        y = syn_mat[i] - syn_mat[i].mean(axis=1, keepdims=True)
        yn = np.linalg.norm(y, axis=1) + 1e-12
        cc = np.fft.irfft(FX * np.fft.rfft(y, n=nfft, axis=1), n=nfft, axis=1)
        # lag k lives at index k for k >= 0 and at nfft+k for k < 0
        band = np.concatenate([cc[:, : L + 1], cc[:, nfft - L:]], axis=1)
        best = band.max(axis=1) / (xn * yn)
        vals[i] = float(np.mean(1.0 - np.clip(best, -1.0, 1.0)))
    return vals
