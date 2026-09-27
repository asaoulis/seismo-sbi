"""Sub-sample time shifts of seismograms by Lanczos interpolation.

The kernel weights depend only on the shift, so ``_apply_lanczos_shift_batch`` shifts every
row of an array at once; ``_apply_lanczos_shift`` is the one-trace form.
"""
from __future__ import annotations

import numpy as np


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


def _shift_components(components: dict, tau_samples: float, order: int = 5) -> dict:
    """``{component: trace}`` of one station, every trace shifted by the same ``tau_samples``."""
    comps = list(components)
    shifted = _apply_lanczos_shift_batch(np.stack([components[c] for c in comps]), tau_samples, order)
    return {c: shifted[j] for j, c in enumerate(comps)}
