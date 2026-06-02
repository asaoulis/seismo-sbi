"""Gutenberg-Richter magnitude statistics for the moment-tensor prior.

Provides:
  * ``estimate_mc_maxcurvature`` - maximum-curvature completeness-magnitude estimate.
  * ``fit_b_value_aki``          - Aki (1965) maximum-likelihood b-value.
  * ``magnitude_to_m0``          - Hanks-Kanamori M -> scalar moment M0 (N.m).
  * ``GutenbergRichterModel``    - a truncated Gutenberg-Richter (truncated
                                   exponential) magnitude distribution with
                                   inverse-CDF sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: ln(10), the conversion factor between the GR b-value and the exponential rate.
_LN10 = np.log(10.0)


def magnitude_to_m0(mw):
    """Hanks-Kanamori scalar moment ``M0 = 10**(1.5*Mw + 9.1)`` (N.m).

    Inverse of the moment-magnitude relation used throughout the library
    (e.g. ``plotting/velocity_models.py``). Vectorised.
    """
    return 10.0 ** (1.5 * np.asarray(mw, dtype=float) + 9.1)


def estimate_mc_maxcurvature(magnitudes, *, delta_m: float = 0.1) -> float:
    """Maximum-curvature estimate of the completeness magnitude ``Mc``.

    Bins the magnitudes and returns the magnitude bin with the greatest
    (non-cumulative) event count - the point of maximum curvature of the
    frequency-magnitude distribution. A common practical correction is to add
    ~0.2; we return the raw maximum-curvature value and leave any correction to
    the caller.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    lo, hi = magnitudes.min(), magnitudes.max()
    edges = np.arange(lo, hi + delta_m, delta_m)
    if edges.size < 2:
        return float(lo)
    counts, edges = np.histogram(magnitudes, bins=edges)
    peak = int(np.argmax(counts))
    return float(edges[peak] + delta_m / 2.0)


def fit_b_value_aki(magnitudes, mc: float, *, delta_m: float = 0.1) -> float:
    """Aki (1965) maximum-likelihood b-value above completeness ``mc``.

    ``b = log10(e) / (mean(M | M >= mc) - (mc - delta_m/2))``

    ``delta_m`` is the magnitude bin width (Utsu correction) accounting for
    rounding of the catalogue magnitudes.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    above = magnitudes[magnitudes >= mc - 1e-9]
    if above.size < 2:
        raise ValueError(
            f"Too few events (>= Mc={mc}) to fit a b-value (got {above.size})"
        )
    mean_mag = above.mean()
    denom = mean_mag - (mc - delta_m / 2.0)
    if denom <= 0:
        raise ValueError(
            "Non-positive (mean - Mc) in Aki b-value fit; check Mc / delta_m"
        )
    return float(np.log10(np.e) / denom)


@dataclass
class GutenbergRichterModel:
    """Truncated Gutenberg-Richter magnitude distribution on ``[mw_min, mw_max]``.

    The (non-cumulative) magnitude PDF implied by ``log10 N(>=M) = a - b*M`` is a
    truncated exponential with rate ``beta = b * ln(10)``:

        p(M) = beta * exp(-beta*(M - mw_min)) / (1 - exp(-beta*(mw_max - mw_min)))

    for ``M in [mw_min, mw_max]``.
    """

    b_value: float
    mw_min: float
    mw_max: float

    def __post_init__(self):
        if self.mw_max <= self.mw_min:
            raise ValueError("mw_max must exceed mw_min")
        if self.b_value <= 0:
            raise ValueError("b_value must be positive")

    @property
    def beta(self) -> float:
        return self.b_value * _LN10

    def sample_magnitudes(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` magnitudes by inverse-CDF transform of the truncated GR."""
        beta = self.beta
        span = 1.0 - np.exp(-beta * (self.mw_max - self.mw_min))
        u = rng.random(n)
        return self.mw_min - np.log1p(-u * span) / beta

    def pdf(self, m):
        """Analytic truncated-exponential PDF (for plotting / KS tests)."""
        m = np.asarray(m, dtype=float)
        beta = self.beta
        norm = 1.0 - np.exp(-beta * (self.mw_max - self.mw_min))
        out = beta * np.exp(-beta * (m - self.mw_min)) / norm
        out = np.where((m >= self.mw_min) & (m <= self.mw_max), out, 0.0)
        return out

    def cdf(self, m):
        """Analytic truncated-exponential CDF (for KS tests)."""
        m = np.asarray(m, dtype=float)
        beta = self.beta
        norm = 1.0 - np.exp(-beta * (self.mw_max - self.mw_min))
        c = (1.0 - np.exp(-beta * (m - self.mw_min))) / norm
        return np.clip(c, 0.0, 1.0)
