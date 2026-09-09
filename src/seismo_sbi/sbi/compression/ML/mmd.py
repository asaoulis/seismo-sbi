"""Maximum Mean Discrepancy for misspecification-robust training.

The summary-space auxiliary loss of Huang et al. (2023), in a two-sample setting: many
quality-controlled real events against a posterior-matched simulation suite. The estimator is
the unbiased U-statistic of Gretton et al. (2012, Eq. 4) rather than the biased V-statistic,
over a mixture of RBF kernels at several bandwidth scales around a median-heuristic base, which
hedges kernel choice. The bandwidth is detached from the graph; any smoothing across steps is
the caller's. Pure torch and stateless.
"""
from __future__ import annotations

from typing import Sequence

import torch

DEFAULT_BANDWIDTH_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def median_bandwidth(z: torch.Tensor, z2: torch.Tensor = None) -> float:
    """Median-heuristic RBF bandwidth over the pooled samples.

    beta = sqrt(med / 2) where ``med`` is the median of squared pairwise two-norm
    distances between distinct points of ``z`` (pooled with ``z2`` when given) —
    the convention of Huang et al. 2023, computed here on the pooled two-sample set.
    Always detached (bandwidth selection must not backprop). Returns a python float;
    falls back to 1.0 if the median is degenerate (all points identical).
    """
    with torch.no_grad():
        pool = z if z2 is None else torch.cat([z.detach(), z2.detach()], dim=0)
        pool = pool.detach()
        if pool.shape[0] < 2:
            return 1.0
        d2 = torch.cdist(pool, pool).pow(2)
        n = pool.shape[0]
        off_diag = d2[~torch.eye(n, dtype=torch.bool, device=pool.device)]
        med = off_diag.median().item()
        if not (med > 0):
            return 1.0
        return float((med / 2.0) ** 0.5)


def _rbf_mixture(d2: torch.Tensor, bandwidths: Sequence[float]) -> torch.Tensor:
    """Mean of RBF kernels exp(-d2 / beta^2) over the bandwidth mixture.

    The MEAN (not sum) keeps the kernel bounded in [0, 1] so ``lambda_mmd`` is
    invariant to how many bandwidths are in the mixture.
    """
    k = torch.zeros_like(d2)
    for b in bandwidths:
        k = k + torch.exp(-d2 / float(b) ** 2)
    return k / len(bandwidths)


def rbf_mixture_mmd2_unbiased(z1: torch.Tensor, z2: torch.Tensor,
                              bandwidths: Sequence[float]) -> torch.Tensor:
    """Unbiased squared-MMD U-statistic with a mixture-of-RBF kernel.

    MMD^2_u = 1/(n(n-1)) sum_{i != j} k(z1_i, z1_j) + 1/(m(m-1)) sum_{i != j} k(z2_i, z2_j)
              - 2/(nm) sum_{i,j} k(z1_i, z2_j)

    Needs >= 2 samples per side. Being unbiased it may go slightly NEGATIVE near zero —
    callers must not clamp before logging (a near-zero/negative value is the "aligned"
    signal). Differentiable w.r.t. both inputs; ``bandwidths`` are plain floats
    (select via :func:`median_bandwidth`, outside the graph).
    """
    n, m = z1.shape[0], z2.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"unbiased MMD needs >=2 samples per side, got {n} and {m}")
    d2_11 = torch.cdist(z1, z1).pow(2)
    d2_22 = torch.cdist(z2, z2).pow(2)
    d2_12 = torch.cdist(z1, z2).pow(2)
    k11 = _rbf_mixture(d2_11, bandwidths)
    k22 = _rbf_mixture(d2_22, bandwidths)
    k12 = _rbf_mixture(d2_12, bandwidths)
    eye_n = torch.eye(n, dtype=torch.bool, device=z1.device)
    eye_m = torch.eye(m, dtype=torch.bool, device=z2.device)
    term_11 = k11.masked_fill(eye_n, 0.0).sum() / (n * (n - 1))
    term_22 = k22.masked_fill(eye_m, 0.0).sum() / (m * (m - 1))
    term_12 = 2.0 * k12.mean()
    return term_11 + term_22 - term_12
