"""Unit tests for the MMD auxiliary-loss estimator (seismo_sbi.sbi.compression.ML.mmd)."""
import numpy as np
import pytest
import torch

from seismo_sbi.sbi.compression.ML.mmd import (
    DEFAULT_BANDWIDTH_SCALES,
    median_bandwidth,
    rbf_mixture_mmd2_unbiased,
)


def _bw(beta):
    return [beta * s for s in DEFAULT_BANDWIDTH_SCALES]


def test_mmd_near_zero_for_same_distribution():
    g = torch.Generator().manual_seed(0)
    z1 = torch.randn(256, 8, generator=g)
    z2 = torch.randn(256, 8, generator=g)
    beta = median_bandwidth(z1, z2)
    m = rbf_mixture_mmd2_unbiased(z1, z2, _bw(beta)).item()
    assert abs(m) < 0.01     # unbiased: fluctuates around 0, may be slightly negative


def test_mmd_positive_for_shifted_distribution_and_ordering():
    g = torch.Generator().manual_seed(1)
    z1 = torch.randn(256, 8, generator=g)
    small = torch.randn(256, 8, generator=g) + 0.5
    large = torch.randn(256, 8, generator=g) + 3.0
    beta = median_bandwidth(z1, small)
    m_small = rbf_mixture_mmd2_unbiased(z1, small, _bw(beta)).item()
    m_large = rbf_mixture_mmd2_unbiased(z1, large, _bw(median_bandwidth(z1, large))).item()
    assert m_small > 0.01
    assert m_large > m_small     # bigger shift => bigger MMD


def test_mmd_symmetry():
    g = torch.Generator().manual_seed(2)
    z1 = torch.randn(64, 4, generator=g)
    z2 = torch.randn(96, 4, generator=g) + 1.0
    bw = _bw(median_bandwidth(z1, z2))
    a = rbf_mixture_mmd2_unbiased(z1, z2, bw).item()
    b = rbf_mixture_mmd2_unbiased(z2, z1, bw).item()
    assert a == pytest.approx(b, rel=1e-5)


def test_mmd_unbiasedness_identical_points_excluded():
    # With the SAME sample on both sides the unbiased estimator is exactly
    # (mean off-diag k11) + (mean off-diag k22) - 2 mean(k12); the diagonal of k12 is 1
    # but the within terms exclude the diagonal, so the result is negative — a direct
    # check that the diagonal (i==j) self-similarity terms are excluded within-sample.
    g = torch.Generator().manual_seed(3)
    z = torch.randn(32, 4, generator=g)
    m = rbf_mixture_mmd2_unbiased(z, z, _bw(median_bandwidth(z))).item()
    assert m < 0.0


def test_mmd_gradients_flow_to_both_sides():
    g = torch.Generator().manual_seed(4)
    z1 = torch.randn(16, 4, generator=g, requires_grad=True)
    z2 = (torch.randn(16, 4, generator=g) + 1.0).requires_grad_(True)
    m = rbf_mixture_mmd2_unbiased(z1, z2, _bw(1.0))
    m.backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all() and z1.grad.abs().sum() > 0
    assert z2.grad is not None and torch.isfinite(z2.grad).all() and z2.grad.abs().sum() > 0


def test_mmd_raises_below_two_samples():
    z1 = torch.randn(1, 4)
    z2 = torch.randn(8, 4)
    with pytest.raises(ValueError):
        rbf_mixture_mmd2_unbiased(z1, z2, [1.0])


def test_median_bandwidth_matches_numpy_and_is_detached():
    g = torch.Generator().manual_seed(5)
    z = torch.randn(64, 6, generator=g, requires_grad=True)
    beta = median_bandwidth(z)
    zn = z.detach().numpy()
    d2 = ((zn[:, None, :] - zn[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[~np.eye(len(zn), dtype=bool)])
    assert beta == pytest.approx(float(np.sqrt(med / 2.0)), rel=1e-4)
    assert isinstance(beta, float)                     # plain float => nothing to backprop
    # degenerate inputs fall back to 1.0
    assert median_bandwidth(torch.zeros(8, 3)) == 1.0
    assert median_bandwidth(torch.zeros(1, 3)) == 1.0
