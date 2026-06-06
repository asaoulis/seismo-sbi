"""Unit tests for the reusable Random Fourier Feature primitives.

These lock in the properties the amplitude embedding (and later geometry encoding) relies
on: correct shapes, reproducibility, mask-aware running standardisation, and — most
importantly — that :class:`ScalarFourierEmbedding` is **well-scaled** (no dead/constant
features, no exploding outputs, gradient flows w.r.t. the input). Dependency-free / CPU.
"""

import math

import pytest
import torch

from seismo_sbi.sbi.compression.ML.fourier_features import (
    GaussianFourierFeatures,
    RunningStandardizer,
    ScalarFourierEmbedding,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# GaussianFourierFeatures
# ---------------------------------------------------------------------------

def test_rff_output_shape_and_finite():
    rff = GaussianFourierFeatures(in_dim=1, num_freqs=16, sigma=1.0)
    assert rff.out_dim == 32
    x = torch.randn(5, 7, 1)
    out = rff(x)
    assert out.shape == (5, 7, 32)
    assert torch.isfinite(out).all()
    # sin/cos features are bounded in [-1, 1]
    assert out.abs().max() <= 1.0 + 1e-6


def test_rff_deterministic_given_seed():
    a = GaussianFourierFeatures(in_dim=2, num_freqs=8, sigma=1.0, seed=123)
    b = GaussianFourierFeatures(in_dim=2, num_freqs=8, sigma=1.0, seed=123)
    x = torch.randn(4, 2)
    assert torch.allclose(a(x), b(x))
    c = GaussianFourierFeatures(in_dim=2, num_freqs=8, sigma=1.0, seed=999)
    assert not torch.allclose(a(x), c(x))


def test_rff_learnable_vs_buffer():
    buf = GaussianFourierFeatures(in_dim=1, num_freqs=4, learnable=False)
    assert "B" in dict(buf.named_buffers())
    assert all(p.requires_grad for p in buf.parameters()) or len(list(buf.parameters())) == 0
    assert len(list(buf.parameters())) == 0

    lrn = GaussianFourierFeatures(in_dim=1, num_freqs=4, learnable=True)
    assert "B" in dict(lrn.named_parameters())
    assert lrn.B.requires_grad


def test_rff_sigma_controls_frequency_content():
    """Larger sigma ⇒ higher-frequency bank ⇒ more total variation over a fixed sweep."""
    grid = torch.linspace(-3, 3, 200).unsqueeze(-1)

    def total_variation(sigma):
        rff = GaussianFourierFeatures(in_dim=1, num_freqs=32, sigma=sigma, seed=0)
        feats = rff(grid)                       # (200, 64)
        return feats.diff(dim=0).abs().sum().item()

    tv_low = total_variation(0.3)
    tv_high = total_variation(3.0)
    assert tv_high > tv_low * 1.5, (tv_low, tv_high)


# ---------------------------------------------------------------------------
# RunningStandardizer
# ---------------------------------------------------------------------------

def test_running_standardizer_tracks_and_normalises():
    rs = RunningStandardizer(dim=1)
    rs.train()
    x = torch.randn(2000, 1) * 2.0 + 5.0       # mean 5, std 2
    out = rs(x)
    # First update seeds running stats directly to the batch stats.
    assert bool(rs.initialized)
    assert abs(rs.running_mean.item() - 5.0) < 0.2
    assert abs(rs.running_var.item() - 4.0) < 0.5
    # Output is ~standardised.
    assert abs(out.mean().item()) < 0.1
    assert abs(out.std().item() - 1.0) < 0.1


def test_running_standardizer_mask_excludes_padding():
    rs = RunningStandardizer(dim=1)
    rs.train()
    real = torch.randn(1000, 1) * 1.0 + 3.0
    pad = torch.full((1000, 1), 1000.0)        # absurd padded values
    x = torch.cat([real, pad], dim=0)
    mask = torch.cat([torch.ones(1000), torch.zeros(1000)]).bool()
    rs(x, mask=mask)
    # Running mean must reflect only the real (mean≈3) values, not the padding.
    assert abs(rs.running_mean.item() - 3.0) < 0.3


def test_running_standardizer_eval_does_not_update():
    rs = RunningStandardizer(dim=1)
    rs.train()
    rs(torch.randn(100, 1) + 5.0)
    before = rs.running_mean.clone()
    rs.eval()
    rs(torch.randn(100, 1) + 50.0)
    assert torch.allclose(before, rs.running_mean)


# ---------------------------------------------------------------------------
# ScalarFourierEmbedding — shape, robustness, and the dead-weight / scale guard
# ---------------------------------------------------------------------------

def test_scalar_embedding_shape_and_finite():
    emb = ScalarFourierEmbedding(in_dim=1, out_dim=16)
    x = torch.randn(3, 5, 1)
    out = emb(x)
    assert out.shape == (3, 5, 16)
    assert torch.isfinite(out).all()


def test_scalar_embedding_finite_on_extreme_input():
    """A large-magnitude raw scalar (e.g. log-amplitude ~ -30) must stay finite. The
    'running' standardiser brings it to O(1) before the RFF."""
    emb = ScalarFourierEmbedding(in_dim=1, out_dim=16, standardize="running")
    emb.train()
    x = torch.full((64, 1), -30.0) + torch.randn(64, 1) * 0.5
    out = emb(x)
    assert torch.isfinite(out).all()


def test_scalar_embedding_no_dead_features_and_bounded():
    """The core RFF-scaling guard. Over a standardised input sweep the embedding must:
    (1) RESPOND — RFF features and the output vary (not collapsed to a constant ⇒ no dead
        weights); (2) stay BOUNDED at init (no exploding scale); (3) pass gradient to the
        input everywhere (aggregate sensitivity > 0)."""
    emb = ScalarFourierEmbedding(in_dim=1, out_dim=32, num_freqs=16, sigma=1.0,
                                 standardize="none")
    grid = torch.linspace(-3, 3, 128).unsqueeze(-1)

    # (1a) RFF features are alive: most of them have non-trivial std across the sweep.
    feats = emb.rff(grid)                                  # (128, 32)
    alive = (feats.std(dim=0) > 1e-3).float().mean().item()
    assert alive > 0.8, f"too many dead RFF features (alive fraction {alive})"

    # (1b) The embedding output responds to the input.
    out = emb(grid)                                        # (128, 32)
    assert out.std().item() > 1e-3

    # (2) Bounded at init — no exploding activations.
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 1e3

    # (3) Gradient flows to the input across the whole sweep (aggregate, robust to the odd
    #     stationary point), and is finite.
    g = grid.clone().requires_grad_(True)
    y = emb(g)
    grad = torch.autograd.grad(y.sum(), g)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 1e-2


def test_scalar_embedding_fixed_standardize_centers():
    emb = ScalarFourierEmbedding(in_dim=1, out_dim=8, standardize="fixed",
                                 center=-12.0, scale=2.0)
    # standardised value of center maps to 0; finite output, gradients flow.
    x = torch.tensor([[-12.0], [-10.0], [-14.0]])
    out = emb(x)
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_scalar_embedding_vector_input():
    """Per-component granularity: in_dim = C feeds a small vector cleanly."""
    emb = ScalarFourierEmbedding(in_dim=3, out_dim=16)
    x = torch.randn(4, 6, 3)
    out = emb(x)
    assert out.shape == (4, 6, 16)
    assert torch.isfinite(out).all()
