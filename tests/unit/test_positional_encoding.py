"""Unit tests for the RFF station positional encoding (review §3.2).

The load-bearing properties: correct shapes, **azimuth periodicity** (11° ≈ 350°, continuity
across the ±180°/0–360° branch cut — the one subtle correctness requirement), depth/distance
dependence with correct broadcasting, a well-scaled RFF (no dead/exploding features, gradient
flows), masking-invariance, and config validation. Dependency-free / CPU.
"""

import math

import pytest
import torch

from seismo_sbi.sbi.compression.ML.positional_encoding import (
    FourierStationPositionalEncoding,
    _CONFIG_KEYS,
)

pytestmark = pytest.mark.unit


def _deg(degrees):
    return torch.deg2rad(torch.tensor(degrees, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Shapes / finiteness
# ---------------------------------------------------------------------------

def test_relative_shape_and_finite():
    pe = FourierStationPositionalEncoding(d_model=32, coords_kind="relative")
    assert pe.in_dim == 3  # (distance, cos az, sin az)
    coords = torch.stack([
        torch.rand(4, 7),                     # distance (radians-ish)
        (torch.rand(4, 7) * 2 - 1) * math.pi,  # azimuth in (-pi, pi]
    ], dim=-1)                                 # (4, 7, 2)
    out = pe(coords)
    assert out.shape == (4, 7, 32)
    assert torch.isfinite(out).all()


def test_absolute_shape_and_finite():
    pe = FourierStationPositionalEncoding(d_model=16, coords_kind="absolute")
    assert pe.in_dim == 2
    coords = torch.randn(3, 5, 2) * 30.0       # lat/lon-ish degrees
    out = pe(coords)
    assert out.shape == (3, 5, 16)
    assert torch.isfinite(out).all()


def test_include_depth_in_dim_and_shape():
    pe = FourierStationPositionalEncoding(d_model=16, coords_kind="relative",
                                          include_depth=True)
    assert pe.in_dim == 4  # + depth
    coords = torch.rand(2, 6, 2)
    depth = torch.tensor([[10.0], [25.0]])     # (B, 1) km
    out = pe(coords, depth=depth)
    assert out.shape == (2, 6, 16)
    assert torch.isfinite(out).all()


def test_include_depth_without_depth_raises():
    pe = FourierStationPositionalEncoding(d_model=8, coords_kind="relative",
                                          include_depth=True)
    with pytest.raises(ValueError):
        pe(torch.rand(2, 3, 2), depth=None)


# ---------------------------------------------------------------------------
# Azimuth periodicity / symmetry — the subtle correctness requirement
# ---------------------------------------------------------------------------

def test_azimuth_exact_periodicity_no_branch_cut():
    """The rock-solid symmetry guarantee: encoding azimuth as (cos, sin) makes the map
    EXACTLY 360°-periodic, so ``emb(θ) == emb(θ + 360°)`` and there is no branch-cut
    discontinuity — independent of the weights or the standardisation mode."""
    for standardize in ("none", "running"):
        torch.manual_seed(0)
        pe = FourierStationPositionalEncoding(d_model=48, coords_kind="relative",
                                              standardize=standardize, num_freqs=16, seed=3)
        if standardize == "running":
            # Populate running stats over a full uniform azimuth sweep at varied distances.
            az = torch.linspace(-math.pi, math.pi, 256)
            dist = torch.rand(256) * 0.8 + 0.1
            pe.train()
            pe(torch.stack([dist, az], dim=-1).unsqueeze(0))   # (1, 256, 2)
        pe.eval()

        def emb_at(az_deg, dist=0.4):
            return pe(torch.tensor([[[dist, _deg(az_deg).item()]]])).reshape(-1)

        assert torch.allclose(emb_at(0.0), emb_at(360.0), atol=1e-5)
        assert torch.allclose(emb_at(181.0), emb_at(-179.0), atol=1e-5)   # same physical angle
        assert torch.allclose(emb_at(11.0), emb_at(371.0), atol=1e-5)


def test_azimuth_closeness_ordering_and_continuity():
    """11° is physically ~21° from 350° but ~169° from 180°; with a smooth (low-σ) RFF the
    embedding distance respects that ordering, and a small step across the 0/360 wrap is far
    smaller than a large angular separation (no discontinuity). Seeded for determinism."""
    torch.manual_seed(0)
    pe = FourierStationPositionalEncoding(d_model=64, coords_kind="relative",
                                          standardize="none", num_freqs=16, sigma=0.3, seed=1)
    pe.eval()

    def emb_at(az_deg, dist=0.4):
        return pe(torch.tensor([[[dist, _deg(az_deg).item()]]])).reshape(-1)

    d_11_350 = (emb_at(11.0) - emb_at(350.0)).norm().item()
    d_11_180 = (emb_at(11.0) - emb_at(180.0)).norm().item()
    assert d_11_350 < d_11_180, (d_11_350, d_11_180)

    # A 2° step straddling the 0/360 cut is much smaller than a 169° separation.
    step_across_wrap = (emb_at(359.0) - emb_at(1.0)).norm().item()
    assert step_across_wrap < d_11_180, (step_across_wrap, d_11_180)


# ---------------------------------------------------------------------------
# Distance / depth dependence and broadcasting
# ---------------------------------------------------------------------------

def test_distance_dependence():
    pe = FourierStationPositionalEncoding(d_model=32, coords_kind="relative",
                                          standardize="none", seed=1)
    pe.eval()
    az = _deg(45.0).item()
    near = pe(torch.tensor([[[0.10, az]]])).reshape(-1)
    far = pe(torch.tensor([[[0.90, az]]])).reshape(-1)
    assert (near - far).norm().item() > 1e-3


def test_depth_dependence_and_broadcast():
    pe = FourierStationPositionalEncoding(d_model=32, coords_kind="relative",
                                          include_depth=True, standardize="none", seed=5)
    pe.eval()
    # Identical coords across stations isolates the shared depth channel.
    coords = torch.rand(1, 1, 2).expand(1, 4, 2).contiguous()
    shallow = pe(coords, depth=torch.tensor([[5.0]]))
    deep = pe(coords, depth=torch.tensor([[80.0]]))
    # Depth changes the embedding...
    assert (shallow - deep).norm().item() > 1e-3
    # ...and the depth is broadcast equally over stations: with identical per-station coords,
    # every station gets the same embedding at a given depth.
    assert torch.allclose(deep[0], deep[0, 0:1].expand_as(deep[0]), atol=1e-5)
    assert torch.allclose(shallow[0], shallow[0, 0:1].expand_as(shallow[0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Reproducibility / learnability
# ---------------------------------------------------------------------------

def test_determinism_given_seed():
    """``seed`` controls the (reproducible) Gaussian frequency bank ``B`` — same seed ⇒ same
    bank, different seed ⇒ different bank. (The MLP head uses the global RNG, by design.)"""
    a = FourierStationPositionalEncoding(d_model=16, coords_kind="relative", seed=42)
    b = FourierStationPositionalEncoding(d_model=16, coords_kind="relative", seed=42)
    assert torch.allclose(a.embed.rff.B, b.embed.rff.B)
    c = FourierStationPositionalEncoding(d_model=16, coords_kind="relative", seed=99)
    assert not torch.allclose(a.embed.rff.B, c.embed.rff.B)


def test_learnable_vs_buffer_freqs():
    buf = FourierStationPositionalEncoding(d_model=16, coords_kind="relative",
                                           learnable_freqs=False)
    assert "embed.rff.B" in dict(buf.named_buffers())
    lrn = FourierStationPositionalEncoding(d_model=16, coords_kind="relative",
                                           learnable_freqs=True)
    assert "embed.rff.B" in dict(lrn.named_parameters())
    assert lrn.embed.rff.B.requires_grad


# ---------------------------------------------------------------------------
# Dead-weight / scale guard (mirrors the ScalarFourierEmbedding guard)
# ---------------------------------------------------------------------------

def test_no_dead_features_bounded_and_gradient_flows():
    """Over an azimuth+distance sweep the encoder must RESPOND (no dead/constant features),
    stay BOUNDED at init, and pass a finite gradient back to the coordinates."""
    pe = FourierStationPositionalEncoding(d_model=64, coords_kind="relative",
                                          standardize="none", num_freqs=32, sigma=1.0, seed=0)
    az = torch.linspace(-math.pi, math.pi, 128)
    dist = torch.linspace(0.05, 1.2, 128)
    coords = torch.stack([dist, az], dim=-1).unsqueeze(0)   # (1, 128, 2)

    out = pe(coords)
    assert torch.isfinite(out).all()
    assert out.std().item() > 1e-3           # responds
    assert out.abs().max().item() < 1e3       # bounded at init

    g = coords.clone().requires_grad_(True)
    y = pe(g)
    grad = torch.autograd.grad(y.sum(), g)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 1e-2


def test_finite_on_zero_and_at_source_coords():
    """Padded / at-source coordinates (all zeros) must stay finite (no NaN from cos/sin)."""
    pe = FourierStationPositionalEncoding(d_model=16, coords_kind="relative",
                                          include_depth=True, standardize="none")
    pe.eval()
    out = pe(torch.zeros(2, 3, 2), depth=torch.zeros(2, 1))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Masking-invariance
# ---------------------------------------------------------------------------

def test_masking_invariance_padded_station():
    """Appending a padded station (mask=False) with extreme coords leaves the real stations'
    embeddings unchanged, and the padded output is zeroed."""
    pe = FourierStationPositionalEncoding(d_model=32, coords_kind="relative",
                                          standardize="none", seed=4)
    pe.eval()
    real = torch.rand(1, 3, 2)
    out_real = pe(real, mask=torch.ones(1, 3, dtype=torch.bool))   # (1, 3, 32)

    padded = torch.cat([real, torch.full((1, 1, 2), 999.0)], dim=1)  # extreme padded coords
    mask = torch.tensor([[True, True, True, False]])
    out_padded = pe(padded, mask=mask)                            # (1, 4, 32)

    assert torch.allclose(out_real, out_padded[:, :3], atol=1e-6)
    assert torch.allclose(out_padded[:, 3], torch.zeros(1, 32))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_from_config_accepts_known_and_rejects_unknown():
    cfg = {"mode": "fourier", "inject_every_layer": True, "num_freqs": 8,
           "sigma": 0.5, "learnable_freqs": True, "include_depth": True, "standardize": "none"}
    pe = FourierStationPositionalEncoding.from_config(32, "relative", cfg)
    assert pe.in_dim == 4 and pe.include_depth
    assert pe.embed.rff.num_freqs == 8

    with pytest.raises(ValueError):
        FourierStationPositionalEncoding.from_config(32, "relative", {"bogus": 1})


def test_config_keys_cover_module_and_transformer_level():
    assert {"mode", "inject_every_layer"} <= _CONFIG_KEYS
    assert {"num_freqs", "sigma", "learnable_freqs", "include_depth", "standardize"} <= _CONFIG_KEYS
