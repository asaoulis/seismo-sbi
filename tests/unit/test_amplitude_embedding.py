"""Unit tests for AmplitudeTokenEmbedding.

These prove the physics contract: (1) the array-relative token feature is invariant to the
overall event scale (M0) while the global vector tracks it; (2) the amplitude reference is
computed per-event (no batch coupling) and over valid stations only (masking-invariance);
(3) shapes for per-station / per-component granularity and finiteness on extreme inputs.
Dependency-free / CPU.
"""

import pytest
import torch

from seismo_sbi.sbi.compression.ML.amplitude_embedding import (
    AmplitudeTokenEmbedding,
    DistanceDetrend,
)

pytestmark = pytest.mark.unit

_B, _N, _C, _T, _D = 4, 5, 3, 40, 16


def _x(B=_B, N=_N, C=_C, T=_T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, C, T, generator=g)


# ---------------------------------------------------------------------------
# Array-relative M0-invariance + global-scale tracking
# ---------------------------------------------------------------------------

def test_array_relative_token_is_scale_invariant_global_tracks():
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="array_relative").eval()
    x = _x()
    c = 7.0
    with torch.no_grad():
        tok1, glob1 = emb(x)
        tok2, glob2 = emb(c * x)        # uniformly scale every station of every event
    assert tok1.shape == (_B, _N, _D)
    assert glob1.shape == (_B, _D)
    # Token embedding is invariant to the common M0 scaling...
    assert torch.allclose(tok1, tok2, atol=1e-5)
    # ...while the global (≈log M0) vector moves.
    assert not torch.allclose(glob1, glob2, atol=1e-3)


def test_array_relative_reference_is_per_event_no_batch_coupling():
    """Scaling ONE event's amplitudes must not change another event's embedding."""
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="array_relative").eval()
    x = _x()
    x2 = x.clone()
    x2[0] = x2[0] * 13.0                # scale only event 0
    with torch.no_grad():
        tok1, glob1 = emb(x)
        tok2, glob2 = emb(x2)
    # Event 1..B-1 unchanged (token + global); only event 0's global moves.
    assert torch.allclose(tok1[1:], tok2[1:], atol=1e-6)
    assert torch.allclose(glob1[1:], glob2[1:], atol=1e-6)
    assert not torch.allclose(glob1[0], glob2[0], atol=1e-3)


# ---------------------------------------------------------------------------
# Masking-invariance (variable stations)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["array_relative", "absolute"])
@pytest.mark.parametrize("reference", ["mean", "median"])
def test_padded_station_does_not_change_real_embeddings(mode, reference):
    emb = AmplitudeTokenEmbedding(
        _D, num_components=_C, mode=mode, reference=reference
    ).eval()
    x = _x()
    mask_real = torch.ones(_B, _N, dtype=torch.bool)

    # Append one all-zero padded station, marked invalid.
    pad = torch.zeros(_B, 1, _C, _T)
    x_pad = torch.cat([x, pad], dim=1)
    mask_pad = torch.cat([mask_real, torch.zeros(_B, 1, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        tok_real, glob_real = emb(x, mask=mask_real)
        tok_pad, glob_pad = emb(x_pad, mask=mask_pad)

    # Real stations' token embeddings are unchanged by the padded station.
    assert torch.allclose(tok_real, tok_pad[:, :_N], atol=1e-6)
    # The padded station's token embedding is zeroed out.
    assert torch.allclose(tok_pad[:, _N], torch.zeros(_B, _D), atol=1e-6)
    if mode == "array_relative":
        assert torch.allclose(glob_real, glob_pad, atol=1e-6)


# ---------------------------------------------------------------------------
# Granularity + robustness
# ---------------------------------------------------------------------------

def test_per_component_granularity_shapes():
    emb = AmplitudeTokenEmbedding(
        _D, num_components=_C, mode="array_relative", per_component=True
    ).eval()
    assert emb.K == _C
    with torch.no_grad():
        tok, glob = emb(_x())
    assert tok.shape == (_B, _N, _D)
    assert glob.shape == (_B, _D)
    assert torch.isfinite(tok).all() and torch.isfinite(glob).all()


def test_absolute_mode_shapes_and_no_global():
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="absolute").eval()
    with torch.no_grad():
        tok, glob = emb(_x())
    assert tok.shape == (_B, _N, _D)
    assert glob is None
    assert torch.isfinite(tok).all()


def test_absolute_mode_running_standardizer_updates():
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="absolute").train()
    rs = emb.token_embed.standardizer
    assert not bool(rs.initialized)
    emb(_x())                           # one training forward seeds the running stats
    assert bool(rs.initialized)


def test_finite_on_near_zero_traces():
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="array_relative").eval()
    x = torch.zeros(_B, _N, _C, _T)     # all-zero ⇒ log clamps, must stay finite
    with torch.no_grad():
        tok, glob = emb(x)
    assert torch.isfinite(tok).all() and torch.isfinite(glob).all()


def test_from_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown amplitude_embedding keys"):
        AmplitudeTokenEmbedding.from_config(_D, _C, {"mode": "absolute", "bogus": 1})


# ---------------------------------------------------------------------------
# Distance de-trend
# ---------------------------------------------------------------------------

def test_distance_detrend_identity_at_init():
    det = DistanceDetrend().eval()
    d = torch.rand(4, 5, 1) * 2 + 0.1
    with torch.no_grad():
        g = det(d)
    assert g.shape == (4, 5, 1)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)   # zero-init ⇒ no correction


def test_distance_detrend_uses_distance_when_active():
    det = DistanceDetrend().eval()
    with torch.no_grad():
        det.alpha.fill_(1.0)               # activate the learnable exponent
    d1 = torch.rand(3, 6, 1) * 2 + 0.1
    d2 = d1 * 3.0
    with torch.no_grad():
        g1, g2 = det(d1), det(d2)
    assert not torch.allclose(g1, g2)      # de-trend genuinely depends on distance


def test_amplitude_distance_correction_identity_at_init():
    """With a zero-init de-trend, supplying distance must not change the output."""
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="array_relative",
                                  distance_correction=True).eval()
    x = _x()
    d = torch.rand(_B, _N, 1) * 2 + 0.1
    with torch.no_grad():
        t_with, g_with = emb(x, distance=d)
        t_none, g_none = emb(x, distance=None)
    assert torch.allclose(t_with, t_none, atol=1e-6)
    assert torch.allclose(g_with, g_none, atol=1e-6)


def test_amplitude_distance_correction_changes_output_when_active():
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="array_relative",
                                  distance_correction=True).eval()
    with torch.no_grad():
        emb.detrend.alpha.fill_(1.0)
    x = _x()
    d = torch.rand(_B, _N, 1) * 2 + 0.1
    with torch.no_grad():
        t_dist, _ = emb(x, distance=d)
        t_nodist, _ = emb(x, distance=None)
    assert not torch.allclose(t_dist, t_nodist, atol=1e-5)


def test_uses_distance_property():
    assert AmplitudeTokenEmbedding(_D, _C, distance_correction=True).uses_distance is True
    assert AmplitudeTokenEmbedding(_D, _C, distance_correction=False).uses_distance is False


# ---------------------------------------------------------------------------
# SNR-weighted reference
# ---------------------------------------------------------------------------

def test_snr_weights_higher_for_clean_than_noisy():
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="array_relative",
                                  snr_weighting=True).eval()
    B, N, C, T = 1, 2, _C, 64
    x = torch.zeros(B, N, C, T)
    x[0, 0, 0, T // 2] = 50.0                                   # station 0: clean high-SNR spike
    x[0, 1] = torch.randn(C, T, generator=torch.Generator().manual_seed(1)) * 0.5  # station 1: noise
    with torch.no_grad():
        w = emb._snr_weights(x)
    assert w.shape == (B, N, 1)
    assert w[0, 0, 0] > w[0, 1, 0]


def test_snr_weighting_changes_reference_embedding():
    """Toggling SNR weighting on the SAME model + input changes the embedding (the reference)."""
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="array_relative",
                                  snr_weighting=True).eval()
    g = torch.Generator().manual_seed(3)
    x = torch.randn(_B, _N, _C, _T, generator=g) * 0.3
    x[:, 0, :, _T // 2] += 40.0                                 # one clean, high-SNR station
    with torch.no_grad():
        out_snr, _ = emb(x)
        emb.snr_weighting = False
        out_plain, _ = emb(x)
    assert not torch.allclose(out_snr, out_plain, atol=1e-5)


def test_reference_weighted_mean_excludes_zero_weight():
    emb = AmplitudeTokenEmbedding(_D, num_components=1, mode="array_relative").eval()
    c = torch.tensor([[[1.0], [3.0], [9.0]]])                   # (1,3,1)
    weight = torch.tensor([[[1.0], [1.0], [0.0]]])             # third station excluded
    ref = emb._reference(c, weight, use_median=False, mask=None)
    assert torch.allclose(ref, torch.tensor([[[2.0]]]))         # mean of 1 and 3


def test_padded_station_invariance_with_snr_weighting():
    emb = AmplitudeTokenEmbedding(_D, num_components=_C, mode="array_relative",
                                  snr_weighting=True).eval()
    x = _x()
    mask_real = torch.ones(_B, _N, dtype=torch.bool)
    pad = torch.randn(_B, 1, _C, _T) * 30.0                     # garbage-amplitude padded station
    x_pad = torch.cat([x, pad], dim=1)
    mask_pad = torch.cat([mask_real, torch.zeros(_B, 1, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        tok_real, glob_real = emb(x, mask=mask_real)
        tok_pad, glob_pad = emb(x_pad, mask=mask_pad)
    assert torch.allclose(tok_real, tok_pad[:, :_N], atol=1e-6)
    assert torch.allclose(glob_real, glob_pad, atol=1e-6)


def test_from_config_accepts_robustness_keys():
    emb = AmplitudeTokenEmbedding.from_config(_D, _C, {
        "mode": "array_relative", "distance_correction": True,
        "snr_weighting": True, "snr_floor_quantile": 0.1,
    })
    assert emb.uses_distance and emb.snr_weighting
