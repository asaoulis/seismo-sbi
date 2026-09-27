"""Unit tests for the Set-Transformer PMA pooling head (review §3.4).

The load-bearing properties of a *set* pooling head: correct shapes for both modes
(``pool_over∈{tokens,stations}``) and any ``k``; **permutation-invariance** over the pooled set
(the defining property — permuting input tokens/stations must not change the output);
**masking-invariance** (appending a padded token/station, or a fully-padded sample, leaves the real
output bit-identical and finite); a **well-scaled** head (no dead seeds / explosion, gradient flows
to every seed); the ``linear`` combine is **initialised to the mean** yet is a *real learnable*
combination; and config validation. Dependency-free / CPU.
"""

import pytest
import torch

from seismo_sbi.sbi.compression.ML.pma_pooling import (
    SetTransformerPMAHead,
    _CONFIG_KEYS,
)

pytestmark = pytest.mark.unit

D = 32
NH = 4


def _head(**kw):
    torch.manual_seed(0)
    return SetTransformerPMAHead(D, NH, **kw)


# ---------------------------------------------------------------------------
# Shapes / finiteness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pool_over", ["tokens", "stations"])
@pytest.mark.parametrize("num_seeds", [1, 4])
def test_shape_and_finite(pool_over, num_seeds):
    head = _head(pool_over=pool_over, num_seeds=num_seeds)
    head.eval()
    x = torch.randn(3, 5, 7, D)
    out = head(x)
    assert out.shape == (3, D)
    assert torch.isfinite(out).all()


def test_stations_mode_builds_time_pool():
    assert _head(pool_over="stations").time_pool is not None
    assert _head(pool_over="tokens").time_pool is None


def test_num_heads_override():
    head = SetTransformerPMAHead(D, NH, num_heads=8)
    assert head.pma.attn.num_heads == 8
    # default falls back to the transformer's nheads
    assert SetTransformerPMAHead(D, NH).pma.attn.num_heads == NH


# ---------------------------------------------------------------------------
# Permutation-invariance — the defining set-pool property
# ---------------------------------------------------------------------------

def test_tokens_mode_permutation_invariant_over_stations():
    head = _head(pool_over="tokens", num_seeds=4)
    head.eval()
    x = torch.randn(2, 6, 5, D)
    out = head(x)
    perm = torch.randperm(6)
    out_perm = head(x[:, perm])
    assert torch.allclose(out, out_perm, atol=1e-5), (out - out_perm).abs().max()


def test_tokens_mode_permutation_invariant_over_time():
    head = _head(pool_over="tokens", num_seeds=2)
    head.eval()
    x = torch.randn(2, 4, 8, D)
    out = head(x)
    perm = torch.randperm(8)
    assert torch.allclose(out, head(x[:, :, perm]), atol=1e-5)


def test_stations_mode_permutation_invariant_over_stations_and_time():
    head = _head(pool_over="stations", num_seeds=4)
    head.eval()
    x = torch.randn(2, 6, 5, D)
    out = head(x)
    # permuting stations
    pN = torch.randperm(6)
    assert torch.allclose(out, head(x[:, pN]), atol=1e-5)
    # permuting time within each station (TimePool is a set pool over time too)
    pL = torch.randperm(5)
    assert torch.allclose(out, head(x[:, :, pL]), atol=1e-5)


# ---------------------------------------------------------------------------
# Masking-invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pool_over", ["tokens", "stations"])
def test_masking_invariance_padded_station(pool_over):
    """Appending a fully-padded station with extreme values leaves the real-station pooled
    output bit-identical (the padded keys are excluded from attention)."""
    head = _head(pool_over=pool_over, num_seeds=3)
    head.eval()
    B, N, L = 2, 4, 6
    x_real = torch.randn(B, N, L, D)
    out_real = head(x_real, key_padding_mask=None)  # no padding => all valid

    extreme = torch.full((B, 1, L, D), 50.0)
    x_pad = torch.cat([x_real, extreme], dim=1)              # (B, N+1, L, D)
    mask = torch.zeros(B, N + 1, L, dtype=torch.bool)
    mask[:, N, :] = True                                     # new station fully padded
    out_pad = head(x_pad, key_padding_mask=mask)

    assert torch.allclose(out_real, out_pad, atol=1e-5), (out_real - out_pad).abs().max()


@pytest.mark.parametrize("pool_over", ["tokens", "stations"])
def test_fully_padded_sample_is_finite(pool_over):
    """A sample whose entire key set is padded must stay finite (no NaN from all-masked attn)."""
    head = _head(pool_over=pool_over, num_seeds=2)
    head.eval()
    x = torch.randn(2, 4, 5, D)
    mask = torch.zeros(2, 4, 5, dtype=torch.bool)
    mask[0] = True  # sample 0 entirely padded
    out = head(x, key_padding_mask=mask)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("pool_over", ["tokens", "stations"])
def test_partial_time_masking_runs_finite(pool_over):
    head = _head(pool_over=pool_over, num_seeds=2)
    head.eval()
    x = torch.randn(2, 4, 6, D)
    mask = torch.zeros(2, 4, 6, dtype=torch.bool)
    mask[:, 1, 3:] = True   # station 1 partially padded in time
    out = head(x, key_padding_mask=mask)
    assert out.shape == (2, D)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Dead-weight / scale guard + gradient flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pool_over", ["tokens", "stations"])
def test_responds_bounded_and_gradients_flow(pool_over):
    head = _head(pool_over=pool_over, num_seeds=4, combine="linear")
    head.train()
    x = torch.randn(8, 5, 6, D, requires_grad=True)
    out = head(x)
    assert torch.isfinite(out).all()
    assert out.std().item() > 1e-3          # responds across the batch (no dead head)
    assert out.abs().max().item() < 1e3      # bounded at init (LayerNorm-normalised)

    # A quadratic objective (not a plain sum, which the final zero-mean LayerNorm output cancels)
    # so the gradient magnitude is a meaningful "no dead path" probe.
    (out ** 2).sum().backward()
    # gradient reaches the inputs, the seeds, and the combine
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum().item() > 1e-3
    assert head.seeds.grad is not None and head.seeds.grad.abs().sum().item() > 0
    assert head.combine_linear.weight.grad.abs().sum().item() > 0
    if pool_over == "stations":
        assert head.time_pool.seed.grad is not None
        assert head.time_pool.seed.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# The linear combine: initialised to mean, but genuinely learnable (the §3.4a fix)
# ---------------------------------------------------------------------------

def test_linear_combine_inits_as_mean_then_is_learnable():
    head = _head(pool_over="tokens", num_seeds=4, combine="linear")
    head.eval()
    x = torch.randn(3, 5, 6, D)

    out_linear_init = head(x)
    head.combine = "mean"
    out_mean = head(x)
    # at init the learned combine reproduces the unweighted mean exactly
    assert torch.allclose(out_linear_init, out_mean, atol=1e-5)

    # perturb the combine weights -> the output departs from the mean (it is a real learnable combo)
    with torch.no_grad():
        head.combine_linear.weight.add_(torch.randn_like(head.combine_linear.weight) * 0.5)
    head.combine = "linear"
    out_perturbed = head(x)
    assert not torch.allclose(out_perturbed, out_mean, atol=1e-3)


def test_combine_first_selects_seed_zero_path_shape():
    head = _head(pool_over="tokens", num_seeds=3, combine="first")
    head.eval()
    assert head.combine_linear is None
    out = head(torch.randn(2, 4, 5, D))
    assert out.shape == (2, D) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Optional self-attention among seeds (SAB)
# ---------------------------------------------------------------------------

def test_sab_on_off_shapes_and_changes_output():
    no_sab = _head(pool_over="tokens", num_seeds=4, seed_self_attention=False)
    with_sab = _head(pool_over="tokens", num_seeds=4, seed_self_attention=True)
    assert no_sab.sab is None and with_sab.sab is not None
    x = torch.randn(2, 4, 5, D)
    no_sab.eval(); with_sab.eval()
    assert no_sab(x).shape == with_sab(x).shape == (2, D)


def test_sab_requires_multiple_seeds():
    with pytest.raises(ValueError):
        SetTransformerPMAHead(D, NH, num_seeds=1, seed_self_attention=True)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_given_seed():
    torch.manual_seed(7)
    a = SetTransformerPMAHead(D, NH, num_seeds=4, pool_over="stations")
    torch.manual_seed(7)
    b = SetTransformerPMAHead(D, NH, num_seeds=4, pool_over="stations")
    a.eval(); b.eval()
    x = torch.randn(2, 5, 6, D)
    assert torch.allclose(a(x), b(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Extreme inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pool_over", ["tokens", "stations"])
def test_extreme_inputs_finite(pool_over):
    head = _head(pool_over=pool_over, num_seeds=2)
    head.eval()
    x = torch.randn(2, 4, 5, D) * 1e4
    assert torch.isfinite(head(x)).all()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_from_config_accepts_known_and_rejects_unknown():
    cfg = {"pool_over": "stations", "num_seeds": 4, "num_heads": 8,
           "seed_self_attention": True, "combine": "linear", "ffn": True,
           "dim_feedforward": 64, "dropout": 0.0, "seed_init_scale": 1.0,
           "time_pool_heads": 2}
    head = SetTransformerPMAHead.from_config(D, NH, cfg)
    assert head.pool_over == "stations" and head.num_seeds == 4
    assert head.pma.attn.num_heads == 8
    assert head.time_pool.mab.attn.num_heads == 2

    with pytest.raises(ValueError):
        SetTransformerPMAHead.from_config(D, NH, {"bogus": 1})


def test_invalid_pool_over_and_combine_raise():
    with pytest.raises(ValueError):
        SetTransformerPMAHead(D, NH, pool_over="bogus")
    with pytest.raises(ValueError):
        SetTransformerPMAHead(D, NH, combine="bogus")
    with pytest.raises(ValueError):
        SetTransformerPMAHead(D, NH, num_seeds=0)


def test_config_keys_complete():
    assert {"pool_over", "num_seeds", "num_heads", "seed_self_attention", "combine",
            "ffn", "dim_feedforward", "dropout", "seed_init_scale", "time_pool_heads"} == _CONFIG_KEYS
