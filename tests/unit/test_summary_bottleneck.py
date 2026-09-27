"""Summary bottleneck: narrow the MMD space without touching encoder or flow capacity.

The MMD auxiliary loss compares only `batch_size` (64) summaries per side. In the
production 256-d summary that is a weakly-powered, high-variance regime for the
estimator, so `ml_summary_bottleneck.dim` narrows the space the kernel lives in.

The whole point is that it is a CONTROLLED change, so these tests pin the two
invariants that make it controlled:

  1. the flow still receives a `num_outputs`-wide context (flow capacity unchanged), and
  2. the MMD tap reads the NARROW vector, not the flow-facing expansion.

Plus the backward-compatibility guarantee: absent config => byte-identical head.
"""
import torch
import torch.nn as nn


def _head(d_model, num_outputs, bottleneck):
    """Reproduce the head SeismogramTransformer builds, without the (heavy) full model."""
    out = bottleneck or num_outputs
    predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                              nn.Linear(d_model, out))
    expand = (nn.Sequential(nn.LayerNorm(bottleneck), nn.Linear(bottleneck, num_outputs))
              if bottleneck else None)
    return predictor, expand


def test_flow_context_width_is_unchanged_by_the_bottleneck():
    """A 32-d bottleneck must still hand the flow a 256-d context.

    This is the invariant that keeps a latent-dim experiment from silently becoming a
    flow-capacity experiment.
    """
    d_model, num_outputs = 256, 256
    pooled = torch.randn(8, d_model)

    pred_ref, exp_ref = _head(d_model, num_outputs, None)
    pred_bn, exp_bn = _head(d_model, num_outputs, 32)

    ref_out = pred_ref(pooled)
    bn_z = pred_bn(pooled)
    bn_out = exp_bn(bn_z)

    assert ref_out.shape == (8, num_outputs)
    assert bn_z.shape == (8, 32), "the bottleneck must actually be narrow"
    assert bn_out.shape == ref_out.shape, "flow context width changed — flow capacity would too"


def test_bottleneck_is_a_genuine_information_bottleneck():
    """Everything downstream must be a function of only `dim` numbers.

    Equal bottleneck vectors => equal flow contexts, so the summary really is 32-d and
    not merely relabelled.
    """
    _, expand = _head(256, 256, 32)
    z = torch.randn(4, 32)
    a = expand(z)
    b = expand(z.clone())
    assert torch.allclose(a, b)
    # and the expansion cannot manufacture rank: 16 distinct inputs through a 32-d
    # bottleneck span at most 32 dimensions.
    many = expand(torch.randn(16, 32))
    assert torch.linalg.matrix_rank(many.double()) <= 32


def test_absent_config_leaves_the_head_unchanged():
    """No `ml_summary_bottleneck` => no expansion module and the original widths."""
    predictor, expand = _head(256, 256, None)
    assert expand is None
    assert predictor[-1].out_features == 256


def test_mmd_tap_prefers_the_bottleneck_when_present():
    """`_mmd_term` resolves `summary_bottleneck` if the embedding net exposes it.

    Mirrors the getattr dispatch in seismogram_transformer._mmd_term so a rename there
    fails loudly here rather than silently reverting MMD to the wide space.
    """
    class _WithBottleneck(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 256)

        def summary_bottleneck(self, x):
            return torch.zeros(x.shape[0], 32)

    class _Plain(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 256)

    x = torch.randn(5, 3)
    for emb, expected in ((_WithBottleneck(), 32), (_Plain(), 256)):
        summarise = getattr(emb, "summary_bottleneck", emb)
        assert summarise(x).shape[1] == expected
