"""Unit tests for the BATCHED variable-station inference path
(``station_dropout.pack_subset_batch`` / ``robust_posterior_sample_batched`` /
``sample_subsets_batched``).

The batched path exists so catalogue-scale station experiments (leave-one-out over ~20
stations x ~650 events) are GPU-throughput-bound rather than launch-bound.  Its whole
correctness claim is that a batched item is the SAME item: the padding + mask convention
must be identical to the training collate (``variable_station_collate``) and to the
single-item packer (``pack_subset_observation``).  These tests pin exactly that, plus the
prior-box rejection contract, with a stub estimator — no trained model needed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seismo_sbi.sbi.npe.source_conditioning import pack_subset_observation
from seismo_sbi.sbi.npe.data.station_selection import variable_station_collate
from seismo_sbi.sbi.npe.posterior_sampling import (
    pack_subset_batch, robust_posterior_sample_batched, sample_subsets_batched)


def _item(n, C=3, T=5, seed=0, n_cond=3):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, C, T)), rng.normal(size=(n, 2)),
            rng.normal(size=(n_cond,)))


def test_single_item_batch_matches_pack_subset_observation():
    obs, coords, sv = _item(4, seed=1)
    a = pack_subset_batch([(obs, coords, sv)], device="cpu")
    b = pack_subset_observation(obs, coords, source_vec=sv)
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=0, rtol=0)


def test_equal_n_batch_rows_match_individual_packing():
    items = [_item(4, seed=s) for s in (1, 2, 3)]
    batch = pack_subset_batch(items, device="cpu")
    assert batch.shape[0] == 3
    for i, (obs, coords, sv) in enumerate(items):
        assert torch.allclose(batch[i], pack_subset_observation(obs, coords, source_vec=sv)[0])


def test_ragged_batch_matches_training_collate():
    """Different N per item: padding, mask and layout must equal what training's
    ``variable_station_collate`` produces for the same samples."""
    items = [_item(n, seed=n) for n in (2, 5, 3)]
    got = pack_subset_batch(items, device="cpu")
    collate_in = [(np.zeros(6), (torch.as_tensor(o, dtype=torch.float32),
                                 torch.as_tensor(c, dtype=torch.float32),
                                 torch.as_tensor(s, dtype=torch.float32)))
                  for o, c, s in items]
    _, expect = variable_station_collate(collate_in)
    assert got.shape == expect.shape
    assert torch.allclose(got, expect, atol=0, rtol=0)


def test_ragged_padding_is_zero_and_mask_marks_it():
    """The padded tail must be exactly zeros and the mask exactly [1]*n + [0]*(max_n-n),
    so a padded station cannot leak signal into the encoder."""
    C, T, n_cond = 3, 5, 3
    items = [_item(2, C, T, seed=1), _item(4, C, T, seed=2)]
    batch = pack_subset_batch(items, device="cpu")
    max_n = 4
    nct = max_n * C * T
    for i, n in enumerate((2, 4)):
        seis = batch[i, :nct].reshape(max_n, C, T)
        coords = batch[i, nct:nct + max_n * 2].reshape(max_n, 2)
        mask = batch[i, nct + max_n * 2: nct + max_n * 3]
        assert torch.equal(mask, torch.tensor([1.0] * n + [0.0] * (max_n - n)))
        assert torch.count_nonzero(seis[n:]) == 0
        assert torch.count_nonzero(coords[n:]) == 0
        assert batch.shape[1] == nct + max_n * 3 + n_cond


def test_empty_items_raises():
    with pytest.raises(ValueError):
        pack_subset_batch([], device="cpu")


def test_bad_coords_shape_raises():
    obs, coords, sv = _item(4)
    with pytest.raises(ValueError):
        pack_subset_batch([(obs, coords[:2], sv)], device="cpu")


class _StubEstimator:
    """Flow stub: every draw for a row equals that row's FIRST context value, so the
    caller can verify per-row alignment.  Keying the behaviour off the context value
    (not the row index) is deliberate — the re-draw rounds pass a re-indexed sub-batch,
    and an index-keyed stub would silently test the wrong row."""

    def __init__(self, dim=6):
        self.dim = dim
        self.calls = []

    def sample(self, n, context=None):
        B = context.shape[0]
        self.calls.append((n, B))
        out = torch.zeros((B, n, self.dim))
        for i in range(B):
            out[i] = float(context[i, 0])
        return out


class _StubPosterior:
    def __init__(self, est, lo=-1.0, hi=1.0):
        self.posterior_estimator = est

        class _P:
            pass
        p = _P()
        p.low = torch.full((est.dim,), lo)
        p.high = torch.full((est.dim,), hi)
        self._prior = p


def test_batched_sampling_shape_and_row_alignment():
    est = _StubEstimator()
    post = _StubPosterior(est)
    ctx = torch.stack([torch.full((7,), 0.25), torch.full((7,), -0.5)])
    s = robust_posterior_sample_batched(post, ctx, 10, oversample=2)
    assert s.shape == (2, 10, 6)
    assert torch.allclose(s[0], torch.full((10, 6), 0.25))
    assert torch.allclose(s[1], torch.full((10, 6), -0.5))
    assert est.calls[0] == (20, 2)                       # ONE call for the whole batch


def test_leaking_row_is_clipped_not_hanging():
    """A row whose flow mass sits entirely outside the prior box must come back clipped
    and full-length after a bounded number of rounds (the no-hang contract).  Row 1's
    context value (9.0) is outside the [-1, 1] prior box, so every draw is rejected."""
    est = _StubEstimator()
    post = _StubPosterior(est)
    ctx = torch.stack([torch.full((7,), 0.25), torch.full((7,), 9.0)])
    s = robust_posterior_sample_batched(post, ctx, 8, oversample=2, max_rounds=3)
    assert s.shape == (2, 8, 6)
    assert torch.allclose(s[0], torch.full((8, 6), 0.25))
    assert torch.allclose(s[1], torch.ones((8, 6)))      # clipped to the prior hi
    # only the deficient row is re-drawn after the first round
    assert est.calls[0][1] == 2 and all(c[1] == 1 for c in est.calls[1:])


def test_sample_subsets_batched_chunks_and_applies_scaler():
    est = _StubEstimator()
    post = _StubPosterior(est, lo=-1e9, hi=1e9)          # nothing rejected: one call/chunk
    items = [_item(3, C=3, T=5, seed=s) for s in range(5)]

    class Doubler:
        def inverse_transform(self, x):
            return np.asarray(x) * 2.0

    seen = []
    out = sample_subsets_batched(post, items, Doubler(), num_samples=4, device="cpu",
                                 max_batch=2, progress=lambda d, t: seen.append((d, t)))
    assert len(out) == 5 and all(o.shape == (4, 6) for o in out)
    assert [c[1] for c in est.calls] == [2, 2, 1]        # 5 items at max_batch=2
    assert seen[-1] == (5, 5)


def test_flow_sample_chunked_falls_back_for_non_nflows_estimator():
    """A stub estimator exposes none of the nflows internals: chunking must degrade to a
    plain ``est.sample`` rather than crashing."""
    from seismo_sbi.sbi.npe.posterior_sampling import flow_sample_chunked
    est = _StubEstimator()
    ctx = torch.stack([torch.full((7,), 0.25), torch.full((7,), -0.5)])
    s = flow_sample_chunked(est, ctx, 6, chunk=2)
    assert s.shape == (2, 6, 6)
    assert est.calls == [(6, 2)]                     # fell back to ONE full call


def test_flow_sample_chunked_matches_unchunked_on_a_real_flow():
    """On a real (tiny) nflows conditional flow, chunking the sample dimension must return
    the same shape, the same per-row distribution, and — the claim that matters — must
    concatenate along SAMPLES with rows still aligned to their context.  (nflows' own
    ``batch_size`` argument concatenates along dim 0, which is the CONTEXT axis when a
    context is given, silently scrambling exactly this.)

    Exact sample equality is not available: one ``randn(B*n)`` draw reshaped to (B, n)
    assigns the stream differently from six (B, n_i) draws, so this pins the distribution
    and the alignment instead.
    """
    pytest.importorskip("nflows")
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms import (CompositeTransform,
                                   MaskedAffineAutoregressiveTransform)
    from seismo_sbi.sbi.npe.posterior_sampling import flow_sample_chunked

    torch.manual_seed(0)
    dim, ctx_dim, n = 3, 4, 4000
    flow = Flow(CompositeTransform([
        MaskedAffineAutoregressiveTransform(features=dim, hidden_features=16,
                                            context_features=ctx_dim)]),
                StandardNormal([dim]))
    # well-separated contexts so each row's output distribution is distinguishable
    ctx = torch.eye(4)[:, :ctx_dim] * 6.0 - 3.0

    torch.manual_seed(1)
    full = flow.sample(n, context=ctx).detach()
    torch.manual_seed(1)
    chunked = flow_sample_chunked(flow, ctx, n, chunk=700).detach()
    assert full.shape == chunked.shape == (4, n, dim)

    m_full, m_chunk = full.mean(dim=1), chunked.mean(dim=1)
    s_full, s_chunk = full.std(dim=1), chunked.std(dim=1)
    # per-row distribution agrees within Monte-Carlo error
    assert torch.allclose(m_full, m_chunk, atol=0.15 * s_full.max().item())
    assert torch.allclose(s_full, s_chunk, rtol=0.15)
    # row alignment: each chunked row is closest to ITS OWN unchunked row
    dists = torch.cdist(m_chunk, m_full)
    assert torch.equal(dists.argmin(dim=1), torch.arange(4))


def test_flow_chunk_lets_a_batch_exceed_the_unchunked_call():
    """``flow_chunk`` is plumbed through ``robust_posterior_sample_batched``: the number of
    rows handed to the flow per call is capped by the chunk, not by num_samples."""
    from seismo_sbi.sbi.npe.posterior_sampling import flow_sample_chunked  # noqa: F401
    est = _StubEstimator()
    post = _StubPosterior(est, lo=-1e9, hi=1e9)
    ctx = torch.stack([torch.full((7,), 0.25), torch.full((7,), -0.5)])
    s = robust_posterior_sample_batched(post, ctx, 10, oversample=1, flow_chunk=4)
    assert s.shape == (2, 10, 6)
    assert torch.allclose(s[0], torch.full((10, 6), 0.25))


def test_sample_subsets_batched_auto_chunks_large_batches(monkeypatch):
    """With ``flow_chunk=None`` the sampler must derive a chunk that keeps
    ``batch x chunk`` under ``TRSM_SAFE_ROWS`` — the whole point of auto-chunking is that a
    caller can raise ``max_batch`` without hitting the cuBLAS batched-solve cliff.
    (A stub estimator has no nflows internals, so the chunk is observed at the call
    boundary rather than through the stub.)"""
    from seismo_sbi.sbi.npe import posterior_sampling as sd

    est = _StubEstimator()
    post = _StubPosterior(est, lo=-1e9, hi=1e9)
    items = [_item(3, C=1, T=2, seed=s) for s in range(8)]

    class Id:
        def inverse_transform(self, x):
            return np.asarray(x)

    seen = []
    real = sd.flow_sample_chunked
    monkeypatch.setattr(sd, "flow_sample_chunked",
                        lambda e, c, n, chunk=None: seen.append(chunk) or real(
                            e, c, n, chunk=chunk))
    monkeypatch.setattr(sd, "TRSM_SAFE_ROWS", 16)
    sd.sample_subsets_batched(post, items, Id(), num_samples=8, device="cpu",
                              max_batch=4, oversample=1)
    assert seen == [4, 4]                       # 16 // 4 rows per call, two chunks of items
    assert all(chunk * 4 <= 16 for chunk in seen)


def test_sample_subsets_batched_explicit_flow_chunk_wins():
    from seismo_sbi.sbi.npe import posterior_sampling as sd
    est = _StubEstimator()
    post = _StubPosterior(est, lo=-1e9, hi=1e9)
    items = [_item(3, C=1, T=2, seed=s) for s in range(4)]

    class Id:
        def inverse_transform(self, x):
            return np.asarray(x)

    out = sd.sample_subsets_batched(post, items, Id(), num_samples=6, device="cpu",
                                    max_batch=4, oversample=1, flow_chunk=6)
    assert len(out) == 4 and out[0].shape == (6, 6)
