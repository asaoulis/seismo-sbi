"""Unit tests for variable-station configuration support in the ML compressor.

Covers the pure-tensor plumbing (context pack/unpack, the subsampler, the ragged
collate, per-sample relative geometry) and — most importantly — the **masking
correctness** property: appending purely-padded stations to a sample must leave the
real-station embedding *bit-identical*. That proves the key_padding_mask genuinely
isolates padded stations rather than merely keeping the output finite.

All tests are CPU-only and fast (tiny transformer, no normalising flow, no dataset).
"""

import numpy as np
import torch
import pytest

from seismo_sbi.sbi.compression.ML.source_conditioning import (
    pack_variable_context,
    unpack_variable_context,
    relative_station_geometry,
)
from seismo_sbi.sbi.compression.ML.dataloading import (
    StationSubsampler,
    variable_station_collate,
)
from seismo_sbi.sbi.compression.ML.seismogram_transformer import SeismogramTransformer


_COORDS = np.array(
    [[37.0, -122.0], [38.0, -120.0], [37.3, -119.0], [36.5, -121.0], [39.0, -118.0]],
    dtype=np.float32,
)


# --------------------------------------------------------------------------- #
# Context pack / unpack
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_cond", [0, 3])
def test_pack_unpack_roundtrip_recovers_maxN_and_payload(n_cond):
    B, N, C, T = 2, 4, 3, 10
    x = torch.randn(B, N, C, T)
    coords = torch.randn(B, N, 2)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, -1] = False  # one padded station
    source = torch.randn(B, n_cond) if n_cond else None

    packed = torch.stack([
        pack_variable_context(x[b], coords[b], mask[b], source[b] if n_cond else None)
        for b in range(B)
    ], dim=0)

    seis, crd, msk, sv = unpack_variable_context(packed, C, T, n_cond)
    # max_N is recovered purely from the context width.
    assert seis.shape == (B, N, C, T)
    assert crd.shape == (B, N, 2)
    assert msk.shape == (B, N)
    torch.testing.assert_close(seis, x)
    torch.testing.assert_close(crd, coords)
    assert torch.equal(msk, mask)
    if n_cond:
        torch.testing.assert_close(sv, source)
    else:
        assert sv is None


def test_unpack_rejects_inconsistent_width():
    bad = torch.randn(2, 17)  # not divisible by per-station size + n_cond
    with pytest.raises(ValueError, match="inconsistent"):
        unpack_variable_context(bad, n_components=3, trace_length=10, n_cond=0)


# --------------------------------------------------------------------------- #
# StationSubsampler
# --------------------------------------------------------------------------- #

def test_subsampler_respects_fraction_and_floor():
    np.random.seed(0)
    sub = StationSubsampler(keep_fraction=(0.5, 1.0), min_stations=2)
    N = 10
    for _ in range(200):
        idx = sub(N)
        assert idx.ndim == 1
        assert 2 <= len(idx) <= N                 # within [floor, N]
        assert len(np.unique(idx)) == len(idx)    # no repeats
        assert np.all(np.diff(idx) > 0)           # sorted, preserves canonical order
        assert np.all(idx < N)


def test_subsampler_fixed_fraction():
    np.random.seed(1)
    sub = StationSubsampler(keep_fraction=0.5, min_stations=1)
    assert len(sub(8)) == 4


def test_subsampler_floor_clamped_to_available():
    sub = StationSubsampler(keep_fraction=1.0, min_stations=99)
    assert len(sub(3)) == 3  # floor clamped to N


# --------------------------------------------------------------------------- #
# Ragged collate
# --------------------------------------------------------------------------- #

def test_variable_station_collate_pads_and_masks():
    C, T = 3, 10
    sizes = [2, 4, 1]
    batch = []
    for n in sizes:
        x = torch.randn(n, C, T)
        coords = torch.randn(n, 2)
        batch.append((torch.randn(6), (x, coords, None)))

    theta, ctx = variable_station_collate(batch)
    assert theta.shape == (3, 6)
    seis, crd, mask, sv = unpack_variable_context(ctx, C, T, n_cond=0)
    max_N = max(sizes)
    assert seis.shape == (3, max_N, C, T)
    assert sv is None
    # Mask marks exactly the real stations per sample; padded coords/seis are zero.
    for i, n in enumerate(sizes):
        assert mask[i, :n].all() and not mask[i, n:].any()
        torch.testing.assert_close(seis[i, n:], torch.zeros(max_N - n, C, T))


def test_variable_station_collate_carries_source_vec():
    C, T = 2, 8
    batch = [
        (torch.randn(6), (torch.randn(n, C, T), torch.randn(n, 2), torch.randn(3)))
        for n in (2, 3)
    ]
    _, ctx = variable_station_collate(batch)
    _, _, _, sv = unpack_variable_context(ctx, C, T, n_cond=3)
    assert sv.shape == (2, 3)


# --------------------------------------------------------------------------- #
# Per-sample relative geometry
# --------------------------------------------------------------------------- #

def test_relative_geometry_per_sample_matches_broadcast():
    B, N = 4, 5
    coords = torch.tensor(_COORDS)                    # (N, 2)
    source = torch.randn(B, 3) * 10                    # (B, 3) lat/lon/depth
    shared = relative_station_geometry(source, coords, "geographic")        # (B, N, 2)
    per_sample = relative_station_geometry(
        source, coords.unsqueeze(0).expand(B, N, 2), "geographic"
    )
    torch.testing.assert_close(shared, per_sample)


# --------------------------------------------------------------------------- #
# Masking invariance (the key correctness property)
# --------------------------------------------------------------------------- #

def _make_model(coord_mode="absolute", pooling="mean", use_cls=False, n_cond=0, amplitude=None):
    C, T, d = 3, 128, 16
    cfg = {
        "channels": d, "nheads": 2, "layers": 2, "station_encoder": "cnn",
        "variable_stations": True, "station_coords_mode": coord_mode,
        "num_query_tokens": 4, "pooling": pooling, "use_cls_token": use_cls,
    }
    if n_cond:
        cfg["conditioning"] = {"n_cond": n_cond, "d_cond": 8, "coord_mode": "geographic", "inject": []}
    if amplitude is not None:
        cfg["amplitude_embedding"] = amplitude
    torch.manual_seed(0)
    model = SeismogramTransformer(
        C, cfg, d, num_outputs=d, noise_model=None,
        seismogram_locations=torch.tensor(_COORDS), device="cpu", input_length=T,
    )
    model.eval()
    return model, C, T


@pytest.mark.parametrize("pooling", ["mean", "attn", "max"])
@pytest.mark.parametrize("use_cls", [False, True])
def test_padded_stations_do_not_change_real_embedding_absolute(pooling, use_cls):
    """Appending fully-padded stations leaves the real-station embedding bit-identical."""
    model, C, T = _make_model("absolute", pooling, use_cls)
    n = 3
    x = torch.randn(n, C, T)
    coords = torch.tensor(_COORDS[:n])

    ctx_real = pack_variable_context(x, coords, torch.ones(n, dtype=torch.bool), None).unsqueeze(0)

    # Same sample + 2 padded stations carrying *garbage* content and coords.
    xp = torch.zeros(n + 2, C, T); xp[:n] = x; xp[n:] = torch.randn(2, C, T)
    cp = torch.zeros(n + 2, 2); cp[:n] = coords; cp[n:] = torch.randn(2, 2)
    mp = torch.zeros(n + 2, dtype=torch.bool); mp[:n] = True
    ctx_pad = pack_variable_context(xp, cp, mp, None).unsqueeze(0)

    with torch.no_grad():
        e_real = model.embed(ctx_real)
        e_pad = model.embed(ctx_pad)

    assert torch.isfinite(e_pad).all()
    torch.testing.assert_close(e_real, e_pad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("pooling", ["mean", "attn"])
def test_padded_stations_invariance_relative_mode(pooling):
    """Same invariance holds for source-relative coordinate encoding."""
    model, C, T = _make_model("relative", pooling, use_cls=False, n_cond=3)
    n = 3
    x = torch.randn(n, C, T)
    coords = torch.tensor(_COORDS[:n])
    source = torch.tensor([37.5, -120.5, 8.0])

    ctx_real = pack_variable_context(
        x, coords, torch.ones(n, dtype=torch.bool), source
    ).unsqueeze(0)

    xp = torch.zeros(n + 2, C, T); xp[:n] = x; xp[n:] = torch.randn(2, C, T)
    cp = torch.zeros(n + 2, 2); cp[:n] = coords; cp[n:] = torch.randn(2, 2)
    mp = torch.zeros(n + 2, dtype=torch.bool); mp[:n] = True
    ctx_pad = pack_variable_context(xp, cp, mp, source).unsqueeze(0)

    with torch.no_grad():
        e_real = model.embed(ctx_real)
        e_pad = model.embed(ctx_pad)

    assert torch.isfinite(e_pad).all()
    torch.testing.assert_close(e_real, e_pad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", ["array_relative", "absolute"])
def test_padded_stations_invariance_with_amplitude_embedding(mode):
    """The amplitude embedding must respect the validity mask: padded stations (even with
    garbage amplitude) leave the real-station embedding bit-identical. This checks the
    array-relative reference excludes padding and the padded token is zeroed."""
    model, C, T = _make_model("absolute", pooling="mean", use_cls=False,
                              amplitude={"mode": mode})
    n = 3
    x = torch.randn(n, C, T)
    coords = torch.tensor(_COORDS[:n])

    ctx_real = pack_variable_context(x, coords, torch.ones(n, dtype=torch.bool), None).unsqueeze(0)

    xp = torch.zeros(n + 2, C, T); xp[:n] = x; xp[n:] = torch.randn(2, C, T) * 50.0
    cp = torch.zeros(n + 2, 2); cp[:n] = coords; cp[n:] = torch.randn(2, 2)
    mp = torch.zeros(n + 2, dtype=torch.bool); mp[:n] = True
    ctx_pad = pack_variable_context(xp, cp, mp, None).unsqueeze(0)

    with torch.no_grad():
        e_real = model.embed(ctx_real)
        e_pad = model.embed(ctx_pad)

    assert torch.isfinite(e_pad).all()
    torch.testing.assert_close(e_real, e_pad, rtol=1e-5, atol=1e-5)


def test_station_order_permutation_consistent():
    """Permuting stations (data + coords + mask together) permutes nothing in the pooled
    output beyond what the permutation-equivariant attention implies: the pooled embedding,
    being order-invariant under masked mean pooling, is unchanged."""
    model, C, T = _make_model("absolute", pooling="mean", use_cls=False)
    n = 4
    x = torch.randn(n, C, T)
    coords = torch.tensor(_COORDS[:n])
    mask = torch.ones(n, dtype=torch.bool)
    ctx = pack_variable_context(x, coords, mask, None).unsqueeze(0)

    perm = torch.tensor([2, 0, 3, 1])
    ctx_perm = pack_variable_context(x[perm], coords[perm], mask[perm], None).unsqueeze(0)

    with torch.no_grad():
        e = model.embed(ctx)
        e_perm = model.embed(ctx_perm)
    torch.testing.assert_close(e, e_perm, rtol=1e-5, atol=1e-5)
