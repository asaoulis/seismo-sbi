"""Unit tests for the pure source-conditioning primitives.

Covers geometry (known azimuth/distance cases), the source embedding, FiLM identity
init, and the pack/unpack context round-trip. No Instaseis/CPS, no disk I/O.
"""

import math

import pytest
import torch

from seismo_sbi.sbi.compression.ML.source_conditioning import (
    relative_station_geometry,
    SourceConditioner,
    FiLM,
    pack_context,
    unpack_context,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# relative_station_geometry — geographic
# ---------------------------------------------------------------------------

def test_geographic_azimuth_due_north_and_east():
    # Source at the equator/prime meridian.
    source = torch.tensor([[0.0, 0.0]])
    # Station due north (+lat) and due east (+lon), both close by.
    stations = torch.tensor([[1.0, 0.0],    # north  → bearing 0
                             [0.0, 1.0]])    # east   → bearing pi/2
    geom = relative_station_geometry(source, stations, coord_mode="geographic")
    assert geom.shape == (1, 2, 2)
    az = geom[0, :, 1]
    assert az[0].abs().item() < 1e-4                       # due north ≈ 0
    assert abs(az[1].item() - math.pi / 2) < 1e-4          # due east ≈ +pi/2


def test_geographic_distance_is_central_angle():
    source = torch.tensor([[0.0, 0.0]])
    # 1 degree north → central angle of 1 degree in radians.
    stations = torch.tensor([[1.0, 0.0]])
    dist = relative_station_geometry(source, stations, "geographic")[0, 0, 0]
    assert abs(dist.item() - math.radians(1.0)) < 1e-6


def test_geographic_zero_distance_when_colocated():
    source = torch.tensor([[12.3, -45.6]])
    stations = torch.tensor([[12.3, -45.6]])
    dist = relative_station_geometry(source, stations, "geographic")[0, 0, 0]
    assert dist.item() < 1e-6


# ---------------------------------------------------------------------------
# relative_station_geometry — cartesian
# ---------------------------------------------------------------------------

def test_cartesian_distance_and_azimuth():
    source = torch.tensor([[0.0, 0.0]])
    stations = torch.tensor([[3.0, 4.0],     # dist 5, az atan2(4,3)
                             [0.0, 2.0]])     # dist 2, az pi/2
    geom = relative_station_geometry(source, stations, "cartesian")
    assert abs(geom[0, 0, 0].item() - 5.0) < 1e-6
    assert abs(geom[0, 0, 1].item() - math.atan2(4.0, 3.0)) < 1e-6
    assert abs(geom[0, 1, 0].item() - 2.0) < 1e-6
    assert abs(geom[0, 1, 1].item() - math.pi / 2) < 1e-6


def test_geometry_batch_shape_and_depth_ignored():
    # Two sources with differing depth (3rd coord) — must not change epicentral geometry.
    source = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 99.0]])
    stations = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    geom = relative_station_geometry(source, stations, "geographic")
    assert geom.shape == (2, 3, 2)
    assert torch.allclose(geom[0], geom[1], atol=1e-6)


def test_unknown_coord_mode_raises():
    with pytest.raises(ValueError):
        relative_station_geometry(torch.zeros(1, 2), torch.zeros(2, 2), "spherical")


# ---------------------------------------------------------------------------
# SourceConditioner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_fourier", [0, 3])
def test_source_conditioner_shape_and_finite(n_fourier):
    cond = SourceConditioner(n_cond=3, d_cond=16, coord_mode="geographic", n_fourier=n_fourier)
    source = torch.tensor([[12.0, 34.0, 10.0], [-5.0, 100.0, 50.0]])
    out = cond(source)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_source_conditioner_accepts_unbatched():
    cond = SourceConditioner(n_cond=3, d_cond=8)
    out = cond(torch.tensor([1.0, 2.0, 3.0]))
    assert out.shape == (1, 8)


def test_source_conditioner_rejects_bad_n_cond():
    with pytest.raises(ValueError):
        SourceConditioner(n_cond=0, d_cond=8)


# ---------------------------------------------------------------------------
# FiLM
# ---------------------------------------------------------------------------

def test_film_is_identity_at_init():
    # Zero-initialised → γ=β=0 → output equals input regardless of cond.
    film = FiLM(d_cond=8, d_model=16)
    x = torch.randn(4, 5, 7, 16)          # (B, N, L, d_model)
    cond = torch.randn(4, 8)
    out = film(x, cond)
    assert torch.allclose(out, x, atol=1e-6)


def test_film_modulates_after_perturbation():
    film = FiLM(d_cond=8, d_model=16)
    with torch.no_grad():
        film.to_film.weight.normal_()
        film.to_film.bias.normal_()
    x = torch.randn(2, 3, 16)             # (B, N, d_model) — middle dim broadcast
    cond = torch.randn(2, 8)
    out = film(x, cond)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------

def test_pack_unpack_round_trip():
    B, N, C, T, n_cond = 4, 3, 2, 5, 3
    seis = torch.randn(B, N, C, T)
    source = torch.randn(B, n_cond)
    ctx = pack_context(seis, source)
    assert ctx.shape == (B, N * C * T + n_cond)
    seis2, source2 = unpack_context(ctx, N, C, T, n_cond)
    assert torch.allclose(seis2, seis)
    assert torch.allclose(source2, source)


def test_unpack_passthrough_for_4d_context():
    seis = torch.randn(2, 3, 2, 5)
    out, source = unpack_context(seis, 3, 2, 5, n_cond=0)
    assert source is None
    assert torch.allclose(out, seis)
