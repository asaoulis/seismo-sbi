"""Collapsing a three-dimensional model to one-dimensional profiles along a corridor."""

import numpy as np
import pytest

from seismo_sbi.simulators.axisem.profiles import (
    SOURCE_CODES, CorridorPath, Model3D, build_profile_ensemble, corridor_mask,
    extract_profile, fill_ladder_joint, sample_polyline,
)

DEPTH_KM = np.array([0.0, 1.0, 2.0, 3.0])


def _model(vp=None, vs=None):
    lat = np.linspace(36.0, 36.2, 5)
    lon = np.linspace(25.0, 25.2, 5)
    shape = (len(DEPTH_KM), len(lat), len(lon))
    return Model3D(vp=np.full(shape, 6.0) if vp is None else vp,
                   vs=np.full(shape, 3.0) if vs is None else vs,
                   depth=DEPTH_KM, lat=lat, lon=lon,
                   easting=np.zeros((len(lat), len(lon))),
                   northing=np.zeros((len(lat), len(lon))),
                   utm_epsg=32634)


@pytest.fixture
def model():
    return _model()


def _path(model, halfwidth_m=5000.0):
    sample_lonlat, sample_xy, length_km = sample_polyline(
        model, (25.0, 36.1), (25.2, 36.1), spacing_m=500.0, halfwidth_m=halfwidth_m)
    return CorridorPath(path_id="P0", mode="A", station="STA1", seed=1,
                        p0_lonlat=(25.0, 36.1), p1_lonlat=(25.2, 36.1),
                        sample_lonlat=sample_lonlat, sample_xy=sample_xy,
                        corridor_halfwidth_m=halfwidth_m, length_km=length_km)


def test_polyline_columns_span_the_path_at_the_requested_spacing(model):
    sample_lonlat, sample_xy, length_km = sample_polyline(
        model, (25.0, 36.1), (25.2, 36.1), spacing_m=500.0, halfwidth_m=5000.0)
    assert len(sample_xy) == len(sample_lonlat)
    steps = np.hypot(*np.diff(sample_xy, axis=0).T)
    assert np.all(steps <= 500.0 + 1e-6)
    assert length_km == pytest.approx(np.hypot(*(sample_xy[-1] - sample_xy[0])) / 1000.0)


def test_a_wider_corridor_covers_more_cells(model):
    narrow = corridor_mask(model, _path(model, halfwidth_m=1000.0))
    wide = corridor_mask(model, _path(model, halfwidth_m=20000.0))
    assert narrow.sum() < wide.sum()
    assert np.all(wide[narrow])


def test_a_uniform_model_collapses_to_its_own_wavespeeds(model):
    profile = extract_profile(model, _path(model), np.full(4, 9.9), np.full(4, 9.9))
    assert profile["vp"] == pytest.approx(np.full(4, 6.0))
    assert profile["vs"] == pytest.approx(np.full(4, 3.0))
    assert np.all(profile["source_vp"] == SOURCE_CODES["corridor"])


def test_the_depth_average_is_the_slowness_mean():
    vp = np.full((4, 5, 5), 6.0)
    vp[0, :, :2] = 2.0
    profile = extract_profile(_model(vp=vp), _path(_model(vp=vp)),
                             np.full(4, 9.9), np.full(4, 9.9))
    assert profile["vp"][0] < np.mean([2.0, 2.0, 6.0, 6.0, 6.0])


def test_a_depth_missing_either_wavespeed_is_not_corridor_sourced():
    vs = np.full((4, 5, 5), 3.0)
    vs[1] = np.nan
    profile = extract_profile(_model(vs=vs), _path(_model(vs=vs)),
                             np.full(4, 9.9), np.full(4, 9.9))
    assert profile["source_vp"][1] == SOURCE_CODES["interp"]
    assert profile["source_vp"][1] == profile["source_vs"][1]


def test_an_internal_gap_is_interpolated_in_slowness():
    vp = np.array([4.0, np.nan, np.nan, 8.0])
    vs = vp / 2.0
    source = fill_ladder_joint(vp, vs, DEPTH_KM, np.full(4, 9.9), np.full(4, 9.9))
    assert list(source) == [SOURCE_CODES["corridor"], SOURCE_CODES["interp"],
                            SOURCE_CODES["interp"], SOURCE_CODES["corridor"]]
    assert 1.0 / vp[1] == pytest.approx(np.interp(1.0, [0.0, 3.0], [0.25, 0.125]))


def test_the_ends_fall_back_to_the_reference_profile():
    vp = np.array([np.nan, 4.0, 4.0, np.nan])
    vs = np.array([np.nan, 2.0, 2.0, np.nan])
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    source = fill_ladder_joint(vp, vs, DEPTH_KM, reference, reference)
    assert vp[0] == 1.0 and vp[-1] == 4.0
    assert source[0] == source[-1] == SOURCE_CODES["reference"]


def test_a_corridor_with_no_data_is_entirely_reference_sourced():
    vp = np.full((4, 5, 5), np.nan)
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    model = _model(vp=vp)
    profile = extract_profile(model, _path(model), reference, reference)
    assert np.all(profile["source_vp"] == SOURCE_CODES["reference"])
    assert profile["vp"] == pytest.approx(reference)


def test_the_ensemble_stacks_one_row_per_corridor(model):
    paths = [_path(model, halfwidth_m=20000.0), _path(model, halfwidth_m=1000.0)]
    ensemble = build_profile_ensemble(model, paths, np.full(4, 9.9), np.full(4, 9.9))
    assert ensemble["VP"].shape == (2, 4)
    assert [record["path_id"] for record in ensemble["records"]] == ["P0", "P0"]
    assert ensemble["records"][0]["n_corridor_cells"] > ensemble["records"][1]["n_corridor_cells"]
