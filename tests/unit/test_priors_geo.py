"""Unit tests for the local equirectangular geo conversions."""

import numpy as np
import pytest

from seismo_sbi.priors.geo import (
    KM_PER_DEG,
    km_offsets_to_latlon,
    latlon_to_km_offsets,
)


def test_one_degree_latitude_is_km_per_deg():
    dx, dy = latlon_to_km_offsets(37.0, 25.0, 36.0, 25.0)
    assert np.isclose(dy, KM_PER_DEG)
    assert np.isclose(dx, 0.0)


def test_one_degree_longitude_scaled_by_cos_lat():
    ref_lat = 36.0
    dx, dy = latlon_to_km_offsets(36.0, 26.0, ref_lat, 25.0)
    assert np.isclose(dx, KM_PER_DEG * np.cos(np.radians(ref_lat)))
    assert np.isclose(dy, 0.0)


def test_round_trip_scalar_and_array():
    ref_lat, ref_lon = 36.4, 25.5
    lat = np.array([36.1, 36.4, 36.9, 36.5])
    lon = np.array([25.2, 25.5, 25.8, 25.4])
    dx, dy = latlon_to_km_offsets(lat, lon, ref_lat, ref_lon)
    lat2, lon2 = km_offsets_to_latlon(dx, dy, ref_lat, ref_lon)
    assert np.allclose(lat, lat2, atol=1e-9)
    assert np.allclose(lon, lon2, atol=1e-9)


def test_corner_plot_km_axes_use_the_flat_earth_scale_at_the_box_centre():
    from seismo_sbi.sbi.types.parameter_labels import DegreeKMConverter, DegreeType

    centre_latitude_deg = 64.65
    to_north_km = DegreeKMConverter(64.5, DegreeType.LATITUDE, centre_latitude_deg)
    to_east_km = DegreeKMConverter(-17.6, DegreeType.LONGITUDE, centre_latitude_deg)
    east_km, north_km = latlon_to_km_offsets(64.6, -17.1, centre_latitude_deg, -17.6)

    assert to_north_km(64.6) == pytest.approx(0.1 * KM_PER_DEG)
    assert to_east_km(-17.1) == pytest.approx(east_km)
