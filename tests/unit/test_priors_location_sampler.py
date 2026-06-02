"""Unit tests for the catalogue source-location (KDE) sampler."""

import numpy as np

from seismo_sbi.priors.catalogue import EventCatalogue
from seismo_sbi.priors.geo import latlon_to_km_offsets
from seismo_sbi.priors.samplers import make_catalogue_location_sampler

# Wide bounds so clipping does not interfere with the spread tests.
WIDE_BOUNDS = np.array([[35.0, 24.0, 0.0, -10.0], [38.0, 27.0, 60.0, 10.0]])


def _single_event_catalogue(lat, lon, depth):
    return EventCatalogue(
        latitude=np.array([lat]),
        longitude=np.array([lon]),
        depth=np.array([depth]),
        magnitude=np.array([3.0]),
        magnitude_type=np.array(["Ml"], dtype=object),
    )


def test_shapes_and_within_bounds():
    cat = _single_event_catalogue(36.4, 25.5, 12.0)
    sampler = make_catalogue_location_sampler(
        catalogue=cat, std_x_km=2.0, std_y_km=2.0, std_z_km=3.0, seed=0
    )
    samples = np.array(list(sampler(WIDE_BOUNDS, 5000)))
    assert samples.shape == (5000, 4)
    assert np.all(samples >= WIDE_BOUNDS[0]) and np.all(samples <= WIDE_BOUNDS[1])


def test_horizontal_and_depth_spread_match_std():
    lat0, lon0, depth0 = 36.4, 25.5, 20.0
    cat = _single_event_catalogue(lat0, lon0, depth0)
    sampler = make_catalogue_location_sampler(
        catalogue=cat, std_x_km=3.0, std_y_km=4.0, std_z_km=2.0, seed=1
    )
    s = np.array(list(sampler(WIDE_BOUNDS, 40_000)))
    dx_km, dy_km = latlon_to_km_offsets(s[:, 0], s[:, 1], lat0, lon0)
    assert np.isclose(dx_km.std(), 3.0, atol=0.1)
    assert np.isclose(dy_km.std(), 4.0, atol=0.1)
    assert np.isclose(s[:, 2].std(), 2.0, atol=0.1)
    # mean stays at the event location
    assert abs(s[:, 0].mean() - lat0) < 1e-2
    assert abs(s[:, 2].mean() - depth0) < 0.1


def test_depth_clipped_at_surface():
    cat = _single_event_catalogue(36.4, 25.5, 0.5)
    sampler = make_catalogue_location_sampler(
        catalogue=cat, std_x_km=1.0, std_y_km=1.0, std_z_km=5.0, seed=2
    )
    s = np.array(list(sampler(WIDE_BOUNDS, 20_000)))
    assert s[:, 2].min() >= 0.0


def test_mean_tracks_catalogue_mixture():
    rng = np.random.default_rng(3)
    lats = rng.uniform(36.2, 36.8, 200)
    lons = rng.uniform(25.3, 25.7, 200)
    depths = rng.uniform(5, 15, 200)
    cat = EventCatalogue(lats, lons, depths, np.full(200, 3.0),
                         np.array(["Ml"] * 200, dtype=object))
    sampler = make_catalogue_location_sampler(
        catalogue=cat, std_x_km=0.5, std_y_km=0.5, std_z_km=0.5, seed=4
    )
    s = np.array(list(sampler(WIDE_BOUNDS, 40_000)))
    assert abs(s[:, 0].mean() - lats.mean()) < 0.02
    assert abs(s[:, 1].mean() - lons.mean()) < 0.02


def test_time_shift_constant_vs_uniform():
    cat = _single_event_catalogue(36.4, 25.5, 12.0)
    const = make_catalogue_location_sampler(
        catalogue=cat, std_x_km=1.0, std_y_km=1.0, std_z_km=1.0,
        time_shift="constant", seed=5)
    s_const = np.array(list(const(WIDE_BOUNDS, 1000)))
    assert np.allclose(s_const[:, 3], WIDE_BOUNDS[0, 3])

    unif = make_catalogue_location_sampler(
        catalogue=cat, std_x_km=1.0, std_y_km=1.0, std_z_km=1.0,
        time_shift="uniform", seed=6)
    s_unif = np.array(list(unif(WIDE_BOUNDS, 5000)))
    assert s_unif[:, 3].min() >= WIDE_BOUNDS[0, 3]
    assert s_unif[:, 3].max() <= WIDE_BOUNDS[1, 3]
    assert s_unif[:, 3].std() > 1.0  # genuinely varying


def test_use_event_errors_broadens_spread():
    cat = EventCatalogue(
        latitude=np.array([36.4]), longitude=np.array([25.5]),
        depth=np.array([20.0]), magnitude=np.array([3.0]),
        magnitude_type=np.array(["Ml"], dtype=object),
        err_h=np.array([5.0]), err_z=np.array([4.0]),
    )
    sampler = make_catalogue_location_sampler(
        catalogue=cat, std_x_km=3.0, std_y_km=3.0, std_z_km=2.0,
        use_event_errors=True, seed=7)
    s = np.array(list(sampler(WIDE_BOUNDS, 40_000)))
    # depth std = sqrt(2^2 + 4^2) ~ 4.47
    assert np.isclose(s[:, 2].std(), np.hypot(2.0, 4.0), atol=0.15)
