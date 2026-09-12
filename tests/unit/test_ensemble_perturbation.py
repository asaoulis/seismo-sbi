"""Perturbing a one-dimensional model: the depth taper, the correlation and the couplings."""
import numpy as np
import pytest

from seismo_sbi.simulators.axisem.perturb import (
    MAX_VS_OVER_VP, control_point_sigma, depth_tapered_sigma, perturb_layered_model,
    perturb_velocity_ratio_model, unit_smooth_field,
)

DEPTH_KM = np.arange(0.0, 100.0, 0.5)


def reference():
    vp = 5.0 + 0.02 * DEPTH_KM
    return DEPTH_KM, vp, vp / 1.75, 2.5 + 0.005 * DEPTH_KM


def test_the_field_is_centred_scaled_and_smoother_the_longer_it_correlates():
    generator = np.random.default_rng(0)
    field = unit_smooth_field(500, 0.5, 30.0, generator)
    assert field.mean() == pytest.approx(0.0, abs=1e-12)
    assert field.std() == pytest.approx(1.0, abs=1e-9)
    rough = unit_smooth_field(500, 0.5, 2.0, np.random.default_rng(0))
    smooth = unit_smooth_field(500, 0.5, 40.0, np.random.default_rng(0))
    assert np.abs(np.diff(smooth)).mean() < np.abs(np.diff(rough)).mean()


def test_the_spread_tapers_from_the_crust_to_the_lid():
    sigma = depth_tapered_sigma(DEPTH_KM, 0.05, 0.015, 30.0, 65.0)
    assert sigma[0] == pytest.approx(0.05)
    assert sigma[DEPTH_KM >= 65.0][0] == pytest.approx(0.015)
    assert np.all(np.diff(sigma) <= 1e-12)
    midpoint = sigma[np.argmin(np.abs(DEPTH_KM - 47.5))]
    assert midpoint == pytest.approx(0.5 * (0.05 + 0.015), abs=1e-3)


def test_the_same_seed_gives_the_same_member_and_a_different_one_does_not():
    sigma = depth_tapered_sigma(DEPTH_KM)
    first = perturb_layered_model(*reference(), sigma_profile=sigma, seed=1)
    again = perturb_layered_model(*reference(), sigma_profile=sigma, seed=1)
    other = perturb_layered_model(*reference(), sigma_profile=sigma, seed=2)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not np.array_equal(first[1], other[1])


def test_a_member_stays_an_elastic_solid():
    sigma = depth_tapered_sigma(DEPTH_KM, 0.3, 0.3)
    _, vp, vs, _ = perturb_layered_model(*reference(), sigma_profile=sigma,
                                         vp_vs_correlation=0.0, seed=3)
    assert np.all(vs <= MAX_VS_OVER_VP * vp + 1e-12)


def test_density_follows_the_compressional_speed_by_the_coupling():
    depth, vp, vs, density = reference()
    sigma = depth_tapered_sigma(depth)
    _, perturbed_vp, _, perturbed_density = perturb_layered_model(
        depth, vp, vs, density, sigma_profile=sigma, density_coupling=0.25, seed=4)
    assert np.allclose(np.log(perturbed_density / density),
                       0.25 * np.log(perturbed_vp / vp))
    _, _, _, fixed = perturb_layered_model(depth, vp, vs, density, sigma_profile=sigma,
                                           density_coupling=0.0, seed=4)
    assert np.array_equal(fixed, density)


def test_the_layer_thicknesses_move_only_when_they_are_asked_to():
    depth, vp, vs, density = reference()
    sigma = depth_tapered_sigma(depth)
    thickness, *_ = perturb_layered_model(depth, vp, vs, density, sigma_profile=sigma,
                                          width_sigma=0.0, seed=5)
    assert np.allclose(thickness, 0.5)
    widened, *_ = perturb_layered_model(depth, vp, vs, density, sigma_profile=sigma,
                                        width_sigma=0.1, seed=5)
    assert not np.allclose(widened, 0.5)
    assert np.all(widened > 0)


def test_the_ensemble_is_wider_where_the_taper_says_it_should_be():
    depth, vp, vs, density = reference()
    sigma = depth_tapered_sigma(depth, 0.05, 0.015, 30.0, 65.0)
    members = np.array([perturb_layered_model(depth, vp, vs, density, sigma_profile=sigma,
                                              seed=seed)[1] for seed in range(60)])
    fractional = members.std(axis=0) / vp
    assert fractional[depth <= 20].mean() > 3 * fractional[depth >= 80].mean()


def test_the_control_point_spread_passes_through_its_own_points():
    depth = np.arange(0.0, 40.0, 0.25)
    sigma = control_point_sigma(depth, (0.0, 10.0, 40.0), (0.20, 0.05, 0.015))
    assert sigma[0] == pytest.approx(0.20, abs=0.02)
    assert sigma[np.argmin(abs(depth - 10.0))] == pytest.approx(0.05, abs=0.01)
    assert sigma[-1] == pytest.approx(0.015, abs=0.005)
    assert np.all(np.diff(sigma) <= 1e-9)


def test_the_smoothing_removes_the_corner_interpolation_leaves():
    depth = np.arange(0.0, 40.0, 0.25)
    rough = control_point_sigma(depth, (0.0, 10.0, 40.0), (0.2, 0.05, 0.015),
                                smoothing_km=0.25)
    smooth = control_point_sigma(depth, (0.0, 10.0, 40.0), (0.2, 0.05, 0.015),
                                 smoothing_km=4.0)
    assert np.abs(np.diff(smooth, 2)).max() < np.abs(np.diff(rough, 2)).max()


def test_the_speed_and_the_ratio_are_perturbed_independently():
    depth, vp, vs, density = reference()
    sigma = np.full_like(vp, 0.05)
    _, member_vp, member_vs, _ = perturb_velocity_ratio_model(
        depth, vp, vs, density, sigma_vp=sigma, sigma_ratio=np.zeros_like(vp), seed=7)
    assert np.allclose(member_vp / member_vs, vp / vs)
    _, fixed_vp, fixed_vs, _ = perturb_velocity_ratio_model(
        depth, vp, vs, density, sigma_vp=np.zeros_like(vp), sigma_ratio=sigma, seed=7)
    assert np.allclose(fixed_vp, vp)
    assert not np.allclose(fixed_vs, vs)


def test_the_ratio_is_held_above_the_solid_limit():
    depth, vp, vs, density = reference()
    sigma = np.full_like(vp, 0.5)
    _, member_vp, member_vs, _ = perturb_velocity_ratio_model(
        depth, vp, vs, density, sigma_vp=np.zeros_like(vp), sigma_ratio=sigma, min_ratio=1.6,
        seed=3)
    assert np.all(member_vp / member_vs >= 1.6 - 1e-9)


def test_the_ratio_member_repeats_for_its_seed_and_moves_its_density_with_the_speed():
    depth, vp, vs, density = reference()
    sigma = np.full_like(vp, 0.05)
    first = perturb_velocity_ratio_model(depth, vp, vs, density, sigma_vp=sigma,
                                         sigma_ratio=sigma, seed=1)
    again = perturb_velocity_ratio_model(depth, vp, vs, density, sigma_vp=sigma,
                                         sigma_ratio=sigma, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert np.allclose(np.log(first[3] / density), 0.25 * np.log(first[1] / vp))
