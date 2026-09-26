"""Covariance paths that used to fail: samplers without receivers, per-element scalar loss and the
estimator's default taper.
"""
import numpy as np

from seismo_sbi.sbi.noises.covariance_estimator import EmpiricalCovarianceEstimator
from seismo_sbi.sbi.noises.diagonal_covariances import DiagonalEmpiricalCovariance, ScalarEmpiricalCovariance
from tests.unit.test_covariance_characterisation import (
    BLOCK_SIZE, autocovariances, make_receivers, write_noise_windows,
)


def test_scalar_sampler_draws_white_noise_of_the_data_length():
    np.random.seed(3)
    noise, _ = ScalarEmpiricalCovariance(0.5, data_vector_length=20000).create_sampler()()
    assert noise.shape == (20000,)
    assert abs(noise.std() - 0.5) < 0.01


def test_diagonal_sampler_draws_each_trace_at_its_variance():
    covariance = DiagonalEmpiricalCovariance(autocovariances(BLOCK_SIZE), make_receivers(), BLOCK_SIZE)
    np.random.seed(4)
    noise = np.stack([covariance.create_sampler()()[0] for _ in range(2000)])
    assert noise.shape == (2000, 6 * BLOCK_SIZE)
    np.testing.assert_allclose(noise.var(axis=0).reshape(6, BLOCK_SIZE).mean(axis=1),
                               covariance.covariance_matrix[::BLOCK_SIZE], rtol=0.05)


def test_scalar_per_element_loss_sums_to_the_loss():
    covariance = ScalarEmpiricalCovariance(0.7)
    residual = np.linspace(-1.0, 1.0, 11)
    per_element = covariance.compute_loss(residual, reduce=False)
    np.testing.assert_allclose(per_element, -0.5 * residual**2 / 0.49)
    np.testing.assert_allclose(per_element.sum(), covariance.compute_loss(residual, reduce=True))


def test_estimator_default_taper_matches_explicit_taper(tmp_path):
    write_noise_windows(tmp_path)
    tapered = EmpiricalCovarianceEstimator(tmp_path, make_receivers(), "ZEN").compute_stationwise_covariances()
    raw = EmpiricalCovarianceEstimator(tmp_path, make_receivers(), "ZEN", covariance_exp_tapering=False)
    expected = EmpiricalCovarianceEstimator.taper_covariances(
        raw.convert_to_covariance(raw._compute_standard_deviation_online()), BLOCK_SIZE)
    for station, components in expected.items():
        for component, autocovariance in components.items():
            np.testing.assert_allclose(tapered[station][component], autocovariance, rtol=1e-12)

