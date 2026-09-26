"""Covariance paths that used to fail: samplers without receivers, per-element scalar loss, the
estimator's default taper and the theory covariance's parameter derivatives.
"""
import numpy as np
from scipy.linalg import block_diag

from seismo_sbi.sbi.compression.gaussian import GaussianCompressor
from seismo_sbi.sbi.noises.covariance_estimator import EmpiricalCovarianceEstimator
from seismo_sbi.sbi.noises.diagonal_covariances import DiagonalEmpiricalCovariance, ScalarEmpiricalCovariance
from seismo_sbi.sbi.noises.theory_block_covariance import TheoryBlockDiagonalEmpiricalCovariance
from seismo_sbi.sbi.noises.toeplitz_covariances import BlockDiagonalKolbCovariance
from tests.unit.test_covariance_characterisation import (
    BLOCK_SIZE, autocovariances, make_receivers, synthetic_problem, theory_covariance_blocks,
    variances, write_noise_windows,
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


def dense_theory_problem():
    data_covariance = BlockDiagonalKolbCovariance(
        variances(), receivers=make_receivers(), data_vector_length=BLOCK_SIZE, num_jobs=1)
    theory = theory_covariance_blocks()
    covariance = TheoryBlockDiagonalEmpiricalCovariance(
        theory, data_covariance.covariance_matrix_arrays, make_receivers(), BLOCK_SIZE,
        diag_regularisation=0.01, covariance_gradients=True, num_jobs=1)
    dense = block_diag(*covariance.covariance_matrix_arrays)
    derivatives = [block_diag(*blocks) for blocks in
                   theory.data_parameter_gradients.reshape(-1, 6, BLOCK_SIZE, BLOCK_SIZE)]
    return covariance, dense, derivatives


def test_theory_fisher_includes_the_covariance_derivative_term():
    covariance, dense, derivatives = dense_theory_problem()
    _, compression_data = synthetic_problem(6 * BLOCK_SIZE)
    gradients = compression_data.data_parameter_gradients
    inverse = np.linalg.inv(dense)
    expected = gradients @ inverse @ gradients.T + 0.5 * np.array(
        [[np.trace(inverse @ da @ inverse @ db) for db in derivatives] for da in derivatives])
    np.testing.assert_allclose(GaussianCompressor(compression_data, covariance).Fisher_mat, expected, rtol=1e-8)


def test_theory_score_includes_the_covariance_derivative_term():
    covariance, dense, derivatives = dense_theory_problem()
    residual, compression_data = synthetic_problem(6 * BLOCK_SIZE)
    inverse = np.linalg.inv(dense)
    expected = compression_data.data_parameter_gradients @ inverse @ residual + np.array(
        [0.5 * residual @ inverse @ da @ inverse @ residual - 0.5 * np.trace(inverse @ da) for da in derivatives])
    score = GaussianCompressor(compression_data, covariance).compute_score(compression_data.data_fiducial + residual)
    np.testing.assert_allclose(score, expected, rtol=1e-8)

