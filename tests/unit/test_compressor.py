import numpy as np
import pytest

from seismo_sbi.sbi.compression.gaussian import GaussianCompressor, ScoreCompressionData
from seismo_sbi.sbi.noises.covariance_estimation import ScalarEmpiricalCovariance

from tests.conftest import TRACE_LEN, N_PARAMS


def test_fisher_matrix_positive_definite(gaussian_compressor):
    eigvals = np.linalg.eigvalsh(gaussian_compressor.Fisher_mat)
    assert np.all(eigvals > 0), "Fisher matrix is not positive definite"


def test_fisher_matrix_symmetric(gaussian_compressor):
    F = gaussian_compressor.Fisher_mat
    assert np.allclose(F, F.T), "Fisher matrix is not symmetric"


def test_fisher_matrix_shape(gaussian_compressor):
    assert gaussian_compressor.Fisher_mat.shape == (N_PARAMS, N_PARAMS)


def test_compress_fiducial_returns_fiducial(gaussian_compressor, score_compression_data):
    """Compressing the fiducial data vector should return the fiducial parameters."""
    D_fid = score_compression_data.data_fiducial
    theta_fid = score_compression_data.theta_fiducial
    compressed = gaussian_compressor.compress_data_vector(D_fid)
    assert np.allclose(compressed, theta_fid, atol=1e-10), (
        f"Expected {theta_fid}, got {compressed}"
    )


def test_compute_misfit_fiducial_is_zero(gaussian_compressor, score_compression_data):
    """Misfit at the fiducial point must be zero (residual is zero)."""
    misfit = gaussian_compressor.compute_misfit(score_compression_data.data_fiducial)
    assert np.isclose(misfit, 0.0, atol=1e-12)


def test_compute_misfit_non_zero_for_perturbed(gaussian_compressor, score_compression_data):
    rng = np.random.default_rng(99)
    D_perturbed = score_compression_data.data_fiducial + rng.standard_normal(TRACE_LEN) * 0.5
    misfit = gaussian_compressor.compute_misfit(D_perturbed)
    assert misfit > 0


def test_score_shape(gaussian_compressor, score_compression_data):
    rng = np.random.default_rng(7)
    D = score_compression_data.data_fiducial + rng.standard_normal(TRACE_LEN) * 0.1
    score = gaussian_compressor.compute_score(D)
    assert score.shape == (N_PARAMS,)


def test_score_zero_at_fiducial(gaussian_compressor, score_compression_data):
    """Score should be zero when data equals the fiducial data vector."""
    score = gaussian_compressor.compute_score(score_compression_data.data_fiducial)
    assert np.allclose(score, 0.0, atol=1e-12)


def test_theta_mle_output_shape(gaussian_compressor, score_compression_data):
    rng = np.random.default_rng(11)
    D = score_compression_data.data_fiducial + rng.standard_normal(TRACE_LEN) * 0.3
    mle = gaussian_compressor.compute_theta_MLE(D)
    assert mle.shape == (N_PARAMS,)


def test_set_compression_variables_updates_fiducial(score_compression_data, scalar_covariance):
    """set_compression_variables should update stored fiducial and recompute Fisher."""
    compressor = GaussianCompressor(score_compression_data, scalar_covariance)
    rng = np.random.default_rng(13)
    new_theta = rng.standard_normal(N_PARAMS)
    new_D = rng.standard_normal(TRACE_LEN)
    new_grads = rng.standard_normal((N_PARAMS, TRACE_LEN))
    new_data = ScoreCompressionData(new_theta, new_D, new_grads, None)
    compressor.set_compression_variables(new_data)
    assert np.allclose(compressor.theta_fiducial, new_theta)
    assert np.allclose(compressor.D_fiducial, new_D)


def test_with_diagonal_covariance():
    """GaussianCompressor works with a non-identity diagonal covariance."""
    rng = np.random.default_rng(17)
    n_data, n_params = 50, 2
    sigma = 3.0
    cov = ScalarEmpiricalCovariance(sigma_noise_level=sigma)
    theta_fid = np.zeros(n_params)
    D_fid = rng.standard_normal(n_data)
    grads = rng.standard_normal((n_params, n_data))
    data = ScoreCompressionData(theta_fid, D_fid, grads, None)
    comp = GaussianCompressor(data, cov)
    # Fisher should scale as 1/sigma^2
    cov_unit = ScalarEmpiricalCovariance(sigma_noise_level=1.0)
    comp_unit = GaussianCompressor(data, cov_unit)
    ratio = comp_unit.Fisher_mat / comp.Fisher_mat
    assert np.allclose(ratio, sigma ** 2, rtol=1e-6)
