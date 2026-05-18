import numpy as np
import pytest
from scipy.linalg import toeplitz

from seismo_sbi.sbi.noises.covariance_estimation import (
    ScalarEmpiricalCovariance,
    GaussianNoiseSampler,
    BlockDiagonalFilteredCovariance,
    BlockDiagonalKolbCovariance,
)

BLOCK_SIZE = 30


# ---------------------------------------------------------------------------
# ScalarEmpiricalCovariance
# ---------------------------------------------------------------------------

def test_scalar_matmul_inverse_covariance():
    sigma = 2.0
    cov = ScalarEmpiricalCovariance(sigma_noise_level=sigma)
    v = np.array([1.0, 2.0, 3.0])
    result = cov.matmul_inverse_covariance(v)
    assert np.allclose(result, v / sigma ** 2)


def test_scalar_loss_callable_returns_negative():
    cov = ScalarEmpiricalCovariance(sigma_noise_level=1.0)
    loss = cov.generic_loss_callable(np.array([1.0, 2.0]))
    assert loss < 0


def test_scalar_loss_callable_zero_residual():
    cov = ScalarEmpiricalCovariance(sigma_noise_level=1.0)
    loss = cov.generic_loss_callable(np.zeros(5))
    assert np.isclose(loss, 0.0)


# ---------------------------------------------------------------------------
# GaussianNoiseSampler
# ---------------------------------------------------------------------------

def test_gaussian_noise_sampler_output_shape(receivers):
    n = BLOCK_SIZE
    cov_block = np.eye(n) * 4.0
    sampler = GaussianNoiseSampler(
        receivers=receivers,
        data_vector_length=n,
        cov_blocks=[cov_block],
    )
    noise, meta = sampler()
    assert noise.shape == (n,)


def test_gaussian_noise_sampler_zero_mean_distribution(receivers):
    """Over many samples the mean should be close to zero."""
    n = BLOCK_SIZE
    cov_block = np.eye(n) * 1.0
    sampler = GaussianNoiseSampler(
        receivers=receivers,
        data_vector_length=n,
        cov_blocks=[cov_block],
    )
    samples = np.array([sampler()[0] for _ in range(500)])
    assert np.abs(samples.mean()) < 0.2


def test_gaussian_noise_sampler_adaptive_covariance(receivers):
    """Scaling via set_adaptive_covariance_with_misc_data changes the variance."""
    n = BLOCK_SIZE
    sigma_initial = 1.0
    cov_block = np.eye(n) * sigma_initial ** 2
    sampler = GaussianNoiseSampler(
        receivers=receivers,
        data_vector_length=n,
        toeplitz_cols=np.array([np.eye(n)[0] * sigma_initial ** 2]),
        cov_blocks=[cov_block],
    )
    sigma_new = 3.0
    misc_data = {"STA1": {"Z": sigma_new ** 2}}
    sampler.set_adaptive_covariance_with_misc_data(misc_data)
    assert np.isclose(sampler.toeplitz_cols[0, 0], sigma_new ** 2)


# ---------------------------------------------------------------------------
# BlockDiagonalFilteredCovariance
# ---------------------------------------------------------------------------

def test_filtered_covariance_blocks_psd(receivers):
    cov = BlockDiagonalFilteredCovariance(
        station_component_covariances={"STA1": {"Z": 1.0}},
        filter={"freqmin": 0.01, "freqmax": 0.1},
        receivers=receivers,
        data_vector_length=BLOCK_SIZE,
        num_jobs=1,
    )
    for block in cov.covariance_matrix_arrays:
        eigvals = np.linalg.eigvalsh(block)
        assert np.all(eigvals >= -1e-10), "Covariance block is not PSD"


def test_filtered_covariance_creates_sampler(receivers):
    cov = BlockDiagonalFilteredCovariance(
        station_component_covariances={"STA1": {"Z": 1.0}},
        filter={"freqmin": 0.01, "freqmax": 0.1},
        receivers=receivers,
        data_vector_length=BLOCK_SIZE,
        num_jobs=1,
    )
    sampler = cov.create_sampler()
    noise, _ = sampler()
    assert noise.shape == (BLOCK_SIZE,)


def test_filtered_covariance_loss_is_negative(receivers):
    cov = BlockDiagonalFilteredCovariance(
        station_component_covariances={"STA1": {"Z": 1.0}},
        filter={"freqmin": 0.01, "freqmax": 0.1},
        receivers=receivers,
        data_vector_length=BLOCK_SIZE,
        num_jobs=1,
    )
    cov.set_toeplitz_cols(cov.toeplitz_cols_list)
    cov.set_data_vector_length(BLOCK_SIZE)
    residuals = np.ones(BLOCK_SIZE)
    loss = cov.generic_loss_callable(residuals)
    assert loss < 0


# ---------------------------------------------------------------------------
# BlockDiagonalKolbCovariance
# ---------------------------------------------------------------------------

def test_kolb_covariance_blocks_psd(receivers):
    cov = BlockDiagonalKolbCovariance(
        station_component_covariances={"STA1": {"Z": 1.0}},
        receivers=receivers,
        data_vector_length=BLOCK_SIZE,
        num_jobs=1,
    )
    for block in cov.covariance_matrix_arrays:
        eigvals = np.linalg.eigvalsh(block)
        assert np.all(eigvals >= -1e-10), "Kolb covariance block is not PSD"
