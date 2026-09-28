"""The Gaussian-likelihood MCMC recovers the analytical posterior of a linear forward model."""
import numpy as np

from seismo_sbi.sbi.inversion.likelihood import GaussianLikelihoodEvaluator, generate_samples
from seismo_sbi.sbi.noises.diagonal_covariances import ScalarEmpiricalCovariance
from seismo_sbi.sbi.scalers import ZeroOneScaler

N_SAMPLES = 60
NOISE_SIGMA = 1.0


def test_mcmc_recovers_the_analytical_posterior():
    rng = np.random.default_rng(0)
    kernels = rng.normal(size=(6, N_SAMPLES)) * 1e-16
    true_moment_tensor = np.array([3.0, -2.0, -1.0, 1.5, -0.5, 2.0]) * 1e16
    observed = kernels.T @ true_moment_tensor + rng.normal(scale=NOISE_SIGMA, size=N_SAMPLES)
    posterior_covariance = NOISE_SIGMA ** 2 * np.linalg.inv(kernels @ kernels.T)
    posterior_mean = posterior_covariance @ kernels @ observed / NOISE_SIGMA ** 2

    scaler = ZeroOneScaler((np.full(6, -5e17), np.full(6, 5e17)))
    covariance = ScalarEmpiricalCovariance(NOISE_SIGMA, data_vector_length=N_SAMPLES)
    evaluator = GaussianLikelihoodEvaluator(
        observed, lambda moment_tensor: kernels.T @ moment_tensor, scaler,
        covariance.create_loss_callable(covariance.inverse_metadata, covariance.data_vector_length))
    proposal = 2.38 ** 2 / 6 * posterior_covariance / np.outer(scaler.range, scaler.range)
    samples = scaler.inverse_transform(generate_samples(
        evaluator.log_probability, False, 6, nsamples_per_walker=20000, nwalkers=1, burn_in=2000,
        num_processes=1, move_size=proposal, mle_start=scaler.transform(posterior_mean.reshape(1, -1)).ravel(),
        seed=1))

    posterior_std = np.sqrt(np.diag(posterior_covariance))
    assert np.all(np.abs(samples.mean(axis=0) - posterior_mean) / posterior_std < 0.3)
    assert np.all(np.abs(samples.std(axis=0) / posterior_std - 1) < 0.15)
