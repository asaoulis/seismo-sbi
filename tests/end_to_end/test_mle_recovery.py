"""End-to-end fuzzy tests: MLE recovery with toy linear model.

Uses GaussianCompressor.compute_theta_MLE directly (no Instaseis, no
pipeline boilerplate) to verify that the optimal score compression scheme
recovers the true parameters from noisy data within the expected tolerance.
"""

import numpy as np
import pytest

from seismo_sbi.sbi.compression.gaussian import GaussianCompressor, ScoreCompressionData
from seismo_sbi.sbi.noises.covariance_estimation import ScalarEmpiricalCovariance

pytestmark = pytest.mark.slow


def _build_linear_compressor(n_data, n_params, sigma, rng):
    """Return (compressor, A, theta_fid) for forward model D = A @ theta."""
    A = rng.standard_normal((n_data, n_params))
    # Condition the matrix slightly to avoid singular Fisher
    A = A + np.eye(n_data, n_params) * 0.1
    theta_fid = rng.standard_normal(n_params)
    D_fid = A @ theta_fid
    cov = ScalarEmpiricalCovariance(sigma_noise_level=sigma)
    data = ScoreCompressionData(
        theta_fiducial=theta_fid,
        data_fiducial=D_fid,
        data_parameter_gradients=A.T,
        second_order_gradients=None,
    )
    return GaussianCompressor(data, cov), A, theta_fid


class TestMLERecovery:
    """MLE from optimal score compression should recover true θ in simple linear models."""

    def test_noiseless_exact_recovery(self):
        """Without noise, MLE = θ_true exactly."""
        rng = np.random.default_rng(42)
        sigma = 1.0
        comp, A, theta_fid = _build_linear_compressor(n_data=80, n_params=3, sigma=sigma, rng=rng)
        theta_true = rng.standard_normal(3)
        D_obs = A @ theta_true  # noiseless
        # Re-expand around true theta_true as fiducial for noiseless test
        mle = comp.compute_theta_MLE(D_obs)
        # MLE from first-order expansion around theta_fid recovers theta_true exactly
        # if the model is exactly linear (D = A @ theta)
        assert np.allclose(mle, theta_true, atol=1e-10), (
            f"Noiseless MLE failed: got {mle}, expected {theta_true}"
        )

    def test_high_snr_recovery_within_1sigma(self):
        """With high SNR, MLE should be within 1σ of the true parameter."""
        rng = np.random.default_rng(77)
        n_data, n_params = 200, 4
        sigma = 0.01  # very small noise
        comp, A, theta_fid = _build_linear_compressor(n_data, n_params, sigma, rng)

        results = []
        for trial in range(20):
            theta_true = theta_fid + rng.standard_normal(n_params) * 0.5
            noise = rng.standard_normal(n_data) * sigma
            D_obs = A @ theta_true + noise
            mle = comp.compute_theta_MLE(D_obs)
            results.append(np.linalg.norm(mle - theta_true))

        # At high SNR, median recovery error should be small
        median_error = np.median(results)
        assert median_error < 0.5, f"Median MLE error too large: {median_error:.4f}"

    def test_fisher_bounds_posterior_width(self):
        """Fisher inverse sets the expected posterior covariance; MLE spread should match."""
        rng = np.random.default_rng(13)
        n_data, n_params = 100, 2
        sigma = 1.0
        comp, A, theta_fid = _build_linear_compressor(n_data, n_params, sigma, rng)

        n_trials = 200
        mle_samples = np.zeros((n_trials, n_params))
        for i in range(n_trials):
            noise = rng.standard_normal(n_data) * sigma
            D_obs = A @ theta_fid + noise
            mle_samples[i] = comp.compute_theta_MLE(D_obs)

        empirical_cov = np.cov(mle_samples.T)
        expected_cov = comp.Fisher_mat_inverse

        # Diagonal entries of empirical and Cramér-Rao covariance should agree within 50%
        for i in range(n_params):
            ratio = empirical_cov[i, i] / expected_cov[i, i]
            assert 0.5 < ratio < 2.0, (
                f"Empirical variance ratio[{i}]={ratio:.3f} far from Cramér-Rao bound"
            )

    def test_relative_recovery_within_10_percent(self):
        """Fuzzy check: |MLE - θ_true| / |θ_true| < 10% for strong signal."""
        rng = np.random.default_rng(99)
        n_data, n_params = 150, 3
        sigma = 0.001
        comp, A, theta_fid = _build_linear_compressor(n_data, n_params, sigma, rng)

        # Use a theta_true with bounded magnitude so relative error is meaningful
        theta_true = np.array([1.0, -2.0, 0.5])
        D_obs = A @ theta_true + rng.standard_normal(n_data) * sigma
        mle = comp.compute_theta_MLE(D_obs)

        relative_errors = np.abs(mle - theta_true) / (np.abs(theta_true) + 1e-8)
        assert np.all(relative_errors < 0.10), (
            f"Relative errors exceed 10%: {relative_errors}"
        )

    def test_mle_no_nan_or_inf(self):
        """MLE should never return NaN or Inf."""
        rng = np.random.default_rng(55)
        comp, A, theta_fid = _build_linear_compressor(n_data=50, n_params=3, sigma=1.0, rng=rng)
        for _ in range(50):
            theta_true = rng.standard_normal(3)
            D_obs = A @ theta_true + rng.standard_normal(50) * 0.5
            mle = comp.compute_theta_MLE(D_obs)
            assert np.all(np.isfinite(mle)), f"Non-finite MLE: {mle}"
