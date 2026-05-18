"""Integration tests: build a compressor from data stubs and compress a data vector."""

import numpy as np
import pytest

from seismo_sbi.sbi.compression.gaussian import GaussianCompressor, ScoreCompressionData
from seismo_sbi.sbi.noises.covariance_estimation import (
    ScalarEmpiricalCovariance,
    BlockDiagonalFilteredCovariance,
    BlockDiagonalKolbCovariance,
)

from tests.conftest import TRACE_LEN, N_PARAMS

RNG = np.random.default_rng(55)


def _make_linear_model(n_data, n_params, rng):
    """Simple linear forward model: D(θ) = A @ θ + D₀."""
    A = rng.standard_normal((n_data, n_params))
    theta_fid = rng.standard_normal(n_params)
    D_fid = A @ theta_fid
    return A, theta_fid, D_fid


class TestCompressionWithScalarCovariance:
    """Compressor built with σ²I covariance (simplest path)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.default_rng(1)
        A, theta_fid, D_fid = _make_linear_model(TRACE_LEN, N_PARAMS, rng)
        self.A = A
        self.theta_fid = theta_fid
        self.D_fid = D_fid
        cov = ScalarEmpiricalCovariance(sigma_noise_level=0.1)
        data = ScoreCompressionData(
            theta_fiducial=theta_fid,
            data_fiducial=D_fid,
            data_parameter_gradients=A.T,
            second_order_gradients=None,
        )
        self.compressor = GaussianCompressor(data, cov)

    def test_output_shape(self):
        compressed = self.compressor.compress_data_vector(self.D_fid)
        assert compressed.shape == (N_PARAMS,)

    def test_fiducial_recovery(self):
        """Compressing noiseless fiducial data should exactly recover θ₀."""
        compressed = self.compressor.compress_data_vector(self.D_fid)
        assert np.allclose(compressed, self.theta_fid, atol=1e-10)

    def test_perturbed_data_shifts_estimate(self):
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(TRACE_LEN) * 0.05
        compressed = self.compressor.compress_data_vector(self.D_fid + noise)
        # Compressed output should differ from fiducial when data is perturbed
        assert not np.allclose(compressed, self.theta_fid, atol=1e-3)

    def test_mle_within_reasonable_range(self):
        """For a small noise level, MLE should be within 3σ of truth."""
        rng = np.random.default_rng(5)
        sigma = 0.01
        theta_true = self.theta_fid + rng.standard_normal(N_PARAMS) * 0.1
        D_obs = self.A @ theta_true + rng.standard_normal(TRACE_LEN) * sigma
        mle = self.compressor.compress_data_vector(D_obs)
        # With good signal-to-noise, MLE should be close to truth
        assert np.linalg.norm(mle - theta_true) < 1.0


class TestCompressionWithFilteredCovariance:
    """Compressor built with band-pass filtered block covariance."""

    @pytest.fixture(autouse=True)
    def setup(self, receivers):
        rng = np.random.default_rng(7)
        cov = BlockDiagonalFilteredCovariance(
            station_component_covariances={"STA1": {"Z": 1.0}},
            filter={"freqmin": 0.01, "freqmax": 0.1},
            receivers=receivers,
            data_vector_length=TRACE_LEN,
            num_jobs=1,
        )
        cov.set_data_vector_length(TRACE_LEN)
        cov.set_toeplitz_cols(cov.toeplitz_cols_list)

        theta_fid = rng.standard_normal(N_PARAMS)
        D_fid = rng.standard_normal(TRACE_LEN)
        gradients = rng.standard_normal((N_PARAMS, TRACE_LEN))
        data = ScoreCompressionData(theta_fid, D_fid, gradients, None)
        self.theta_fid = theta_fid
        self.D_fid = D_fid
        self.compressor = GaussianCompressor(data, cov)

    def test_output_shape(self):
        compressed = self.compressor.compress_data_vector(self.D_fid)
        assert compressed.shape == (N_PARAMS,)

    def test_fiducial_recovery(self):
        compressed = self.compressor.compress_data_vector(self.D_fid)
        assert np.allclose(compressed, self.theta_fid, atol=1e-10)

    def test_fisher_is_positive_definite(self):
        eigvals = np.linalg.eigvalsh(self.compressor.Fisher_mat)
        assert np.all(eigvals > 0)


class TestCompressionWithKolbCovariance:
    """Compressor built with Kolb-parameterised covariance."""

    def test_fiducial_recovery(self, receivers):
        rng = np.random.default_rng(9)
        cov = BlockDiagonalKolbCovariance(
            station_component_covariances={"STA1": {"Z": 1.0}},
            receivers=receivers,
            data_vector_length=TRACE_LEN,
            num_jobs=1,
        )
        cov.set_data_vector_length(TRACE_LEN)
        cov.set_toeplitz_cols(cov.toeplitz_cols_list)

        theta_fid = rng.standard_normal(N_PARAMS)
        D_fid = rng.standard_normal(TRACE_LEN)
        gradients = rng.standard_normal((N_PARAMS, TRACE_LEN))
        data = ScoreCompressionData(theta_fid, D_fid, gradients, None)
        compressor = GaussianCompressor(data, cov)

        compressed = compressor.compress_data_vector(D_fid)
        assert np.allclose(compressed, theta_fid, atol=1e-10)
