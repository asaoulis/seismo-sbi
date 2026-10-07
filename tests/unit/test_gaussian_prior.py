"""The Gaussian prior on the source parameters, in the compressor, the likelihood and least squares."""
from types import SimpleNamespace

import numpy as np
import pytest

from seismo_sbi.sbi.compression.gaussian import GaussianCompressor, ScoreCompressionData
from seismo_sbi.sbi.inversion.gaussian_prior import GaussianPrior
from seismo_sbi.sbi.inversion.least_squares import IterativeLeastSquaresSolver
from seismo_sbi.sbi.inversion.likelihood import GaussianLikelihoodEvaluator
from seismo_sbi.sbi.noises.diagonal_covariances import ScalarEmpiricalCovariance
from seismo_sbi.sbi.types.parameters import IterativeLeastSquaresParameters, ModelParameters

NOISE_SIGMA = 0.3
PRIOR = GaussianPrior(mean=np.array([0.5, -0.2, 0.1, 0.0, 1.0, -1.0]),
                      variances=np.array([0.4, 1.0, 2.0, 0.25, 0.1, 3.0]))


def linear_problem(n_data=40):
    """``(G, data)`` with ``data = theta @ G + noise`` for six parameters."""
    rng = np.random.default_rng(5)
    G = rng.standard_normal((6, n_data))
    return G, rng.standard_normal(6) @ G + NOISE_SIGMA * rng.standard_normal(n_data)


def posterior_mean(G, data):
    precision = G @ G.T / NOISE_SIGMA ** 2 + np.diag(1 / PRIOR.variances)
    return np.linalg.solve(precision, G @ data / NOISE_SIGMA ** 2 + PRIOR.mean / PRIOR.variances)


def compressor_at(theta, G, prior):
    compression_data = ScoreCompressionData(theta, theta @ G, G, None)
    return GaussianCompressor(compression_data, ScalarEmpiricalCovariance(NOISE_SIGMA, G.shape[1]), prior=prior)


def test_the_prior_terms_are_the_gaussian_density_and_its_gradient():
    theta = np.array([1.0, 0.0, -1.0, 0.5, 0.2, 0.3])
    residual = theta - PRIOR.mean
    assert PRIOR.log_density(theta) == pytest.approx(-0.5 * np.sum(residual ** 2 / PRIOR.variances))
    assert np.allclose(PRIOR.score(theta), -residual / PRIOR.variances)
    assert np.allclose(PRIOR.precision(), np.diag(1 / PRIOR.variances))


def test_the_compressor_estimate_with_a_prior_is_the_posterior_mean():
    G, data = linear_problem()
    compressor = compressor_at(np.array([2.0, 1.0, -3.0, 0.0, 0.5, 1.0]), G, PRIOR)

    assert np.allclose(compressor.compute_theta_MLE(data), posterior_mean(G, data))
    assert np.allclose(compressor.Fisher_mat, G @ G.T / NOISE_SIGMA ** 2 + np.diag(1 / PRIOR.variances))


def test_the_likelihood_adds_the_prior_log_density():
    G, data = linear_problem()
    scaler = SimpleNamespace(inverse_transform=lambda scaled: 4 * scaled - 2)
    loss = lambda residual: -0.5 * np.sum(residual ** 2) / NOISE_SIGMA ** 2
    evaluator = GaussianLikelihoodEvaluator(data, lambda theta: theta @ G, scaler, loss, prior=PRIOR)
    scaled = np.array([0.6, 0.4, 0.55, 0.45, 0.5, 0.7])
    theta = 4 * scaled - 2

    assert evaluator.log_probability(scaled) == pytest.approx(loss(theta @ G - data) + PRIOR.log_density(theta))
    assert evaluator.log_probability(np.full(6, 1.2)) == -np.inf


def test_least_squares_keeps_the_prior_and_reaches_the_posterior_mean():
    G, data = linear_problem()
    model_parameters = ModelParameters()
    model_parameters.theta_fiducial = {"moment_tensor": np.array([2.0, 1.0, -3.0, 0.0, 0.5, 1.0])}

    def compression_data(parameters, *stencil_args, seed=None):
        theta = parameters.parameter_to_vector("theta_fiducial", True)
        return ScoreCompressionData(theta, theta @ G, G, None), None

    solver = IterativeLeastSquaresSolver.__new__(IterativeLeastSquaresSolver)
    solver.model_parameters, solver.stencil_args, solver.seed = model_parameters, (), None
    solver.data_manager = SimpleNamespace(compute_required_compression_data=compression_data)
    solver.least_squares_configuration = IterativeLeastSquaresParameters(max_iterations=4, damping_factor=0.0)
    compressor = compressor_at(model_parameters.theta_fiducial["moment_tensor"], G, PRIOR)

    final, _ = solver.solve_least_squares(data, compressor, single_step=True)

    assert compressor.prior is PRIOR
    assert np.allclose(final.theta_fiducial, posterior_mean(G, data))
