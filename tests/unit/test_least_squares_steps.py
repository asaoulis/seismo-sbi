"""The iterative least-squares solver never steps from a worse model than one it already had."""
from types import SimpleNamespace

import numpy as np

from seismo_sbi.sbi.compression.gaussian import GaussianCompressor, ScoreCompressionData
from seismo_sbi.sbi.inversion.least_squares import IterativeLeastSquaresSolver
from seismo_sbi.sbi.noises.diagonal_covariances import ScalarEmpiricalCovariance
from seismo_sbi.sbi.types.parameters import IterativeLeastSquaresParameters, ModelParameters

TIMES_S = np.linspace(0.0, 10.0, 200)
TRUTH = np.array([2.0, 0.3, 0.05, -0.4])


def model(theta):
    """A decaying exponential on a linear trend; nonlinear in the decay rate."""
    a, b, c, d = theta
    return a * np.exp(-b * TIMES_S) + c * TIMES_S + d


def jacobian(theta):
    a, b, _, _ = theta
    decay = np.exp(-b * TIMES_S)
    return np.vstack([decay, -a * TIMES_S * decay, TIMES_S, np.ones_like(TIMES_S)])


def chi2(theta):
    return np.sum((model(TRUTH) - model(theta)) ** 2) / TIMES_S.size


def solve(start, damping=0.01, dynamic_damping=True, iterations=40):
    parameters = ModelParameters()
    parameters.names = {"source_location": ["latitude", "longitude", "depth", "time_shift"]}
    parameters.theta_fiducial = {"source_location": list(start)}

    def compression_data(params, *args, seed=None):
        theta = params.parameter_to_vector("theta_fiducial", True).astype(float)
        return ScoreCompressionData(theta, model(theta), jacobian(theta), None), None

    data_manager = SimpleNamespace(data_loader=None, compute_required_compression_data=compression_data)
    solver = IterativeLeastSquaresSolver(
        None, [], parameters, data_manager, SimpleNamespace(simulator=None),
        IterativeLeastSquaresParameters(iterations, damping, dynamic_damping=dynamic_damping))
    compressor = GaussianCompressor(compression_data(parameters)[0], ScalarEmpiricalCovariance(1.0, TIMES_S.size))
    origins = []
    step = compressor.compute_theta_MLE
    compressor.compute_theta_MLE = lambda D, **kwargs: origins.append(np.copy(compressor.theta_fiducial)) or step(D, **kwargs)
    final, _ = solver.solve_least_squares(model(TRUTH), compressor, single_step=False)
    return final.theta_fiducial, [chi2(origin) for origin in origins]


def test_chi2_never_rises_between_the_models_steps_are_taken_from():
    _, chi2_at_origins = solve([0.5, 2.0, 0.0, 0.0], damping=0.0001)
    assert all(later <= earlier for earlier, later in zip(chi2_at_origins, chi2_at_origins[1:]))


def test_the_known_minimum_is_reached_from_a_poor_start():
    final, _ = solve([0.5, 2.0, 0.0, 0.0], damping=0.0001)
    np.testing.assert_allclose(final, TRUTH, atol=1e-3)


def test_a_fixed_damping_still_rejects_an_uphill_step():
    _, chi2_at_origins = solve([0.5, 2.0, 0.0, 0.0], damping=0.0001, dynamic_damping=False)
    assert all(later <= earlier for earlier, later in zip(chi2_at_origins, chi2_at_origins[1:]))
