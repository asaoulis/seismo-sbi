"""Second-order score compression on a quadratic toy model with white noise, where every term is known."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seismo_sbi.sbi.compression.compressor_options import (
    SecondOrderScoreOptions, TheoryOptimalScoreOptions, check_one_extra_gradient_compressor)
from seismo_sbi.sbi.compression.compressors import build_compressor
from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData, SecondOrderCompressor
from seismo_sbi.sbi.datasets.dataset_compressor import DatasetCompressor
from seismo_sbi.sbi.noises.covariances import CovarianceLayout
from seismo_sbi.sbi.noises.diagonal_covariances import ScalarEmpiricalCovariance
from seismo_sbi.sbi.types.parameters import ModelParameters
from seismo_sbi.utils.errors import InvalidConfiguration

REPO = Path(__file__).resolve().parents[2]
N_DATA = 40
SIGMA = 0.5
rng = np.random.default_rng(3)
OFFSET = rng.standard_normal(N_DATA)
LINEAR = rng.standard_normal((3, N_DATA))
QUADRATIC = rng.standard_normal((3, 3, N_DATA))
QUADRATIC = QUADRATIC + QUADRATIC.transpose(1, 0, 2)
THETA_FIDUCIAL = np.array([0.4, -0.2, 0.7])


def toy_data(theta):
    """``a + L theta + theta^T Q theta / 2``, sample by sample."""
    return OFFSET + theta @ LINEAR + 0.5 * np.einsum("i,ijm,j->m", theta, QUADRATIC, theta)


def toy_compression_data():
    gradients = LINEAR + np.einsum("ijm,j->im", QUADRATIC, THETA_FIDUCIAL)
    return ScoreCompressionData(THETA_FIDUCIAL, toy_data(THETA_FIDUCIAL), gradients, None)


def test_the_summary_is_the_score_and_the_hessian_term_of_the_log_likelihood():
    compressor = SecondOrderCompressor(toy_compression_data(), QUADRATIC, ScalarEmpiricalCovariance(SIGMA, N_DATA))
    data = toy_data(np.array([0.5, 0.1, 0.3])) + 0.1 * rng.standard_normal(N_DATA)
    residual = data - toy_data(THETA_FIDUCIAL)
    gradients = toy_compression_data().data_parameter_gradients

    summary = compressor.compress_data_vector(data)

    np.testing.assert_allclose(summary[:3], gradients @ residual / SIGMA**2)
    np.testing.assert_allclose(summary[3:], (QUADRATIC @ residual / SIGMA**2).flatten())
    np.testing.assert_allclose(summary[:3], compressor.compute_score(data))


def toy_parameters():
    parameters = ModelParameters()
    parameters.names = {"source_location": ["latitude", "longitude", "depth"]}
    parameters.theta_fiducial = {"source_location": list(THETA_FIDUCIAL)}
    parameters.stencil_deltas = {"source_location": [0.01, 0.02, 0.01]}
    return parameters


def simulate_and_save(source, output_path, use_fiducial=True):
    np.save(output_path + ".npy", toy_data(np.asarray(source["source_location"], float)))


def test_the_hessian_stencil_recovers_the_second_derivatives_of_a_quadratic_model(tmp_path):
    loader = SimpleNamespace(load_simulation_data_array=lambda path: np.load(str(path) + ".npy"))
    dataset_compressor = DatasetCompressor(loader, simulate_and_save)
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first_order = dataset_compressor.run_derivative_stencil_for_compression_data(toy_parameters(), tmp_path / "first")

    hessian = dataset_compressor.run_hessian_stencil(toy_parameters(), first_order, tmp_path / "second")

    np.testing.assert_allclose(hessian, QUADRATIC, atol=1e-8)


def test_the_example_config_builds_a_second_order_compressor(monkeypatch):
    from seismo_sbi.sbi.configuration import SBI_Configuration

    monkeypatch.chdir(REPO / "examples")
    config = SBI_Configuration.from_file(REPO / "examples/configs/second_order_score.yaml")
    (name, options), = config.compression_methods
    layout = CovarianceLayout(None, N_DATA, N_DATA)

    compressor = build_compressor(options, toy_compression_data(), None, layout, extra_gradients=QUADRATIC)

    assert name == "second_order_score" and isinstance(compressor, SecondOrderCompressor)
    assert compressor.compress_data_vector(toy_data(THETA_FIDUCIAL)).shape == (3 + 9,)


def test_a_run_builds_at_most_one_compressor_from_extra_gradients():
    methods = [("theory_optimal_score", TheoryOptimalScoreOptions("kolb", 1e-8)),
               ("second_order_score", SecondOrderScoreOptions(1e-8))]
    with pytest.raises(InvalidConfiguration, match="extra gradients"):
        check_one_extra_gradient_compressor(methods)
    check_one_extra_gradient_compressor(methods[1:])
