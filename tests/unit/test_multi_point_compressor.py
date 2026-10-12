"""Multi-point score compression on a linear toy model with white noise, where every estimate is the
generalised-least-squares solution."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seismo_sbi.sbi.compression.compressors import build_compressor
from seismo_sbi.sbi.compression.gaussian import MultiPointGaussianCompressor, ScoreCompressionData
from seismo_sbi.sbi.data_manager import DataManager
from seismo_sbi.sbi.noises.covariances import CovarianceLayout
from seismo_sbi.sbi.noises.diagonal_covariances import ScalarEmpiricalCovariance
from seismo_sbi.sbi.types.parameters import ModelParameters
from seismo_sbi.utils.errors import InvalidConfiguration

REPO = Path(__file__).resolve().parents[2]
N_DATA = 60
SIGMA = 0.3
rng = np.random.default_rng(5)
OFFSET = rng.standard_normal(N_DATA)
GRADIENTS = rng.standard_normal((3, N_DATA))


def toy_compression_data(theta):
    theta = np.asarray(theta, float)
    return ScoreCompressionData(theta, OFFSET + theta @ GRADIENTS, GRADIENTS, None)


def test_every_estimate_of_a_linear_model_is_the_least_squares_solution():
    points = [toy_compression_data([1.0, 0.0, -1.0]), toy_compression_data([-2.0, 0.5, 0.3])]
    compressor = MultiPointGaussianCompressor(toy_compression_data([0.2, 0.2, 0.2]), points,
                                              ScalarEmpiricalCovariance(SIGMA, N_DATA))
    data = OFFSET + np.array([0.7, -0.4, 0.1]) @ GRADIENTS + SIGMA * rng.standard_normal(N_DATA)
    least_squares, *_ = np.linalg.lstsq(GRADIENTS.T, data - OFFSET, rcond=None)

    summary = compressor.compress_data_vector(data)

    np.testing.assert_allclose(summary, np.tile(least_squares, 3), rtol=1e-9)


def toy_parameters():
    parameters = ModelParameters()
    parameters.names = {"source_location": ["latitude", "longitude", "depth"]}
    parameters.theta_fiducial = {"source_location": [0.2, 0.2, 0.2]}
    return parameters


def test_each_fixed_source_is_compressed_once_and_must_name_every_parameter():
    stencils = []
    data_manager = DataManager(None, None)
    data_manager.compute_compression_data_from_stencil = lambda parameters: stencils.append(
        parameters.theta_fiducial) or toy_compression_data(parameters.theta_fiducial["source_location"])
    source = {"source_location": [1.0, 0.0, -1.0]}

    first = data_manager.compression_data_at_fixed_source(toy_parameters(), source)
    again = data_manager.compression_data_at_fixed_source(toy_parameters(), source)

    assert again is first and stencils == [{"source_location": [1.0, 0.0, -1.0]}]
    with pytest.raises(InvalidConfiguration, match="source_location"):
        data_manager.compression_data_at_fixed_source(toy_parameters(), {"moment_tensor": [1.0] * 6})


def test_the_example_config_builds_a_multi_point_compressor(monkeypatch):
    from seismo_sbi.sbi.configuration import SBI_Configuration

    monkeypatch.chdir(REPO / "examples")
    config = SBI_Configuration.from_file(REPO / "examples/configs/multi_optimal_score.yaml")
    (name, options), = config.compression_methods
    points = [toy_compression_data([1.0, 0.0, -1.0])] * len(options.fiducials)

    compressor = build_compressor(options, toy_compression_data([0.2, 0.2, 0.2]), None,
                                  CovarianceLayout(None, N_DATA, N_DATA), extra_gradients=points)

    assert name == "multi_optimal_score" and len(options.fiducials) == 2
    assert compressor.compress_data_vector(OFFSET).shape == (3 * 3,)


def test_the_fixed_sources_are_a_list_of_parameter_dictionaries():
    from seismo_sbi.sbi.compression.compressor_options import compressor_options

    with pytest.raises(InvalidConfiguration, match="fiducials"):
        compressor_options("multi_optimal_score", {"noise_level": 1e-8, "fiducials": [[1.0, 2.0]]})
