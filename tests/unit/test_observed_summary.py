"""The observed summary and its scaler: the MLE for a first-order compressor, the compressed data otherwise."""
from types import SimpleNamespace

import numpy as np
import pytest

from seismo_sbi.sbi.compression.gaussian import Compressor, GaussianCompressor
from seismo_sbi.sbi.event_pipelines import observed_summary_and_scaler
from seismo_sbi.utils.errors import PipelineStateError

FIDUCIAL = SimpleNamespace(theta_fiducial=np.array([1.0, 2.0]))


class LongSummaryCompressor(Compressor):
    """Four numbers per data vector: its sum, its mean, its first value and a constant."""

    def compress_data_vector(self, D, matmul_callable=None):
        return np.array([D.sum(), D.mean(), D[0], 7.0])


def test_a_parameter_estimate_is_the_fiducial_scaled_like_the_parameters():
    compressor = GaussianCompressor.__new__(GaussianCompressor)
    parameter_scaler = object()
    x_0, scaler = observed_summary_and_scaler(compressor, FIDUCIAL, None, np.zeros((5, 2)), parameter_scaler)
    assert x_0 is FIDUCIAL.theta_fiducial and scaler is parameter_scaler


def test_a_longer_summary_is_the_compressed_observation_scaled_over_the_training_set():
    training = np.array([[0.0, -1.0, 2.0, 7.0], [10.0, 1.0, 4.0, 7.0]])
    data_vector = np.array([3.0, 1.0, 1.0])
    x_0, scaler = observed_summary_and_scaler(LongSummaryCompressor(), FIDUCIAL, data_vector, training, None)
    np.testing.assert_allclose(x_0, [5.0, 5.0 / 3.0, 3.0, 7.0])
    np.testing.assert_allclose(scaler.transform(training), [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])


def test_a_longer_summary_needs_the_observation():
    with pytest.raises(PipelineStateError, match="data_vector"):
        observed_summary_and_scaler(LongSummaryCompressor(), FIDUCIAL, None, np.zeros((2, 4)), None)
