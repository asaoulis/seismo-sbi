"""The Fisher-constrained prior box stays inside each parameter's configured bounds."""
from types import SimpleNamespace

import numpy as np

from seismo_sbi.sbi.pipeline import SingleEventPipeline
from seismo_sbi.sbi.types.parameters import ModelParameters


def pipeline_with(fisher_sigmas, theta_fiducial):
    parameters = ModelParameters()
    parameters.names = {"moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
                        "source_location": ["latitude", "longitude", "depth", "time_shift"]}
    parameters.theta_fiducial = {"moment_tensor": list(theta_fiducial[:6]),
                                 "source_location": list(theta_fiducial[6:])}
    parameters.bounds = {"moment_tensor": [[-5e16] * 6, [5e16] * 6],
                         "source_location": [[38.0, -29.0, 5.0, -2.0], [39.0, -28.0, 30.0, 6.0]]}
    pipeline = SingleEventPipeline.__new__(SingleEventPipeline)
    pipeline.parameters = parameters
    pipeline.compressors = {"score": SimpleNamespace(Fisher_mat_inverse=np.diag(np.square(fisher_sigmas)))}
    return pipeline


def test_a_fisher_box_wider_than_the_location_bounds_is_clipped_to_them():
    theta = np.array([1e15] * 6 + [38.5, -28.5, 12.0, 1.0])
    sigmas = np.array([1e14] * 6 + [1.0, 1.0, 10.0, 5.0])
    pipeline = pipeline_with(sigmas, theta)
    dataset = SimpleNamespace(use_fisher_to_constrain_bounds=3, sampling_method={})

    pipeline.use_fisher_to_constrain_bounds("score", dataset, SimpleNamespace(theta_fiducial=theta))

    np.testing.assert_allclose(pipeline.parameters.bounds["source_location"],
                               [[38.0, -29.0, 5.0, -2.0], [39.0, -28.0, 30.0, 6.0]])
    np.testing.assert_allclose(pipeline.parameters.bounds["moment_tensor"], [[7e14] * 6, [1.3e15] * 6])
    assert dataset.sampling_method == {"moment_tensor": "uniform", "source_location": "uniform"}


def test_a_fisher_box_inside_the_bounds_is_kept():
    theta = np.array([1e15] * 6 + [38.5, -28.5, 12.0, 1.0])
    sigmas = np.array([1e14] * 6 + [0.1, 0.1, 1.0, 0.5])
    pipeline = pipeline_with(sigmas, theta)
    dataset = SimpleNamespace(use_fisher_to_constrain_bounds=2, sampling_method={})

    pipeline.use_fisher_to_constrain_bounds("score", dataset, SimpleNamespace(theta_fiducial=theta))

    np.testing.assert_allclose(pipeline.parameters.bounds["source_location"],
                               [[38.3, -28.7, 10.0, 0.0], [38.7, -28.3, 14.0, 2.0]])
