"""ModelParameters flattens parameter blocks of different sizes into one vector."""
import numpy as np

from seismo_sbi.sbi.types.parameters import ModelParameters


def _moment_tensor_and_location():
    params = ModelParameters()
    params.theta_fiducial["moment_tensor"] = [1e15] * 6
    params.bounds["moment_tensor"] = [[-5e16] * 6, [5e16] * 6]
    params.theta_fiducial["source_location"] = [39.9, -29.9, 8.0, 0.0]
    params.bounds["source_location"] = [[39.9, -29.9, 8.5, -1.0], [39.9, -29.9, 24.5, 1.0]]
    return params


def test_bounds_of_blocks_with_different_sizes_flatten_in_block_order():
    bounds = _moment_tensor_and_location().parameter_to_vector("bounds", only_theta_fiducial=True)

    assert len(bounds) == 4
    assert list(bounds[0]) == [-5e16] * 6
    assert list(bounds[3]) == [39.9, -29.9, 24.5, 1.0]


def test_fiducials_still_flatten_to_a_float_vector():
    theta = _moment_tensor_and_location().parameter_to_vector("theta_fiducial")

    assert theta.dtype == np.float64
    assert theta.shape == (10,)
