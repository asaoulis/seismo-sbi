"""The uniform moment-tensor sampler: its scalar moment, and uniformity on the sphere of tensors.

Uniformity is checked against a rotation-invariant reference, the symmetric part of a 3x3 matrix
of independent standard normals, and against Tape and Tape (2015, GJI 202), whose lune coordinates
``v = sin(3 gamma) / 3`` and ``w = 3 pi / 8 - u(beta)`` are uniform for uniform moment tensors.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import kstest

from seismo_sbi.moment_tensor.conventions import scalar_moment
from seismo_sbi.moment_tensor.conventions import create_matrix
from seismo_sbi.moment_tensor.lune_angles import mts6_to_gamma_delta
from seismo_sbi.priors.moment_tensor_sampling import uniform_moment_tensor_on_sphere

N_TENSORS = 100_000
#: Four standard errors of the difference of two proportions near 0.25 at ``N_TENSORS`` each.
PROPORTION_TOLERANCE = 0.008


def sampled_tensors(n_tensors, seed):
    """``(n_tensors, 6)`` draws of unit scalar moment."""
    rng = np.random.default_rng(seed)
    return np.array([uniform_moment_tensor_on_sphere(1.0, rng) for _ in range(n_tensors)])


def as_m6(matrices):
    """``(n, 6)`` up-south-east components of ``(n, 3, 3)`` symmetric matrices."""
    return np.column_stack([matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 2, 2],
                            matrices[:, 0, 1], matrices[:, 0, 2], matrices[:, 1, 2]])


def symmetric_gaussian_reference(n_tensors, seed):
    """``(n_tensors, 6)`` symmetric parts of 3x3 matrices of independent standard normals."""
    matrices = np.random.default_rng(seed).standard_normal((n_tensors, 3, 3))
    return as_m6(0.5 * (matrices + np.transpose(matrices, (0, 2, 1))))


def orientation_statistics(m6):
    """``(P(T-axis plunge > 60 deg), P(|delta| > 30 deg))`` of a set of tensors ``(n, 6)``."""
    _, eigenvectors = np.linalg.eigh(create_matrix(m6))
    t_axis_up = np.abs(eigenvectors[:, 0, 2])
    _, delta_deg = mts6_to_gamma_delta(m6)
    return np.mean(t_axis_up > np.sin(np.radians(60.0))), np.mean(np.abs(delta_deg) > 30.0)


def test_scalar_moment_equals_requested_m0():
    rng = np.random.default_rng(0)
    for m0 in [1e12, 1e15, 3.3e16, 1e18]:
        for _ in range(20):
            mt = uniform_moment_tensor_on_sphere(m0, rng)
            assert mt.shape == (6,)
            assert np.isclose(scalar_moment(mt), m0, rtol=1e-10)


def test_orientations_match_a_rotation_invariant_reference():
    tensors = sampled_tensors(N_TENSORS, seed=1)
    rotations = Rotation.random(N_TENSORS, random_state=2).as_matrix()
    rotated = as_m6(rotations @ create_matrix(tensors) @ np.transpose(rotations, (0, 2, 1)))
    reference = orientation_statistics(symmetric_gaussian_reference(N_TENSORS, seed=3))
    np.testing.assert_allclose(reference, [0.128, 0.252], atol=PROPORTION_TOLERANCE)
    np.testing.assert_allclose(orientation_statistics(tensors), reference, atol=PROPORTION_TOLERANCE)
    np.testing.assert_allclose(orientation_statistics(rotated), reference, atol=PROPORTION_TOLERANCE)


def test_lune_coordinates_are_uniform_as_tape_and_tape_predict():
    gamma_deg, delta_deg = mts6_to_gamma_delta(sampled_tensors(20_000, seed=4))
    beta = np.radians(90.0 - delta_deg)
    v = np.sin(3.0 * np.radians(gamma_deg)) / 3.0
    w = 3.0 * np.pi / 8.0 - (0.75 * beta - 0.5 * np.sin(2.0 * beta) + np.sin(4.0 * beta) / 16.0)
    assert kstest(v, "uniform", args=(-1.0 / 3.0, 2.0 / 3.0)).pvalue > 0.01
    assert kstest(w, "uniform", args=(-3.0 * np.pi / 8.0, 3.0 * np.pi / 4.0)).pvalue > 0.01


def test_scales_linearly_with_m0():
    rng = np.random.default_rng(2)
    rng2 = np.random.default_rng(2)
    a = uniform_moment_tensor_on_sphere(1e15, rng)
    b = uniform_moment_tensor_on_sphere(2e15, rng2)
    # same RNG draw -> same orientation, scaled by the M0 ratio
    assert np.allclose(b, 2.0 * a, rtol=1e-10)
