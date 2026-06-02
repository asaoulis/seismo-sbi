"""Unit tests for the uniform-on-sphere moment-tensor sampler."""

import numpy as np

from seismo_sbi.priors.moment_tensor import (
    scalar_moment,
    uniform_moment_tensor_on_sphere,
)


def test_scalar_moment_equals_requested_m0():
    rng = np.random.default_rng(0)
    for m0 in [1e12, 1e15, 3.3e16, 1e18]:
        for _ in range(20):
            mt = uniform_moment_tensor_on_sphere(m0, rng)
            assert mt.shape == (6,)
            assert np.isclose(scalar_moment(mt), m0, rtol=1e-10)


def test_orientation_is_isotropic():
    rng = np.random.default_rng(1)
    n = 40_000
    mts = np.array([uniform_moment_tensor_on_sphere(1.0, rng) for _ in range(n)])
    unit = mts / np.linalg.norm(mts, axis=1, keepdims=True)
    # mean direction ~ 0 (no preferred orientation)
    assert np.all(np.abs(unit.mean(axis=0)) < 0.03)
    # covariance of the unit directions ~ (1/6) I (uniform on the 6-sphere)
    cov = np.cov(unit, rowvar=False)
    assert np.allclose(np.diag(cov), 1.0 / 6.0, atol=0.02)
    off_diag = cov - np.diag(np.diag(cov))
    assert np.all(np.abs(off_diag) < 0.02)


def test_scales_linearly_with_m0():
    rng = np.random.default_rng(2)
    rng2 = np.random.default_rng(2)
    a = uniform_moment_tensor_on_sphere(1e15, rng)
    b = uniform_moment_tensor_on_sphere(2e15, rng2)
    # same RNG draw -> same orientation, scaled by the M0 ratio
    assert np.allclose(b, 2.0 * a, rtol=1e-10)
