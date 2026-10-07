"""The Gaussian prior on the source parameters, in the compressor, the likelihood and least squares."""
import numpy as np
import pytest

from seismo_sbi.sbi.inversion.gaussian_prior import GaussianPrior

PRIOR = GaussianPrior(mean=np.array([0.5, -0.2, 0.1, 0.0, 1.0, -1.0]),
                      variances=np.array([0.4, 1.0, 2.0, 0.25, 0.1, 3.0]))


def test_the_prior_terms_are_the_gaussian_density_and_its_gradient():
    theta = np.array([1.0, 0.0, -1.0, 0.5, 0.2, 0.3])
    residual = theta - PRIOR.mean
    assert PRIOR.log_density(theta) == pytest.approx(-0.5 * np.sum(residual ** 2 / PRIOR.variances))
    assert np.allclose(PRIOR.score(theta), -residual / PRIOR.variances)
    assert np.allclose(PRIOR.precision(), np.diag(1 / PRIOR.variances))
