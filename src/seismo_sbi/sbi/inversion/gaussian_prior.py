"""A Gaussian prior on the inferred source parameters.

:class:`GaussianPrior` holds a mean and independent variances in the units and order of the
parameter vector. The Gaussian compressor adds its precision to the Fisher matrix and its score to
the data score, and the Gaussian-likelihood sampler adds its log density to the log-likelihood.
"""
from typing import NamedTuple

import numpy as np


class GaussianPrior(NamedTuple):
    """An independent Gaussian prior: ``mean`` and ``variances``, each ``(n_params,)``."""

    mean: np.ndarray
    variances: np.ndarray

    def precision(self):
        """``(n_params, n_params)``: the inverse of the diagonal prior covariance."""
        return np.linalg.inv(np.diag(self.variances))

    def score(self, theta):
        """``(n_params,)``: the gradient of the log density at ``theta``."""
        return np.dot(self.precision(), self.mean - theta)

    def log_density(self, theta):
        """The log density at ``theta``, up to a constant."""
        return -0.5 * np.dot((theta - self.mean).T, np.dot(self.precision(), (theta - self.mean)))
