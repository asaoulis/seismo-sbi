"""The ensemble MCMC path hands its log-probability to forked workers without pickling it."""
import numpy as np

from seismo_sbi.sbi import likelihood


def test_ensemble_sampler_accepts_an_unpicklable_log_probability():
    centre = np.array([0.3, 0.6])

    def log_probability(theta):
        return -0.5 * np.sum((theta - centre) ** 2) / 0.05**2

    np.random.seed(0)
    samples = likelihood.generate_samples(log_probability, True, 2, 300, 8, burn_in=100, num_processes=2)
    assert samples.shape == (2400, 2)
    np.testing.assert_allclose(samples.mean(axis=0), centre, atol=0.02)
