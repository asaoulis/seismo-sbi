"""The ensemble MCMC path: forked workers inherit the log-probability, spawned ones receive it pickled."""
import multiprocessing

import numpy as np

from seismo_sbi.sbi.inversion import likelihood

CENTRE = np.array([0.3, 0.6])


def gaussian_log_probability(theta):
    return -0.5 * np.sum((theta - CENTRE) ** 2) / 0.05**2


def draw(log_probability):
    np.random.seed(0)
    return likelihood.generate_samples(log_probability, True, 2, 300, 8, burn_in=100, num_processes=2)


def test_ensemble_sampler_accepts_an_unpicklable_log_probability():
    samples = draw(lambda theta: gaussian_log_probability(theta))
    assert samples.shape == (2400, 2)
    np.testing.assert_allclose(samples.mean(axis=0), CENTRE, atol=0.02)


def test_ensemble_sampler_pickles_the_log_probability_where_fork_is_unavailable(monkeypatch):
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    samples = draw(gaussian_log_probability)
    np.testing.assert_allclose(samples.mean(axis=0), CENTRE, atol=0.02)
