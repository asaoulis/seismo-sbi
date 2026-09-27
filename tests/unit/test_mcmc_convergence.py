"""Unit tests for the MCMC split-R-hat convergence diagnostic.

Guards the helper added so the Gaussian-likelihood sampler reports whether its
independent chains have mixed (R-hat ~1) or are stuck in per-walker islands
(R-hat >> 1.1).
"""
import numpy as np

from seismo_sbi.sbi.likelihood import split_rhat


def test_rhat_converged_chains_near_one():
    rng = np.random.default_rng(0)
    # 4 well-mixed chains from the same distribution -> R-hat ~ 1
    chains = [rng.standard_normal((4000, 3)) for _ in range(4)]
    rhat = split_rhat(chains)
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.1)


def test_rhat_island_chains_flagged():
    rng = np.random.default_rng(1)
    # chains trapped in separate modes (different means) -> R-hat >> 1.1
    chains = [rng.standard_normal((2000, 2)) + offset
              for offset in ([0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [50.0, 50.0])]
    rhat = split_rhat(chains)
    assert np.nanmax(rhat) > 1.5


def test_rhat_single_chain_uses_split():
    rng = np.random.default_rng(2)
    # one long well-mixed chain: split-R-hat still defined (~1) via the two halves
    rhat = split_rhat([rng.standard_normal((6000, 2))])
    assert rhat.shape == (2,)
    assert np.all(rhat < 1.2)


def test_rhat_too_short_returns_nan():
    rhat = split_rhat([np.zeros((2, 3))])
    assert rhat.shape == (3,)
    assert np.all(np.isnan(rhat))
