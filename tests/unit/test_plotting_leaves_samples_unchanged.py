"""The lune, beachball and posterior-predictive figures draw a random subset of the posterior samples
without reordering the caller's array."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np

from seismo_sbi.plotting.distributions import PosteriorPlotter
from seismo_sbi.plotting.posterior_predictive_checks import PosteriorPredictiveChecks
from seismo_sbi.sbi.types.parameter_labels import ParameterInformation


class MomentTensorParameters:
    def vector_to_simulation_inputs(self, vector, only_theta_fiducial=True):
        return {"moment_tensor": np.asarray(vector)[:6]}


def moment_tensor_samples(n=200):
    double_couple = np.array([0.0, 1.0, -1.0, 0.3, 0.2, -0.1]) * 1e16
    return double_couple + 0.2e16 * np.random.default_rng(0).normal(size=(n, 6))


def plotter():
    info = [ParameterInformation(name, "N m") for name in ["Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"]]
    return PosteriorPlotter(None, info, parameters=MomentTensorParameters())


def test_lune_figures_leave_the_samples_unchanged(tmp_path):
    samples = moment_tensor_samples()
    original = samples.copy()
    ensembles = {"posterior": (samples[0], samples, None, None)}
    plotter().plot_lunes(ensembles, num_samples=50, plot_beachballs=False, figsave=tmp_path / "lune.png")
    plotter().plot_lunes_kde(ensembles, num_samples=100, plot_beachballs=False, figsave=tmp_path / "kde.png")
    np.testing.assert_array_equal(samples, original)


def test_beachball_figures_leave_the_samples_unchanged(tmp_path):
    samples = moment_tensor_samples(40)
    original = samples.copy()
    plotter().plot_beachball_samples((samples[0], samples, None, None), plot_path=tmp_path / "beachballs.png")
    np.testing.assert_array_equal(samples, original)


def test_simulate_ensemble_leaves_the_samples_unchanged():
    samples = np.random.default_rng(1).normal(size=(30, 3))
    original = samples.copy()
    ppc = PosteriorPredictiveChecks(simulator=lambda params: np.repeat(params, 4), n_traces=1, n_jobs=2)
    synthetics = ppc.simulate_ensemble(SimpleNamespace(samples=samples), num_samples=10)
    assert synthetics.shape == (10, 12)
    np.testing.assert_array_equal(samples, original)
