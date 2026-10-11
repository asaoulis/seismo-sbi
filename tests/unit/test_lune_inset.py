"""The zoomed inset of the lune KDE figure labels its axes gamma and delta with or without LaTeX."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seismo_sbi.plotting.distributions import PosteriorPlotter
from seismo_sbi.sbi.types.parameter_labels import ParameterInformation


class MomentTensorParameters:
    def vector_to_simulation_inputs(self, vector, only_theta_fiducial=True):
        return {"moment_tensor": np.asarray(vector)[:6]}


def test_lune_kde_inset_draws_without_usetex(tmp_path):
    double_couple = np.array([0.0, 1.0, -1.0, 0.3, 0.2, -0.1]) * 1e16
    samples = double_couple + 0.2e16 * np.random.default_rng(0).normal(size=(300, 6))
    info = [ParameterInformation(name, "N m") for name in ["Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"]]
    plotter = PosteriorPlotter(None, info, parameters=MomentTensorParameters())
    with plt.rc_context({"text.usetex": False}):
        plotter.plot_lunes_kde({"posterior": (double_couple, samples, None, None)}, num_samples=300,
                               plot_beachballs=False, plot_inset=True, figsave=tmp_path / "kde.png")
    assert (tmp_path / "kde.png").stat().st_size > 0
