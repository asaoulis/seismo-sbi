"""A corner plot draws with LaTeX and a serif font and leaves the caller's matplotlib settings as they were."""
import shutil

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from seismo_sbi.plotting.distributions import PosteriorPlotter
from seismo_sbi.sbi.types.parameter_labels import ParameterInformation


@pytest.mark.skipif(shutil.which("latex") is None, reason="the corner plot typesets with LaTeX")
def test_corner_plot_restores_the_rc_settings(tmp_path):
    before = (matplotlib.rcParams["text.usetex"], list(matplotlib.rcParams["font.family"]),
              matplotlib.rcParams["text.latex.preamble"])
    samples = np.random.default_rng(0).normal(size=(500, 2))
    plotter = PosteriorPlotter(None, [ParameterInformation("a", ""), ParameterInformation("b", "km")])
    plotter.plot_chain_consumer({"posterior": (np.zeros(2), samples)}, figsave=tmp_path / "corner.png",
                                tick_font_size=10)
    after = (matplotlib.rcParams["text.usetex"], list(matplotlib.rcParams["font.family"]),
             matplotlib.rcParams["text.latex.preamble"])
    assert (tmp_path / "corner.png").stat().st_size > 0
    assert after == before
