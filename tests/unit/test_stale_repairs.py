"""A code path that could not run: the coverage figure."""
import importlib

import matplotlib

matplotlib.use("Agg")

import numpy as np
from seismo_sbi.plotting.coverage import plot_credibility_levels_histograms_dictionary


def test_the_credibility_histogram_figure_draws():
    # importing sbi.analysis (as the pipeline does) drops scienceplots' styles; register again
    importlib.reload(importlib.import_module("scienceplots"))
    alphas = np.linspace(0.0, 1.0, 11)
    coverage = (np.tile(alphas, (5, 1)) + 0.01 * np.arange(5)[:, None], alphas)
    plot_credibility_levels_histograms_dictionary({"run": coverage}, ["C0"])
