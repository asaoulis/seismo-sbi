"""The lune HPD contours draw 68 % solid and 95 % dashed, as the lune legends say."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seismo_sbi.plotting.lune import plot_kde_contours_on_lune


def test_68_percent_contour_is_solid_and_95_dashed():
    rng = np.random.default_rng(0)
    gamma, delta = rng.normal(0, 5, 400), rng.normal(10, 8, 400)
    fig, ax = plt.subplots()
    drawn = {}

    def capture(*args, levels, linestyles, linewidths, **kwargs):
        drawn.update(levels=levels, linestyles=linestyles, linewidths=linewidths)

    ax.contour = capture
    plot_kde_contours_on_lune(ax, lambda x, y: (x, y), gamma, delta)
    plt.close(fig)
    # the density threshold of the 95 % region is the lower level
    assert drawn["levels"][0] < drawn["levels"][1]
    assert drawn["linestyles"] == ["--", "-"]
    assert drawn["linewidths"][1] > drawn["linewidths"][0]
