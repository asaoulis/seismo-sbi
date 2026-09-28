"""A beachball placed with ``plot_beachball_on_axes`` is centred on the data position it is given."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seismo_sbi.moment_tensor.comparison import pyrocko_mt
from seismo_sbi.plotting.rocko_beachball_patch import plot_beachball_on_axes


def square_axes():
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    return fig, ax


def test_beachball_on_axes_is_centred_on_its_data_position():
    fig, ax = square_axes()
    plot_beachball_on_axes(ax, pyrocko_mt([1e17, -1e17, 0.0, 0.0, 0.0, 0.0]), 0.3, -0.2, diameter=0.5,
                           color_t="red", color_p="red", linewidth=0)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
    rows, columns = np.nonzero((image[..., 0] > 0.7) & (image[..., 1] < 0.4))
    centroid_px = np.array([columns.mean(), image.shape[0] - 1 - rows.mean()])
    np.testing.assert_allclose(centroid_px, ax.transData.transform((0.3, -0.2)), atol=1.5)
    plt.close(fig)
