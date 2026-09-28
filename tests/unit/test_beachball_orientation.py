"""Every seismo-sbi beachball path draws a moment tensor the way pyrocko draws ``pyrocko_mt(m6)``.

Rendered beachballs are compared, ray by ray over the lower focal hemisphere, with the sign of the
first motion ``ray . M . ray``, which fixes the P and T axes. Paths that hand a tensor to a pyrocko
or obspy routine are checked by the P and T axes of the tensor they hand over.
"""
import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyrocko.moment_tensor as pmt
import pytest
from obspy.imaging.beachball import beach
from pyrocko.plot import beachball as pyrocko_beachball

import seismo_sbi.plotting.distributions as distributions
import seismo_sbi.plotting.rocko_beachball_patch as rocko_beachball_patch
import seismo_sbi.plotting.velocity_models as velocity_models
from seismo_sbi.moment_tensor.comparison import from_pyrocko, pyrocko_mt
from seismo_sbi.plotting.distributions import PosteriorPlotter
from seismo_sbi.plotting.evaluation import add_decomposition_beachballs

#: (strike, dip, rake) in degrees of a thrust and an oblique strike-slip fault.
MECHANISMS_DEG = [(30.0, 60.0, 90.0), (20.0, 80.0, 10.0)]
#: Largest angle in degrees between a drawn P or T axis and pyrocko's.
AXIS_TOLERANCE_DEG = 5.0


def mechanism_m6(strike_deg, dip_deg, rake_deg):
    """``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m of a double couple with M0 = 1e17 N.m."""
    return np.array(from_pyrocko(pmt.MomentTensor(strike=strike_deg, dip=dip_deg, rake=rake_deg,
                                                  scalar_moment=1e17)))


def east_west_mirror(m6):
    """The mirror image of ``m6`` through the north-south vertical plane: m_rp, m_tp negated."""
    return np.asarray(m6) * np.array([1, 1, 1, 1, -1, -1])


def lower_hemisphere_rays():
    """Unit ray directions ``(n_rays, 3)`` in north-east-down, on a grid over the lower hemisphere."""
    azimuth, takeoff = np.meshgrid(np.radians(np.arange(0.0, 360.0, 9.0)),
                                   np.radians(np.arange(5.0, 90.0, 10.0)))
    azimuth, takeoff = azimuth.ravel(), takeoff.ravel()
    return np.column_stack([np.sin(takeoff) * np.cos(azimuth), np.sin(takeoff) * np.sin(azimuth),
                            np.cos(takeoff)])


def lambert_position(rays):
    """Equal-area position ``(n_rays, 2)``, east then north, of each ray on a unit beachball."""
    radius = np.sqrt(2.0) * np.sin(np.arccos(rays[:, 2]) / 2.0)
    horizontal = np.hypot(rays[:, 0], rays[:, 1])
    return np.column_stack([radius * rays[:, 1] / horizontal, radius * rays[:, 0] / horizontal])


def drawn_colours(fig, points_px):
    """``(is_red, is_white)`` of the rendered figure at each display point ``(n, 2)`` in pixels."""
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
    rows = image.shape[0] - 1 - np.round(points_px[:, 1]).astype(int)
    colours = image[rows, np.round(points_px[:, 0]).astype(int)]
    is_red = (colours[:, 0] > 0.7) & (colours[:, 1] < 0.4) & (colours[:, 2] < 0.4)
    return is_red, np.all(colours > 0.85, axis=1)


def first_motion_agreement(fig, centre_px, radius_px, m6):
    """Fraction of well-resolved rays whose drawn colour (red = compression) matches ``m6``.

    Rays that land on an outline, neither red nor white, are left out.
    """
    rays = lower_hemisphere_rays()
    first_motion = np.einsum("ni,ij,nj->n", rays, pyrocko_mt(m6).m(), rays)
    resolved = np.abs(first_motion) > 0.3 * np.abs(first_motion).max()
    points_px = np.asarray(centre_px) + radius_px * lambert_position(rays[resolved])
    is_red, is_white = drawn_colours(fig, points_px)
    coloured = is_red | is_white
    assert coloured.mean() > 0.8
    return np.mean(is_red[coloured] == (first_motion[resolved][coloured] > 0))


def square_axes():
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    return fig, ax


def unit_ball_in_pixels(ax):
    """Display centre and radius in pixels of a beachball of radius 1 at the data origin."""
    centre_px = ax.transData.transform((0.0, 0.0))
    return centre_px, ax.transData.transform((1.0, 0.0))[0] - centre_px[0]


def axis_angle_deg(axis_a, axis_b):
    """Angle in degrees between two undirected axes."""
    cosine = abs(np.dot(axis_a, axis_b)) / np.linalg.norm(axis_a) / np.linalg.norm(axis_b)
    return np.degrees(np.arccos(min(cosine, 1.0)))


def assert_same_p_and_t_axes(drawn, m6):
    """``drawn`` (a pyrocko tensor or a 3x3 north-east-down matrix) has the P and T axes of ``m6``."""
    drawn, reference = pmt.as_mt(drawn), pyrocko_mt(m6)
    assert axis_angle_deg(drawn.p_axis(), reference.p_axis()) < AXIS_TOLERANCE_DEG
    assert axis_angle_deg(drawn.t_axis(), reference.t_axis()) < AXIS_TOLERANCE_DEG


class MomentTensorParameters:
    """Stands in for ``ModelParameters``: a vector's first six entries are its moment tensor."""

    def vector_to_simulation_inputs(self, vector, only_theta_fiducial=True):
        return {"moment_tensor": np.asarray(vector)[:6]}


def posterior_plotter():
    return PosteriorPlotter(data_scaler=None, parameters_info=[], parameters=MomentTensorParameters())


def samples_around(m6, n_samples=400):
    """Posterior-like samples ``(n_samples, 6)``: ``m6`` with 1 % scatter."""
    rng = np.random.default_rng(0)
    return m6 + 0.01 * np.abs(m6).max() * rng.standard_normal((n_samples, 6))


def pyrocko_drawing(ax, m6):
    pyrocko_beachball.plot_beachball_mpl(pyrocko_mt(m6), ax, position=(0.0, 0.0), size=2.0,
                                         size_units="data", color_t="red", linewidth=0)


def obspy_drawing(ax, m6):
    ax.add_collection(beach(m6, xy=(0.0, 0.0), width=2.0, facecolor="red", linewidth=0))


def patched_pyrocko_drawing(ax, m6):
    rocko_beachball_patch.plot_beachball_on_axes(ax, pyrocko_mt(m6), 0.0, 0.0, diameter=2.0 / 2.2,
                                                 color_t="red", linewidth=0)


DRAWINGS = [pyrocko_drawing, obspy_drawing, patched_pyrocko_drawing]


@pytest.mark.parametrize("mechanism_deg", MECHANISMS_DEG)
@pytest.mark.parametrize("drawing", DRAWINGS)
def test_beachball_routines_draw_the_first_motions_of_m6(drawing, mechanism_deg):
    m6 = mechanism_m6(*mechanism_deg)
    fig, ax = square_axes()
    drawing(ax, m6)
    centre_px, radius_px = unit_ball_in_pixels(ax)
    assert first_motion_agreement(fig, centre_px, radius_px, m6) > 0.97
    plt.close(fig)


@pytest.mark.parametrize("mechanism_deg", MECHANISMS_DEG)
def test_first_motion_check_rejects_the_east_west_mirror(mechanism_deg):
    m6 = mechanism_m6(*mechanism_deg)
    fig, ax = square_axes()
    pyrocko_drawing(ax, east_west_mirror(m6))
    centre_px, radius_px = unit_ball_in_pixels(ax)
    assert first_motion_agreement(fig, centre_px, radius_px, m6) < 0.8
    plt.close(fig)


@pytest.mark.parametrize("mechanism_deg", MECHANISMS_DEG)
def test_add_beachball_plot_draws_the_first_motions_of_m6(mechanism_deg):
    m6 = mechanism_m6(*mechanism_deg)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    posterior_plotter().add_beachball_plot(ax, "", m6, (5.0, 0.0), col="red")
    radius_px = 50 * 0.5 / 72.0 * fig.dpi
    assert first_motion_agreement(fig, ax.transData.transform((0.0, 0.0)), radius_px, m6) > 0.97
    plt.close(fig)


@pytest.mark.parametrize("mechanism_deg", MECHANISMS_DEG)
def test_projection_samples_hand_obspy_m6_unchanged(monkeypatch, tmp_path, mechanism_deg):
    m6 = mechanism_m6(*mechanism_deg)
    handed = []
    monkeypatch.setattr(distributions, "beach",
                        lambda fm, **kwargs: handed.append(np.asarray(fm)) or beach(fm, **kwargs))
    posterior_plotter().plot_beachball_projection_samples(np.tile(m6, (3, 1)),
                                                          figsave=tmp_path / "projection.png")
    assert len(handed) == 3
    for fm in handed:
        np.testing.assert_allclose(fm, m6)


@pytest.mark.parametrize("mechanism_deg", MECHANISMS_DEG)
def test_fuzzy_beachball_hands_pyrocko_the_axes_of_m6(monkeypatch, tmp_path, mechanism_deg):
    m6 = mechanism_m6(*mechanism_deg)
    handed = []
    monkeypatch.setattr(pyrocko_beachball, "plot_fuzzy_beachball_mpl_pixmap",
                        lambda mts, axes, best_mt=None, **kwargs: handed.extend(list(mts) + [best_mt]))
    posterior_plotter().plot_fuzzy_beachball_samples(samples_around(m6, 20), m6,
                                                     figsave=tmp_path / "fuzzy.png")
    assert len(handed) == 21
    for mt in handed:
        assert_same_p_and_t_axes(mt, m6)


def capture_beachballs_on_axes(monkeypatch, module):
    handed = []
    monkeypatch.setattr(module, "plot_beachball_on_axes", lambda ax, mt, *args, **kwargs: handed.append(mt))
    return handed


@pytest.mark.parametrize("mechanism_deg", MECHANISMS_DEG)
@pytest.mark.parametrize("lune_plot", ["plot_lunes", "plot_lunes_kde"])
def test_lune_beachballs_carry_the_axes_of_m6(monkeypatch, tmp_path, lune_plot, mechanism_deg):
    m6 = mechanism_m6(*mechanism_deg)
    handed = capture_beachballs_on_axes(monkeypatch, distributions)
    getattr(posterior_plotter(), lune_plot)({"posterior": (m6, samples_around(m6))},
                                            figsave=tmp_path / "lune.png")
    assert len(handed) == 4
    for mt in handed:
        assert_same_p_and_t_axes(mt, m6)


@pytest.mark.parametrize("mechanism_deg", MECHANISMS_DEG)
def test_decomposition_beachball_carries_the_axes_of_m6(monkeypatch, mechanism_deg):
    m6 = mechanism_m6(*mechanism_deg)
    handed = capture_beachballs_on_axes(monkeypatch, rocko_beachball_patch)
    fig, ax = plt.subplots()
    add_decomposition_beachballs(ax, m6)
    assert len(handed) == 1
    assert_same_p_and_t_axes(handed[0], m6)
    plt.close(fig)


@pytest.mark.parametrize("mechanism_deg", MECHANISMS_DEG)
def test_map_beachball_carries_the_axes_of_m6(monkeypatch, mechanism_deg):
    m6 = mechanism_m6(*mechanism_deg)
    handed = []
    monkeypatch.setattr(velocity_models, "plot_beachball_mpl", lambda mt, ax, **kwargs: handed.append(mt))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    event = {"moment_tensor": list(m6), "source_location": [37.6, -118.9, 5.0, 0.0], "name": "event"}
    velocity_models.add_event_to_map(ax, event)
    assert len(handed) == 1
    assert_same_p_and_t_axes(handed[0], m6)
    plt.close(fig)
