"""Source-type lune drawn with Basemap.

:func:`plot_lune_frame` draws the Tape and Tape lune in a Hammer projection, and the
``plot_*_on_lune`` helpers add scatter points and KDE contours of lune angles
``(gamma_deg, delta_deg)`` to it; :mod:`seismo_sbi.moment_tensor.lune_angles` computes the angles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import gaussian_kde
from pyproj import Geod

if TYPE_CHECKING:
    from mpl_toolkits.basemap import Basemap


# KDE utilities

def kde_on_grid(x, y, xgrid, ygrid, bw_method='scott'):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bw_method)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(grid_points).reshape(X.shape)
    return X, Y, Z, kde


def kde_hpd_contour_levels(Z, levels=(0.6827, 0.9545)):
    Zflat = Z.ravel()
    idx = np.argsort(Zflat)[::-1]
    Zsort = Zflat[idx]
    cdf = np.cumsum(Zsort)
    cdf /= cdf[-1]
    thr = []
    for p in levels:
        k = np.searchsorted(cdf, p)
        k = min(max(k, 0), Zsort.size - 1)
        thr.append(Zsort[k])
    return tuple(thr)


# Basemap Hammer-projected Lune

def plot_lune_frame(ax, frame_color='k', grid_color='lightgray', fontweight='bold',
                    clvd_left=True, clvd_right=True, lon_0=0):
    """Draw the standard Tape & Tape lune frame using a Hammer projection and
    remove any outer frame/spines/ticks. Returns the Basemap instance."""
    try:
        from mpl_toolkits.basemap import Basemap
    except ImportError as error:
        raise ImportError("basemap is required for lune plots: install it with "
                          "`conda install -c conda-forge basemap` or `pip install basemap`.") from error
    g = Geod(ellps='sphere')
    bm = Basemap(projection='hammer', lon_0=lon_0, ax=ax)
    ax.set_aspect('equal')

    # Remove outer map boundary/frame and axis frame/spines/ticks
    try:
        bm.drawmapboundary(fill_color=None, color='none', linewidth=0)
    except Exception:
        pass
    ax.set_frame_on(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Meridian grid lines (gamma = const): -30..30 every 10 deg
    lats = np.arange(-90, 91)
    for lo in range(-30, 31, 10):
        lons = np.ones(len(lats)) * lo
        x, y = bm(lons, lats)
        ax.plot(x, y, lw=0.5, c=grid_color)

    # Lune longitudinal boundaries (gamma=-30 and gamma=30)
    lons = np.ones(len(lats)) * -30
    x, y = bm(lons, lats)
    ax.plot(x, y, lw=1, color=frame_color)
    lons = np.ones(len(lats)) * 30
    x, y = bm(lons, lats)
    ax.plot(x, y, lw=1, c=frame_color)

    # Parallel grid lines (delta = const): -90..90 every 10 deg
    lons = np.arange(-30, 31)
    for la in range(-90, 91, 10):
        lats = np.ones(len(lons)) * la
        x, y = bm(lons, lats)
        ax.plot(x, y, lw=0.5, c=grid_color)

    # Special points and annotations
    # Isotropic points
    x, y = bm(0, 89)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    ax.annotate('+ISO', xy=(x, y*0.99), fontweight=fontweight, ha='center', va='bottom')
    x, y = bm(0, -90)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    ax.annotate('-ISO', xy=(x, -y*.03), fontweight=fontweight, ha='center', va='top')

    # CLVD points
    x, y = bm(30, 0)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    if clvd_right:
        ax.annotate('-CLVD', xy=(0.6, 0.5), xycoords='axes fraction', fontweight=fontweight,
                    rotation='vertical', va='center')
    x, y = bm(-30, 0)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    if clvd_left:
        ax.annotate('+CLVD', xy=(0.4, 0.5), xycoords='axes fraction', fontweight=fontweight,
                    rotation='vertical', ha='right', va='center')

    # Double-couple point
    x, y = bm(0, 0)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    ax.annotate('DC', xy=(x, y*1.03), fontweight=fontweight, ha='center', va='bottom')

    # LVD arc
    lvd_lon = 30
    lvd_lat = np.degrees(np.arcsin(1/np.sqrt(3)))
    x, y = bm(-lvd_lon, lvd_lat)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    x, y = bm(lvd_lon, 90-lvd_lat)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    arc = g.npts(-lvd_lon, lvd_lat, lvd_lon, 90-lvd_lat, 50)
    x, y = bm([p[0] for p in arc], [p[1] for p in arc])
    ax.plot(x, y, lw=1, c=frame_color)

    x, y = bm(-lvd_lon, lvd_lat-90)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    x, y = bm(lvd_lon, -lvd_lat)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    arc = g.npts(-lvd_lon, lvd_lat-90, lvd_lon, -lvd_lat, 50)
    x, y = bm([p[0] for p in arc], [p[1] for p in arc])
    ax.plot(x, y, lw=1, c=frame_color)

    return bm


def project_points_to_lune(bm: Basemap, gamma, delta):
    return bm(gamma, delta)


def plot_scatter_on_lune(ax, bm: Basemap, gamma, delta, **scatter_kwargs):
    x, y = bm(gamma, delta)
    ax.scatter(x, y, **scatter_kwargs)


def plot_kde_contours_on_lune(ax, bm: Basemap, gamma, delta, colors='C0', grid_res=(200, 300),
                               levels=(0.6827, 0.9545), linestyles=('--', '-'), linewidths=(1.5, 1.8)):
    gx = np.linspace(-30, 30, grid_res[0])
    gy = np.linspace(-90, 90, grid_res[1])
    GX, GY = np.meshgrid(gx, gy)
    XX, YY = bm(GX, GY)
    _, _, Z, _ = kde_on_grid(gamma, delta, gx, gy)
    thr = kde_hpd_contour_levels(Z, levels=levels)
    # contour needs strictly increasing levels but HPD thresholds come back decreasing: sort them
    # with their styles and drop duplicates from a degenerate cloud.
    n = min(len(thr), len(linestyles), len(linewidths))
    triples = sorted(zip(thr[:n], linestyles[:n], linewidths[:n]), key=lambda t: t[0])
    lv, ls, lw = [], [], []
    for level, style, width in triples:
        if not lv or level > lv[-1]:
            lv.append(level); ls.append(style); lw.append(width)
    if lv:
        ax.contour(XX, YY, Z, levels=lv, colors=colors, linestyles=ls, linewidths=lw)


def plot_filled_kde_on_lune(ax, bm: Basemap, gamma, delta, cmap='Purples',
                            grid_res=(200, 300), levels=(0.6827, 0.9545),
                            alpha=0.85, area_weighted=False):
    """Filled HPD density of a (γ, δ) cloud on the lune (for pooled catalogue
    samples).  ``area_weighted`` multiplies the KDE by the lune area element
    ``cos δ`` so the HPD regions are in posterior mass, not raw density.
    Returns the ``contourf`` set (usable for a colourbar)."""
    gx = np.linspace(-30, 30, grid_res[0])
    gy = np.linspace(-90, 90, grid_res[1])
    X, Y, Z, _ = kde_on_grid(gamma, delta, gx, gy)
    if area_weighted:
        Z = Z * np.cos(np.radians(Y))
    thr = kde_hpd_contour_levels(Z, levels=levels)
    XX, YY = bm(X, Y)
    # contourf needs strictly increasing levels: sort, drop duplicates, cap with the density maximum.
    lv = []
    for t in sorted(thr):
        if not lv or t > lv[-1]:
            lv.append(t)
    zmax = float(Z.max())
    if not lv or lv[-1] >= zmax:
        return None
    return ax.contourf(XX, YY, Z, levels=lv + [zmax], cmap=cmap, alpha=alpha)


__all__ = [
    'plot_lune_frame',
    'project_points_to_lune', 'plot_scatter_on_lune', 'plot_kde_contours_on_lune',
    'plot_filled_kde_on_lune', 'kde_on_grid', 'kde_hpd_contour_levels'
]
