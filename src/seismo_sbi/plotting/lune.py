# Minimal, reusable lune utilities: conversion to Tape & Tape gamma/delta and Basemap-projected plotting
import numpy as np
from scipy.stats import gaussian_kde
from mpl_toolkits.basemap import Basemap
from pyproj import Geod
import matplotlib.pyplot as plt


# -----------------------------
# Core math: eigenvalue handling and lam2lune
# -----------------------------

def sort_eigvals_desc(lam: np.ndarray) -> np.ndarray:
    idx = np.argsort(lam, axis=-1)[..., ::-1]
    return np.take_along_axis(lam, idx, axis=-1)


def lam2lune(lam: np.ndarray):
    """
    Convert eigenvalues (lam) to lune coordinates (gamma, delta) and extras.
    Implements the same equations as Carl Tape's lam2lune.m.

    lam: array-like of shape (n, 3) or (3,), or (3,n)
    returns: gamma (deg), delta (deg), M0, thetadc (deg), lamdev, lamiso
    """
    lam = np.array(lam)
    if lam.ndim == 1:
        lam = lam.reshape(1, 3)
    elif lam.ndim == 2 and lam.shape[0] == 3 and lam.shape[1] != 3:
        lam = lam.T
    elif lam.ndim != 2 or lam.shape[1] != 3:
        raise ValueError("lam must be shape (n,3), (3,), or (3,n)")

    lam = sort_eigvals_desc(lam)
    lam1 = lam[:, 0]
    lam2 = lam[:, 1]
    lam3 = lam[:, 2]

    rho = np.sqrt(lam1**2 + lam2**2 + lam3**2)
    M0 = rho / np.sqrt(2.0)

    trM = np.sum(lam, axis=1)
    delta = np.zeros_like(trM)
    idev = np.nonzero(trM != 0)[0]
    bdot = trM / (np.sqrt(3.0) * rho)
    bdot = np.clip(bdot, -1.0, 1.0)
    if idev.size > 0:
        delta_dev = 90.0 - np.degrees(np.arccos(bdot[idev]))
        delta[idev] = delta_dev

    num = (-lam1 + 2.0 * lam2 - lam3)
    den = (np.sqrt(3.0) * (lam1 - lam3))
    ratio = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    gamma = np.degrees(np.arctan(ratio))
    XEPS = 1e-6
    biso = np.nonzero(np.abs(lam1 - lam3) < XEPS)[0]
    if biso.size > 0:
        gamma[biso] = 0.0

    with np.errstate(invalid='ignore'):
        arg = (lam1 - lam3) / (np.sqrt(2.0) * rho)
    arg = np.clip(arg, -1.0, 1.0)
    thetadc = np.degrees(np.arccos(arg))

    lamiso_val = (1.0 / 3.0) * trM
    lamiso = np.repeat(lamiso_val[:, None], 3, axis=1)
    lamdev = lam - lamiso

    return gamma, delta, M0, thetadc, lamdev, lamiso


def m6_to_matrix(m6: np.ndarray) -> np.ndarray:
    """
    Convert 6-component moment tensor(s) [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    into 3x3 symmetric matrices. Accepts shape (6,) or (n,6).
    """
    m6 = np.asarray(m6)
    if m6.ndim == 1:
        m6 = m6.reshape(1, 6)
    M = np.zeros((m6.shape[0], 3, 3), dtype=m6.dtype)
    M[:, 0, 0] = m6[:, 0]
    M[:, 1, 1] = m6[:, 1]
    M[:, 2, 2] = m6[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = m6[:, 3]
    M[:, 0, 2] = M[:, 2, 0] = m6[:, 4]
    M[:, 1, 2] = M[:, 2, 1] = m6[:, 5]
    return M


def mts6_to_gamma_delta(m6: np.ndarray):
    """
    Vectorized conversion from 6-component MT(s) to Tape & Tape lune (gamma, delta).
    Returns gamma, delta in degrees (shape (n,)).
    """
    M = m6_to_matrix(m6)
    # eigvalsh returns ascending; reverse for descending
    lam = np.linalg.eigvalsh(M)[:, ::-1]
    gamma, delta, *_ = lam2lune(lam)
    return gamma, delta


# -----------------------------
# KDE utilities
# -----------------------------

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


# -----------------------------
# Basemap Hammer-projected Lune
# -----------------------------

def plot_lune_frame(ax, frame_color='k', grid_color='lightgray', fontweight='bold',
                    clvd_left=True, clvd_right=True, lon_0=0):
    """Draw the standard Tape & Tape lune frame using a Hammer projection and
    remove any outer frame/spines/ticks. Returns the Basemap instance."""
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
    ax.contour(XX, YY, Z, levels=list(thr), colors=colors, linestyles=list(linestyles), linewidths=list(linewidths))


__all__ = [
    'lam2lune', 'm6_to_matrix', 'mts6_to_gamma_delta', 'plot_lune_frame',
    'project_points_to_lune', 'plot_scatter_on_lune', 'plot_kde_contours_on_lune',
    'kde_on_grid', 'kde_hpd_contour_levels'
]
