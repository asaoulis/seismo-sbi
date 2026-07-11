#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# -----------------------------
# Linear-algebra utilities
# -----------------------------

def sort_eigvals_desc(lam: np.ndarray) -> np.ndarray:
    """
    Sort eigenvalues in descending order along the last axis.
    lam: array of shape (..., 3)
    returns array of shape (..., 3) sorted such that lam[...,0]>=lam[...,1]>=lam[...,2]
    """
    # argsort ascending then reverse
    idx = np.argsort(lam, axis=-1)[..., ::-1]
    return np.take_along_axis(lam, idx, axis=-1)


def eigvals_from_MT(M: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of symmetric 3x3 moment tensor matrix/matrices.
    M: shape (..., 3, 3), assumed symmetric. Returns eigenvalues shape (..., 3)
    """
    # eigh is for Hermitian/symmetric, returns ascending order
    vals = np.linalg.eigvalsh(M)
    # sort descending for consistency with Tape & Tape equations
    return sort_eigvals_desc(vals)


# -----------------------------
# Tape & Tape (2012, 2013) lam2lune in Python
# -----------------------------

def lam2lune(lam: np.ndarray):
    """
    Convert eigenvalues (lam) to lune coordinates (gamma, delta) and extras.

    Input
      lam: array-like of shape (n, 3) or (3,) or (3, n)
           Eigenvalues of moment tensors. Order does not matter (will be sorted).

    Output
      gamma: angle from DC meridian to lune point in degrees (-30 <= gamma <= 30)
      delta: angle from deviatoric plane to lune point in degrees (-90 <= delta <= 90)
      M0: seismic moment = ||lam|| / sqrt(2)
      thetadc: angle from DC to lune point (0 <= thetadc <= 90)
      lamdev: eigenvalues of deviatoric component
      lamiso: eigenvalues of isotropic component
    """
    lam = np.array(lam)

    # Normalize shapes to (n,3)
    if lam.ndim == 1:
        lam = lam.reshape(1, 3)
    elif lam.ndim == 2 and lam.shape[0] == 3 and lam.shape[1] != 3:
        lam = lam.T  # assume (3, n) -> (n, 3)
    elif lam.ndim != 2 or lam.shape[1] != 3:
        raise ValueError("lam must be of shape (n,3), (3,), or (3,n)")

    # Sort eigenvalues (row-wise samples)
    lam = sort_eigvals_desc(lam)
    lam1 = lam[:, 0]
    lam2 = lam[:, 1]
    lam3 = lam[:, 2]

    # magnitude of lambda vector (rho of TT2012)
    rho = np.sqrt(lam1**2 + lam2**2 + lam3**2)

    # seismic moment
    M0 = rho / np.sqrt(2.0)

    # delta
    # numerical safety: if trace(M) = 0, delta = 0
    # numerical safety: clip bdot into [-1, 1]
    trM = np.sum(lam, axis=1)
    delta = np.zeros_like(trM)
    idev = np.nonzero(trM != 0)[0]
    bdot = trM / (np.sqrt(3.0) * rho)
    bdot = np.clip(bdot, -1.0, 1.0)
    deg = 180.0 / np.pi
    if idev.size > 0:
        delta_dev = 90.0 - np.arccos(bdot[idev]) * deg
        delta[idev] = delta_dev

    # gamma (use atan of ratio to match MATLAB exactly)
    num = (-lam1 + 2.0 * lam2 - lam3)
    den = (np.sqrt(3.0) * (lam1 - lam3))
    ratio = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    gamma = np.degrees(np.arctan(ratio))
    # set gamma=0 for isotropic cases lam1 ~= lam3
    XEPS = 1e-6
    biso = np.nonzero(np.abs(lam1 - lam3) < XEPS)[0]
    if biso.size > 0:
        gamma[biso] = 0.0

    # thetadc (TT2013 Eq. S1)
    with np.errstate(invalid='ignore'):  # guard division by zero when rho=0
        arg = (lam1 - lam3) / (np.sqrt(2.0) * rho)
    arg = np.clip(arg, -1.0, 1.0)
    thetadc = np.degrees(np.arccos(arg))

    # deviatoric and isotropic parts
    lamiso_val = (1.0 / 3.0) * trM
    lamiso = np.repeat(lamiso_val[:, None], 3, axis=1)
    lamdev = lam - lamiso

    return gamma, delta, M0, thetadc, lamdev, lamiso


# -----------------------------
# Sampling utilities to build example ensembles
# -----------------------------

def random_rotation_matrix(random_state=None):
    """Generate a random proper rotation matrix (det=+1)."""
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    H = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(H)
    # Ensure right-handed
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1
    return Q


def base_deviatoric(kind: str) -> np.ndarray:
    """Return a base deviatoric eigenvalue vector (trace=0)."""
    kind = kind.lower()
    if kind == 'dc':
        v = np.array([1.0, 0.0, -1.0])
    elif kind == 'clvd+':  # tensile crack-like
        v = np.array([2.0, -1.0, -1.0])
    elif kind == 'clvd-':  # compressive crack-like
        v = np.array([1.0, 1.0, -2.0])
    else:
        raise ValueError("kind must be one of {'dc','clvd+','clvd-'}")
    # normalize to unit rho
    v = v / np.linalg.norm(v)
    return v


def sample_ensemble(n: int,
                    dev_kind: str = 'dc',
                    dev_scale: float = 1.0,
                    dev_jitter: float = 0.2,
                    iso_mean: float = 0.0,
                    iso_std: float = 0.2,
                    random_state=None):
    """
    Build an ensemble of random symmetric moment tensors with controllable
    source-type tendencies.

    Returns
      M: array (n, 3, 3) symmetric moment tensors
      lam_true: array (n, 3) eigenvalues used to construct M (sorted desc)
    """
    rng = np.random.default_rng(random_state)
    base = base_deviatoric(dev_kind)

    Ms = []
    lam_list = []
    for _ in range(n):
        # Deviatoric eigenvalues (trace=0)
        jitter = rng.normal(scale=dev_jitter, size=3)
        jitter -= jitter.mean()  # keep deviatoric
        lamdev = dev_scale * base + jitter
        lamdev -= lamdev.mean()

        # Isotropic component
        iso = rng.normal(loc=iso_mean, scale=iso_std)
        lam = lamdev + iso

        # Random orientation
        R = random_rotation_matrix(rng)
        D = np.diag(lam)
        M = R @ D @ R.T
        # Ensure symmetry numerically
        M = 0.5 * (M + M.T)

        Ms.append(M)
        # Store sorted eigenvalues (desc) for reference
        lam_list.append(sort_eigvals_desc(np.linalg.eigvalsh(M)[None, :])[0])

    return np.stack(Ms, axis=0), np.stack(lam_list, axis=0)


# -----------------------------
# KDE utilities and plotting (rectangular gamma/delta)
# -----------------------------

def kde_on_grid(x, y, xgrid, ygrid, bw_method='scott'):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bw_method)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(grid_points).reshape(X.shape)
    return X, Y, Z, kde


def kde_hpd_contour_levels(Z, levels=(0.6827, 0.9545)):
    """Compute density thresholds such that the highest-density regions contain given probabilities."""
    Zflat = Z.ravel()
    idx = np.argsort(Zflat)[::-1]  # descending
    Zsort = Zflat[idx]
    cdf = np.cumsum(Zsort)
    cdf /= cdf[-1]
    thr = []
    for p in levels:
        k = np.searchsorted(cdf, p)
        # Guard for bounds
        k = min(max(k, 0), Zsort.size - 1)
        thr.append(Zsort[k])
    return tuple(thr)


def plot_lune_background(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(-30, 30)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('gamma (deg)')
    ax.set_ylabel('delta (deg)')
    ax.set_title('Lune (Tape & Tape)')
    # Helpful reference lines
    ax.axvline(0, color='k', lw=0.8, alpha=0.5, linestyle='--', label='DC meridian (gamma=0)')
    ax.axhline(0, color='k', lw=0.8, alpha=0.5, linestyle=':')
    return ax


def plot_scatter_ensembles(gd_list, labels, colors=None, s=12, alpha=0.6, outpath=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_lune_background(ax)
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (g, d) in enumerate(gd_list):
        ax.scatter(g, d, s=s, alpha=alpha, label=labels[i], color=colors[i % len(colors)], edgecolor='none')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    return fig, ax


def plot_kde_contours(gd_list, labels, colors=None, grid_res=(200, 300),
                      levels=(0.6827, 0.9545), outpath=None):
    """
    Plot KDE and 68%/95% HPD contours for each ensemble.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_lune_background(ax)
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    gx = np.linspace(-30, 30, grid_res[0])
    gy = np.linspace(-90, 90, grid_res[1])

    for i, (g, d) in enumerate(gd_list):
        X, Y, Z, _ = kde_on_grid(g, d, gx, gy)
        thr68, thr95 = kde_hpd_contour_levels(Z, levels=levels)
        cs = ax.contour(X, Y, Z, levels=[thr95, thr68], colors=colors[i % len(colors)], linestyles=['--', '-'], linewidths=[1.5, 1.8])

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    return fig, ax


# -----------------------------
# Basemap-based Lune projection (Hammer) and projected plotting
# -----------------------------
from mpl_toolkits.basemap import Basemap
from pyproj import Geod

def plot_lune_frame(ax, frame_color='k', grid_color='lightgray', fontweight='bold',
                    clvd_left=True, clvd_right=True, lon_0=0):
    """Draw the standard Tape & Tape lune frame using a Hammer projection.

    Returns the Basemap instance for projecting (gamma, delta) -> (x, y).
    """
    g = Geod(ellps='sphere')
    bm = Basemap(projection='hammer', lon_0=lon_0, ax=ax)
    # Make sure that the axis has equal aspect ratio
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
    x, y = bm(0, 90)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    ax.annotate('+ISO', xy=(x, y*1.03), fontweight=fontweight, ha='center')
    x, y = bm(0, -90)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    ax.annotate('-ISO', xy=(x, -y*.03), fontweight=fontweight, ha='center', va='top')

    # CLVD points
    x, y = bm(30, 0)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    if clvd_right:
        ax.annotate('-CLVD', xy=(1., 0.5), xycoords='axes fraction', fontweight=fontweight,
                    rotation='vertical', va='center')
    x, y = bm(-30, 0)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    if clvd_left:
        ax.annotate('+CLVD', xy=(0, 0.5), xycoords='axes fraction', fontweight=fontweight,
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


def project_points_to_lune(bm, gamma, delta):
    """Project arrays of (gamma, delta) to Basemap coordinates (x, y)."""
    return bm(gamma, delta)


def plot_scatter_ensembles_on_lune(gd_list, labels, colors=None, s=12, alpha=0.6, outpath=None, lon_0=0):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bm = plot_lune_frame(ax, lon_0=lon_0)
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (g, d) in enumerate(gd_list):
        x, y = project_points_to_lune(bm, g, d)
        ax.scatter(x, y, s=s, alpha=alpha, label=labels[i], color=colors[i % len(colors)], edgecolor='none')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    return fig, ax


def plot_kde_contours_on_lune(gd_list, labels, colors=None, grid_res=(200, 300),
                               levels=(0.6827, 0.9545), outpath=None, lon_0=0):
    """Plot KDE 68%/95% HPD contours for each ensemble on the Hammer-projected lune."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bm = plot_lune_frame(ax, lon_0=lon_0)
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    gx = np.linspace(-30, 30, grid_res[0])
    gy = np.linspace(-90, 90, grid_res[1])

    # Build 2D grids in projected coords for contour plotting
    GX, GY = np.meshgrid(gx, gy)
    XX, YY = bm(GX, GY)

    for i, (g, d) in enumerate(gd_list):
        # KDE on gamma/delta grid
        _, _, Z, _ = kde_on_grid(g, d, gx, gy)
        thr68, thr95 = kde_hpd_contour_levels(Z, levels=levels)
        cs = ax.contour(XX, YY, Z, levels=[thr95, thr68], colors=colors[i % len(colors)],
                        linestyles=['--', '-'], linewidths=[1.5, 1.8])

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    return fig, ax


# -----------------------------
# Demo / main
# -----------------------------

def main():
    rng_seed = 123

    # Ensemble A: near-DC, mostly deviatoric (delta ~ 0)
    MA, lamA = sample_ensemble(
        n=1000,
        dev_kind='dc',
        dev_scale=1.0,
        dev_jitter=0.25,
        iso_mean=0.0,
        iso_std=0.08,
        random_state=rng_seed,
    )

    # Ensemble B: CLVD+ tendency with slight explosive isotropy (delta > 0)
    MB, lamB = sample_ensemble(
        n=1000,
        dev_kind='clvd+',
        dev_scale=1.0,
        dev_jitter=0.25,
        iso_mean=0.25,
        iso_std=0.10,
        random_state=rng_seed + 1,
    )

    # Compute eigenvalues from the moment tensor matrices (as requested), then map to lune
    lamA_fromM = eigvals_from_MT(MA)
    lamB_fromM = eigvals_from_MT(MB)

    gA, dA, *_ = lam2lune(lamA_fromM)
    gB, dB, *_ = lam2lune(lamB_fromM)

    # Scatter plot on rectangular axes (for reference)
    plot_scatter_ensembles(
        gd_list=[(gA, dA), (gB, dB)],
        labels=['Ensemble A (near-DC)', 'Ensemble B (CLVD+ w/ ISO+)'],
        outpath='plots/lune_scatter.png',
    )

    # KDE contours on rectangular axes (for reference)
    plot_kde_contours(
        gd_list=[(gA, dA), (gB, dB)],
        labels=['A', 'B'],
        outpath='plots/lune_kde.png',
    )

    # Scatter on standard lune projection (Hammer)
    plot_scatter_ensembles_on_lune(
        gd_list=[(gA, dA), (gB, dB)],
        labels=['Ensemble A (near-DC)', 'Ensemble B (CLVD+ w/ ISO+)'],
        outpath='plots/lune_scatter_proj.png',
    )

    # KDE contours on standard lune projection (Hammer)
    plot_kde_contours_on_lune(
        gd_list=[(gA, dA), (gB, dB)],
        labels=['A', 'B'],
        outpath='plots/lune_kde_proj.png',
    )

    plt.show()


if __name__ == '__main__':
    main()
