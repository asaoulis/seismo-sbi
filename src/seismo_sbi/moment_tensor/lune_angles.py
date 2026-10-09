"""Tape and Tape lune angles of moment tensors.

:func:`mts6_to_gamma_delta` maps six-component tensors ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]``
to the lune longitude ``gamma_deg`` and latitude ``delta_deg``; :func:`lam2lune` does the same
from eigenvalues, following Carl Tape's ``lam2lune.m``.
"""

import numpy as np

from seismo_sbi.moment_tensor.conventions import create_matrix


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


def mts6_to_gamma_delta(m6: np.ndarray):
    """
    Vectorized conversion from 6-component MT(s) to Tape & Tape lune (gamma, delta).
    Returns gamma, delta in degrees (shape (n,)).
    """
    M = create_matrix(np.asarray(m6).reshape(-1, 6))
    # eigvalsh returns ascending; reverse for descending
    lam = np.linalg.eigvalsh(M)[:, ::-1]
    gamma, delta, *_ = lam2lune(lam)
    return gamma, delta


def lune_credible_area(gamma, delta, mass: float = 0.95, grid_res=(121, 181),
                       bw_method="scott") -> float:
    """Fraction of the lune's area inside the ``mass`` highest-density region of the (γ, δ)
    posterior, from a Gaussian KDE of the samples on a ``grid_res`` grid; small means the
    source type is tightly resolved, 1 means unconstrained.

    The lune area element is ``cos δ`` over ``γ ∈ [-30, 30]``, ``δ ∈ [-90, 90]`` degrees; the
    region is found by accumulating area-weighted density to ``mass``. ``nan`` with fewer
    than five finite samples or a degenerate cloud.
    """
    from scipy.stats import gaussian_kde

    gamma = np.asarray(gamma, float); delta = np.asarray(delta, float)
    m = np.isfinite(gamma) & np.isfinite(delta)
    gamma, delta = gamma[m], delta[m]
    if gamma.size < 5 or np.allclose(gamma, gamma[0]) and np.allclose(delta, delta[0]):
        return float("nan")
    gg = np.linspace(-30.0, 30.0, grid_res[0])
    dd = np.linspace(-90.0, 90.0, grid_res[1])
    G, Dl = np.meshgrid(gg, dd)
    try:
        kde = gaussian_kde(np.vstack([gamma, delta]), bw_method=bw_method)
    except Exception:
        return float("nan")
    Z = kde(np.vstack([G.ravel(), Dl.ravel()])).reshape(G.shape)
    w = np.cos(np.radians(Dl))
    dens = Z * w
    order = np.argsort(dens.ravel())[::-1]
    mass_sorted = dens.ravel()[order]
    area_sorted = w.ravel()[order]
    cum_mass = np.cumsum(mass_sorted)
    cum_mass /= cum_mass[-1]
    k = int(np.searchsorted(cum_mass, mass))
    k = min(max(k, 0), area_sorted.size - 1)
    area_in = float(np.sum(area_sorted[: k + 1]))
    return area_in / float(np.sum(w))
