"""Tape and Tape lune angles of moment tensors.

:func:`mts6_to_gamma_delta` maps six-component tensors ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]``
to the lune longitude ``gamma_deg`` and latitude ``delta_deg``; :func:`lam2lune` does the same
from eigenvalues, following Carl Tape's ``lam2lune.m``.
"""

import numpy as np


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
