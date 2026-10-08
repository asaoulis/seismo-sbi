"""Moment-tensor component conventions, the 3x3 matrix and the scalar moment.

``m6`` is ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m (up, south, east). ``create_matrix``
builds the symmetric 3x3 tensor from six components in that index order, the matrix pyrocko's
``m_up_south_east`` takes. ``scalar_moment`` is the full-tensor moment of Silver and Jordan
(1982), ``M0 = sqrt(0.5 * sum_ij M_ij^2)``, in which each off-diagonal component counts twice, and
``moment_magnitude`` is the IASPEI (2013) ``Mw = (2/3) (log10 M0 - 9.1)`` with ``M0`` in N.m, and
``magnitude_to_m0`` its inverse.
"""
import numpy as np


def create_matrix(moment_tensor_sol):
    """The symmetric tensor ``[[m_rr, m_rt, m_rp], [m_rt, m_tt, m_tp], [m_rp, m_tp, m_pp]]``.

    ``(3, 3)`` for one tensor ``(6,)``, ``(n, 3, 3)`` for a cloud ``(n, 6)``, in the dtype of the input.
    """
    m6 = np.asarray(moment_tensor_sol)
    M = np.zeros(m6.shape[:-1] + (3, 3), dtype=m6.dtype)
    M[..., 0, 0] = m6[..., 0]
    M[..., 1, 1] = m6[..., 1]
    M[..., 2, 2] = m6[..., 2]
    M[..., 0, 1] = M[..., 1, 0] = m6[..., 3]
    M[..., 0, 2] = M[..., 2, 0] = m6[..., 4]
    M[..., 1, 2] = M[..., 2, 1] = m6[..., 5]
    return M


def scalar_moment(mt6):
    """``M0 = sqrt(0.5 * (m_rr^2 + m_tt^2 + m_pp^2 + 2 * (m_rt^2 + m_rp^2 + m_tp^2)))`` in N.m.

    A float for one tensor ``(6,)``, an array ``(n,)`` for a cloud ``(n, 6)``.
    """
    mt6 = np.asarray(mt6, dtype=float)
    diagonal, off_diagonal = np.sum(mt6[..., :3] ** 2, axis=-1), np.sum(mt6[..., 3:] ** 2, axis=-1)
    m0 = np.sqrt(0.5 * (diagonal + 2.0 * off_diagonal))
    return float(m0) if mt6.ndim == 1 else m0


def moment_magnitude(mt6):
    """``Mw = (2/3) (log10 M0 - 9.1)`` with ``M0 = scalar_moment(mt6)`` in N.m (IASPEI, 2013).

    A float for one tensor ``(6,)``, an array ``(n,)`` for a cloud ``(n, 6)``.
    """
    return moment_magnitude_from_scalar_moment(scalar_moment(mt6))


def moment_magnitude_from_scalar_moment(m0_nm):
    """``Mw = (2/3) (log10 M0 - 9.1)`` of a scalar moment ``m0_nm`` in N.m (IASPEI, 2013)."""
    return (2.0 / 3.0) * (np.log10(m0_nm) - 9.1)


def magnitude_to_m0(mw):
    """Scalar moment ``M0 = 10**(1.5*Mw + 9.1)`` in N.m of a moment magnitude ``mw``, the inverse of
    :func:`moment_magnitude_from_scalar_moment`; a scalar or an array.
    """
    return 10.0 ** (1.5 * np.asarray(mw, dtype=float) + 9.1)
