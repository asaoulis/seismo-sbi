"""Moment-tensor component conventions, the 3x3 matrix and the scalar moment.

``m6`` is ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m (up, south, east). ``create_matrix``
builds the symmetric 3x3 tensor from six components in that index order, the matrix pyrocko's
``m_up_south_east`` takes. ``scalar_moment`` is the full-tensor moment of Silver and Jordan
(1982), ``M0 = sqrt(0.5 * sum_ij M_ij^2)``, in which each off-diagonal component counts twice, and
``moment_magnitude`` is the IASPEI (2013) ``Mw = (2/3) (log10 M0 - 9.1)`` with ``M0`` in N.m.
"""
import numpy as np


def create_matrix(moment_tensor_sol):
    moment_tensor_matrix = np.array([[moment_tensor_sol[0], moment_tensor_sol[3], moment_tensor_sol[4]],
                                        [moment_tensor_sol[3], moment_tensor_sol[1], moment_tensor_sol[5]],
                                        [moment_tensor_sol[4], moment_tensor_sol[5], moment_tensor_sol[2]]])

    return moment_tensor_matrix


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
    return (2.0 / 3.0) * (np.log10(scalar_moment(mt6)) - 9.1)
