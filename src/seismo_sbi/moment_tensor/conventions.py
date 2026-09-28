"""Moment-tensor component conventions, the 3x3 matrix and the scalar moment.

``m6`` is ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m (up, south, east). ``create_matrix``
builds the symmetric 3x3 tensor from six components in that index order, the matrix pyrocko's
``m_up_south_east`` takes. ``scalar_moment`` is the full-tensor moment of Silver and Jordan
(1982), ``M0 = sqrt(0.5 * sum_ij M_ij^2)``, in which each off-diagonal component counts twice.
"""
import numpy as np


def create_matrix(moment_tensor_sol):
    moment_tensor_matrix = np.array([[moment_tensor_sol[0], moment_tensor_sol[3], moment_tensor_sol[4]],
                                        [moment_tensor_sol[3], moment_tensor_sol[1], moment_tensor_sol[5]],
                                        [moment_tensor_sol[4], moment_tensor_sol[5], moment_tensor_sol[2]]])

    return moment_tensor_matrix


def scalar_moment(mt6) -> float:
    """``M0 = sqrt(0.5 * (m_rr^2 + m_tt^2 + m_pp^2 + 2 * (m_rt^2 + m_rp^2 + m_tp^2)))`` in N.m."""
    mt6 = np.asarray(mt6, dtype=float)
    return float(np.sqrt(0.5 * (np.dot(mt6[:3], mt6[:3]) + 2.0 * np.dot(mt6[3:], mt6[3:]))))
