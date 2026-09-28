"""Moment-tensor component conventions, the 3x3 matrix and the scalar moment.

``m6`` is ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m (up, south, east). ``create_matrix``
builds the symmetric 3x3 tensor from six components in that index order, the matrix pyrocko's
``m_up_south_east`` takes. The two scalar moments differ: ``compute_scalar_moment`` sums the
full 3x3 tensor, counting each off-diagonal twice; ``scalar_moment`` sums the six components
once each.
"""
import numpy as np


def create_matrix(moment_tensor_sol):
    moment_tensor_matrix = np.array([[moment_tensor_sol[0], moment_tensor_sol[3], moment_tensor_sol[4]],
                                        [moment_tensor_sol[3], moment_tensor_sol[1], moment_tensor_sol[5]],
                                        [moment_tensor_sol[4], moment_tensor_sol[5], moment_tensor_sol[2]]])

    return moment_tensor_matrix


def compute_scalar_moment(moment_tensor_sol):
    """``(3x3 matrix, M0)`` with ``M0 = sqrt(0.5 * sum over the full tensor of m_ij^2)``."""
    moment_tensor_matrix = create_matrix(moment_tensor_sol)

    M_0 = (1/np.sqrt(2)) * np.sum(moment_tensor_matrix**2)**(1/2)
    return moment_tensor_matrix, M_0


def scalar_moment(mt6) -> float:
    """``M0 = sqrt(0.5 * dot(mt6, mt6))`` in N.m, each of the six components counted once."""
    mt6 = np.asarray(mt6, dtype=float)
    return float(np.sqrt(0.5 * np.dot(mt6, mt6)))
