"""Moment magnitude, CLVD ratio and nodal planes of a moment tensor.

``mt`` is ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m (up, south, east). The nodal planes are
those of :func:`~seismo_sbi.moment_tensor.comparison.pyrocko_mt`, in degrees.
"""
import numpy as np

from seismo_sbi.moment_tensor.comparison import pyrocko_mt
from seismo_sbi.moment_tensor.conventions import create_matrix, moment_magnitude


def get_MW_and_epsilon(moment_tensor_sol):

    moment_tensor_matrix = create_matrix(moment_tensor_sol)
    MW = moment_magnitude(moment_tensor_sol)

    M_isotropic = 1/3 * np.trace(moment_tensor_matrix) * np.eye(3)
    M_deviatoric = moment_tensor_matrix - M_isotropic

    eigenvalues = list(sorted(np.linalg.eigvals(M_deviatoric), reverse=True))
    epsilon = eigenvalues[1]/ max(abs(eigenvalues[0]), abs(eigenvalues[2]))
    
    return (MW, epsilon)

def get_nodal_planes(theta):
    m = pyrocko_mt(theta)
    nodal_planes = m.both_strike_dip_rake()
    return nodal_planes
