"""Moment magnitude, CLVD ratio and nodal planes of a moment tensor.

``mt`` is ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m (up, south, east). The nodal planes are
those of :func:`~seismo_sbi.moment_tensor.comparison.pyrocko_mt`, in degrees.
:func:`lune_angles_and_magnitude` and :func:`mechanism_parameters` give the source type, Mw and
one nodal plane of a whole cloud of tensors.
"""
import numpy as np

from seismo_sbi.moment_tensor.comparison import pyrocko_mt
from seismo_sbi.moment_tensor.conventions import create_matrix, moment_magnitude
from seismo_sbi.moment_tensor.lune_angles import mts6_to_gamma_delta


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


def lune_angles_and_magnitude(mt6):
    """``(gamma_deg, delta_deg, mw)``, each ``(n,)``, of the tensors ``mt6`` ``(n, 6)`` in N.m."""
    mt6 = np.asarray(mt6, dtype=float)
    gamma_deg, delta_deg = mts6_to_gamma_delta(mt6)
    return gamma_deg, delta_deg, moment_magnitude(mt6)


def mechanism_parameters(mt6, nodal_plane_choice=None):
    """``(n, 6)`` columns ``gamma_deg, delta_deg, Mw, strike_deg, dip_deg, rake_deg`` of the tensors
    ``mt6`` ``(n, 6)`` in N.m.

    The nodal plane is the first of :func:`get_nodal_planes`, or the one ``nodal_plane_choice``
    returns when called with both.
    """
    mt6 = np.asarray(mt6, dtype=float)
    gamma_deg, delta_deg, mw = lune_angles_and_magnitude(mt6)
    out = np.empty((len(mt6), 6), dtype=float)
    for i, mt in enumerate(mt6):
        nodal_plane_pair = get_nodal_planes(mt)
        sdr = nodal_plane_pair[0] if nodal_plane_choice is None else nodal_plane_choice(nodal_plane_pair)
        out[i] = [gamma_deg[i], delta_deg[i], mw[i], float(sdr[0]), float(sdr[1]), float(sdr[2])]
    return out
