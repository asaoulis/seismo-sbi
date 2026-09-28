"""Moment magnitude, CLVD ratio, pyrocko moment tensor and nodal planes of a moment tensor.

``mt`` is ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m (up, south, east).
:func:`convert_to_pyrocko` builds pyrocko's ``MomentTensor`` from the north-east-down components
(north = -south, down = -up), the same tensor :func:`~seismo_sbi.moment_tensor.comparison.pyrocko_mt`
builds from the up-south-east ones.
"""
import numpy as np
from pyrocko import moment_tensor as pmt

from seismo_sbi.moment_tensor.conventions import compute_scalar_moment


def get_MW_and_epsilon(moment_tensor_sol):

    moment_tensor_matrix, M_0 = compute_scalar_moment(moment_tensor_sol)

    MW = (np.log10(M_0) - 9.1)/1.5

    M_isotropic = 1/3 * np.trace(moment_tensor_matrix) * np.eye(3)
    M_deviatoric = moment_tensor_matrix - M_isotropic

    eigenvalues = list(sorted(np.linalg.eigvals(M_deviatoric), reverse=True))
    epsilon = eigenvalues[1]/ max(abs(eigenvalues[0]), abs(eigenvalues[2]))
    
    return (MW, epsilon)

def convert_to_pyrocko(mt):
    #up, south, east to north east down
    m = pmt.MomentTensor(
        mnn=mt[1],
        mee=mt[2],
        mdd=mt[0],
        mne=-mt[5],
        mnd=mt[3],
        med=-mt[4]
    )
    return m

def get_nodal_planes(theta):
    m = convert_to_pyrocko(theta)
    nodal_planes = m.both_strike_dip_rake()
    return nodal_planes
