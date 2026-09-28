"""Uniform random moment tensors at a fixed scalar moment.

The diagonal components are drawn from N(0, 1) and the off-diagonal ones from N(0, 1/2), the
rotation-invariant Gaussian on symmetric 3x3 tensors, then rescaled so the full-tensor scalar
moment (:func:`~seismo_sbi.moment_tensor.conventions.scalar_moment`) equals the requested ``m0``.
The result is uniform on the sphere of moment tensors of that moment (Tape and Tape, 2015).
"""
from __future__ import annotations

import numpy as np

from seismo_sbi.moment_tensor.conventions import scalar_moment

#: Component order of the 6-vector (RTP / Aki-Richards), N.m.
MT_COMPONENT_ORDER = ("m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp")

#: Standard deviation of each component of the rotation-invariant Gaussian tensor.
_COMPONENT_STD = np.array([1.0, 1.0, 1.0, np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)])


def uniform_moment_tensor_on_sphere(m0: float, rng: np.random.Generator) -> np.ndarray:
    """One moment tensor of scalar moment ``m0`` oriented uniformly on the sphere.

    Returns the 6-vector ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` (N.m) with
    ``scalar_moment(result) == m0`` to floating-point precision.
    """
    g = _COMPONENT_STD * rng.standard_normal(6)
    moment = scalar_moment(g)
    while moment == 0.0:  # astronomically unlikely; guard against divide-by-zero
        g = _COMPONENT_STD * rng.standard_normal(6)
        moment = scalar_moment(g)
    return m0 * g / moment
