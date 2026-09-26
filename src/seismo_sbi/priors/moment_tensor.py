"""Uniform random moment tensors at a fixed scalar moment.

Six i.i.d. standard normals are normalised onto the unit sphere and rescaled so the scalar
moment equals the requested ``m0``. This is consistent with the library's convention
``M0 = sqrt(0.5 * ||mt6||^2)``: since ``M0 = ||mt6|| / sqrt(2)``, setting
``mt6 = sqrt(2) * m0 * g / ||g||`` gives exactly that moment with the orientation uniform on the
sphere in the same metric. It is the simpler equivalent of the trigonometric parametrisation.
"""
from __future__ import annotations

import numpy as np

#: Component order of the 6-vector (RTP / Aki-Richards), N.m.
MT_COMPONENT_ORDER = ("m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp")

_SQRT2 = np.sqrt(2.0)


def uniform_moment_tensor_on_sphere(m0: float, rng: np.random.Generator) -> np.ndarray:
    """One moment tensor of scalar moment ``m0`` oriented uniformly on the sphere.

    Returns the 6-vector ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` (N.m) with
    ``scalar_moment(result) == m0`` to floating-point precision.
    """
    g = rng.standard_normal(6)
    norm = np.linalg.norm(g)
    while norm == 0.0:  # astronomically unlikely; guard against divide-by-zero
        g = rng.standard_normal(6)
        norm = np.linalg.norm(g)
    return _SQRT2 * m0 * g / norm
