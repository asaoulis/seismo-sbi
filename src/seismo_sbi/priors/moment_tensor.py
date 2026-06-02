"""Uniform random moment tensors on the sphere of fixed scalar moment.

The orientation prior is the simple Gaussian-normalise method: draw six i.i.d.
standard normals, normalise to the unit sphere, then rescale so that the scalar
moment of the result equals the requested ``m0``.

This is self-consistent with the library's scalar-moment convention
``M0 = sqrt(0.5 * ||mt6||^2)`` (see ``instaseis_simulator.wrapper._scalar_moment``):
since ``M0 = ||mt6|| / sqrt(2)``, setting ``mt6 = sqrt(2) * m0 * g/||g||`` gives
``scalar_moment(mt6) == m0`` exactly, with the orientation distributed uniformly on
the sphere in that same metric.  It is the conceptually simpler equivalent of the
trigonometric Stahler-Sigloch (2014) / Tashiro parametrisation.
"""
from __future__ import annotations

import numpy as np

#: Component order of the 6-vector (RTP / Aki-Richards), N.m.
MT_COMPONENT_ORDER = ("m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp")

_SQRT2 = np.sqrt(2.0)


def scalar_moment(mt6) -> float:
    """Scalar moment ``M0 = sqrt(0.5 * dot(mt6, mt6))`` (N.m).

    Mirrors ``instaseis_simulator.wrapper._scalar_moment`` but is kept here so the
    ``priors`` package does not depend on instaseis (imported by ``wrapper``).
    """
    mt6 = np.asarray(mt6, dtype=float)
    return float(np.sqrt(0.5 * np.dot(mt6, mt6)))


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
