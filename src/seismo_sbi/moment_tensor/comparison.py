"""Moment-tensor comparison: the up-south-east pyrocko tensor, Kagan angles and principal axes.

``m6`` is the pipeline vector ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in the spherical
up-south-east convention the simulator feeds instaseis and the published catalogues report.
``pyrocko_mt`` builds a pyrocko ``MomentTensor`` directly in ``m_up_south_east`` with no sign
change, so ``M = [[m_rr, m_rt, m_rp], [m_rt, m_tt, m_tp], [m_rp, m_tp, m_pp]]``; negating
``m_rp``/``m_tp`` would be a reflection that leaves Kagan angles, scalar moment and source type
untouched while mirroring the absolute orientation.
"""
from __future__ import annotations

import numpy as np


def pyrocko_mt(m6):
    """pyrocko ``MomentTensor`` from ``m6 = [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]`` (GCMT
    up-south-east), built directly in ``m_up_south_east`` with no sign flip."""
    from pyrocko import moment_tensor as pmt
    M = np.array([[m6[0], m6[3], m6[4]],
                  [m6[3], m6[1], m6[5]],
                  [m6[4], m6[5], m6[2]]])
    return pmt.MomentTensor(m_up_south_east=M)


def kagan(m6_a, m6_b):
    """Kagan angle (degrees) between two moment tensors ``m6_a``, ``m6_b``.

    Symmetric; ``kagan(m, m) == 0``.  Returns ``nan`` if pyrocko's
    ``kagan_angle`` raises (e.g. a degenerate tensor)."""
    from pyrocko import moment_tensor as pmt
    try:
        return float(pmt.kagan_angle(pyrocko_mt(m6_a), pyrocko_mt(m6_b)))
    except Exception:  # noqa: BLE001
        return float("nan")


# --- Batched eigen-frame primitives: pyrocko's algebra in numpy, ~4e4x faster than kagan() ---

# USE -> NED (north = -south, east = east, down = -up); pyrocko stores m() in NED, so the
# results are identical to pyrocko's rather than merely equivalent.
_USE_TO_NED = np.array([[0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0]])

# eigh returns columns (p, b, t) in ascending eigenvalue order; pyrocko's frame is rows (t, p, b).
_PBT_TO_TPB = np.array([[0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]])


def m6_to_matrix_ned(m6):
    """Batched ``(n, 3, 3)`` NED moment-tensor matrices from ``(n, 6)`` USE m6."""
    m6 = np.asarray(m6, dtype=float).reshape(-1, 6)
    M = np.empty((m6.shape[0], 3, 3), dtype=float)
    M[:, 0, 0] = m6[:, 0]
    M[:, 1, 1] = m6[:, 1]
    M[:, 2, 2] = m6[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = m6[:, 3]
    M[:, 0, 2] = M[:, 2, 0] = m6[:, 4]
    M[:, 1, 2] = M[:, 2, 1] = m6[:, 5]
    return _USE_TO_NED @ M @ _USE_TO_NED.T


def _eigen_frames(m6):
    """Batched rotation frames ``(n, 3, 3)`` (rows = T, P, B axes in NED).

    Right-handed: the eigenvector basis is sign-flipped where ``det < 0`` so
    every frame is a proper rotation (a reflection would break the quaternion
    step in :func:`kagan_batch`).
    """
    _, evecs = np.linalg.eigh(m6_to_matrix_ned(m6))     # columns ascending: p, b, t
    det = np.linalg.det(evecs)
    evecs = np.where(det[:, None, None] < 0.0, -evecs, evecs)
    return _PBT_TO_TPB @ np.swapaxes(evecs, -1, -2)


def kagan_batch(m6_a, m6_b):
    """Kagan angles (degrees) between batches of moment tensors — vectorised.

    ``m6_a``/``m6_b`` are ``(n, 6)`` or a single ``(6,)``/``(1, 6)`` tensor,
    which broadcasts against the other.  Returns a ``(n,)`` array.

    Numerically identical to :func:`kagan` (max ``|Δ|`` ~1e-12 deg over random and
    real posterior tensors — locked by ``tests/unit/test_moment_tensor_comparison.py``)
    but ~4e4x faster, which is what makes posterior-wide orientation statistics
    affordable.  The rotation between the two eigen-frames is converted to a
    quaternion and the *largest* component taken, which is exactly pyrocko's
    minimum-over-the-four-symmetry-operations answer without branching.
    """
    A = _eigen_frames(m6_a)
    B = _eigen_frames(m6_b)
    if B.shape[0] == 1 and A.shape[0] != 1:
        B = np.broadcast_to(B, A.shape)
    elif A.shape[0] == 1 and B.shape[0] != 1:
        A = np.broadcast_to(A, B.shape)
    if A.shape != B.shape:
        raise ValueError(f"kagan_batch: incompatible shapes {A.shape} vs {B.shape}")
    u = A @ np.swapaxes(B, -1, -2)                     # relative rotation
    d = np.einsum("nii->ni", u)                        # its diagonal
    tr = d.sum(-1)
    # 4|q_i|^2 for the four quaternion components of the relative rotation.
    tq = np.stack([1.0 + tr,
                   1.0 + 2.0 * d[:, 0] - tr,
                   1.0 + 2.0 * d[:, 1] - tr,
                   1.0 + 2.0 * d[:, 2] - tr], axis=-1)
    qmax = 0.5 * np.sqrt(np.clip(tq.max(-1), 0.0, None))
    return 2.0 * np.degrees(np.arccos(np.clip(qmax, -1.0, 1.0)))


def mt_axes(m6):
    """Batched P/T/N axis azimuth & plunge (degrees, NED lower hemisphere).

    Returns a dict of ``(n,)`` arrays ``p_az, p_plunge, t_az, t_plunge,
    n_az, n_plunge`` — the quantities pyrocko gives per tensor, off one batched
    eigen-decomposition.

    Axes are sign-ambiguous, so each is forced into the lower hemisphere
    (``plunge >= 0``); a *near-horizontal* axis is therefore azimuth-ambiguous
    by 180° and either convention is correct.
    """
    _, evecs = np.linalg.eigh(m6_to_matrix_ned(m6))     # columns ascending: p, b, t
    out = {}
    for key, col in (("p", 0), ("n", 1), ("t", 2)):
        v = evecs[:, :, col]
        v = np.where((v[:, 2] < 0.0)[:, None], -v, v)   # force plunge >= 0
        norm = np.linalg.norm(v, axis=1)
        norm = np.where(norm == 0.0, 1.0, norm)
        out[f"{key}_az"] = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        out[f"{key}_plunge"] = np.degrees(np.arcsin(np.clip(v[:, 2] / norm, -1.0, 1.0)))
    return out
