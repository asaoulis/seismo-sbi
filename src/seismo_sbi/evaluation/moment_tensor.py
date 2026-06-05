"""
moment_tensor.py
================
Domain-agnostic moment-tensor comparison primitives for the evaluation harness.

Lifted verbatim (behaviour-preserving) from the gitignored
``scripts/santorini_pathbreaker/compare_to_reference.py`` so that the public,
tracked evaluation CLI (``scripts/evaluate_model.py``) and any domain adapter can
import them from ``src/`` instead of from a gitignored script.

The two functions encode the project's *verified* moment-tensor convention:

  * ``m6`` is the raw pipeline vector ``[Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]``.
  * ``pyrocko_mt`` builds a pyrocko ``MomentTensor`` in ``m_up_south_east`` after
    the ``convert_mt_convention`` handedness fix (negate ``Mrp``, ``Mtp`` — the
    ``diag(1, 1, -1)`` correction), so the Kagan angle is computed in the same
    frame the rest of the library uses.

``pyrocko`` is imported lazily inside each function so importing this module
during the fast unit-test gate stays cheap (no pyrocko import at module load).
"""
from __future__ import annotations

import numpy as np


def pyrocko_mt(m6):
    """pyrocko ``MomentTensor`` in the project's verified convention.

    ``m6 = [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]`` (raw pipeline order).  The
    ``[m0, m1, m2, m3, -m4, -m5]`` re-sign is the ``convert_mt_convention``
    ``diag(1, 1, -1)`` handedness fix to land in ``m_up_south_east``.
    """
    from pyrocko import moment_tensor as pmt
    c = [m6[0], m6[1], m6[2], m6[3], -m6[4], -m6[5]]  # convert_mt_convention
    M = np.array([[c[0], c[3], c[4]], [c[3], c[1], c[5]], [c[4], c[5], c[2]]])
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
