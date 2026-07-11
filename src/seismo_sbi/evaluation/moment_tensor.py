"""
moment_tensor.py
================
Domain-agnostic moment-tensor comparison primitives for the evaluation harness.

Lifted verbatim (behaviour-preserving) from the gitignored
``scripts/santorini_pathbreaker/compare_to_reference.py`` so that the public,
tracked evaluation CLI (``scripts/evaluate_model.py``) and any domain adapter can
import them from ``src/`` instead of from a gitignored script.

The two functions encode the project's moment-tensor convention:

  * ``m6`` is the raw pipeline vector ``[Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]`` in the
    standard Harvard/GCMT spherical (r=up, t=south, p=east) convention — exactly
    what the simulator feeds instaseis (``wrapper.py`` maps ``m_rr=m6[0] … m_tp=m6[5]``
    1:1, no sign change) and what the published reference catalogues report.
  * ``pyrocko_mt`` builds the pyrocko ``MomentTensor`` DIRECTLY in
    ``m_up_south_east`` (pyrocko's up-south-east == GCMT USE), with NO sign flip:
    ``M = [[Mrr, Mrt, Mrp], [Mrt, Mtt, Mtp], [Mrp, Mtp, Mpp]]``.

    HISTORY: this used to negate ``Mrp``/``Mtp`` (a spurious ``diag(1, 1, -1)``
    "handedness fix"). That is a REFLECTION about the E–W vertical plane: it leaves
    Kagan angles (relative, reflection-invariant), scalar moment and source type
    (eigenvalue-based) unchanged — which is why it went unnoticed — but it MIRRORS
    the absolute orientation, rotating beachballs / P–T axes / strike-dip-rake (a
    visible ~90° rotation for strike-slip, ``strike -> 180-strike`` for dip-slip).
    Removed after a round-trip proof (a known mechanism's true USE 6-vector recovers
    its own strike/dip/rake only WITHOUT the flip) + the Santorini–Amorgos geology
    (no-flip gives the correct NW–SE extension / NE–SW normal faults).

``pyrocko`` is imported lazily inside each function so importing this module
during the fast unit-test gate stays cheap (no pyrocko import at module load).
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
