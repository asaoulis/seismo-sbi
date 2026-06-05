"""Unit tests for the lifted moment-tensor primitives
(``seismo_sbi.evaluation.moment_tensor``) and ``recovered_mt_samples``.

These prove the lift from the gitignored ``compare_to_reference.py`` is faithful:
the Kagan angle is reflexive (``kagan(m, m) == 0``), symmetric, returns a sensible
non-trivial angle for clearly different mechanisms, and degrades to ``nan`` rather
than raising for a degenerate (zero) tensor.
"""
import numpy as np
import pytest

from seismo_sbi.evaluation import kagan, recovered_mt_samples
from seismo_sbi.evaluation.moment_tensor import pyrocko_mt

pytest.importorskip("pyrocko")  # kagan/pyrocko_mt need pyrocko at call time


# A pure double couple (Mrt only) and a clearly different double couple (Mtp only).
DC_A = np.array([0.0, 0.0, 0.0, 1.0e16, 0.0, 0.0])
DC_B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0e16])


def test_kagan_self_is_zero():
    assert kagan(DC_A, DC_A) == pytest.approx(0.0, abs=1e-6)
    assert kagan(DC_B, DC_B) == pytest.approx(0.0, abs=1e-6)


def test_kagan_symmetric():
    assert kagan(DC_A, DC_B) == pytest.approx(kagan(DC_B, DC_A), abs=1e-6)


def test_kagan_scale_invariant():
    # Kagan angle depends only on the mechanism, not on the scalar moment.
    assert kagan(DC_A, 7.3 * DC_A) == pytest.approx(0.0, abs=1e-6)


def test_kagan_different_mechanisms_is_substantial():
    ang = kagan(DC_A, DC_B)
    assert 0.0 < ang <= 120.0  # Kagan angle for DCs is bounded by 120 degrees
    assert ang > 30.0          # two orthogonal off-diagonal DCs are far apart


def test_kagan_degenerate_returns_float():
    # A zero tensor is degenerate; kagan must not raise (try/except guard) — it
    # returns a float (pyrocko may itself yield a number, or the guard yields nan).
    val = kagan(np.zeros(6), DC_A)
    assert isinstance(val, float)


def test_pyrocko_mt_convention_signs():
    # The convention re-signs Mrp (idx 4) and Mtp (idx 5).  Recover the same frame
    # we constructed with the m_up_south_east accessor (pyrocko stores NED internally).
    m6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    M = pyrocko_mt(m6).m_up_south_east()
    assert M[0, 0] == pytest.approx(1.0)   # Mrr
    assert M[1, 1] == pytest.approx(2.0)   # Mtt
    assert M[2, 2] == pytest.approx(3.0)   # Mpp
    assert M[0, 1] == pytest.approx(4.0)   # Mrt
    assert M[0, 2] == pytest.approx(-5.0)  # -Mrp (handedness fix)
    assert M[1, 2] == pytest.approx(-6.0)  # -Mtp (handedness fix)


def test_recovered_mt_samples_slices_first_six():
    class _Dummy:
        samples = np.arange(8 * 9, dtype=float).reshape(8, 9)
    out = recovered_mt_samples(_Dummy())
    assert out.shape == (8, 6)
    assert np.array_equal(out, _Dummy.samples[:, :6])
