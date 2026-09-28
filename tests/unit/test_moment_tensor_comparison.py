"""The moment-tensor comparison primitives (``seismo_sbi.moment_tensor.comparison``) and
``recovered_mt_samples``: the Kagan angle is reflexive, symmetric, non-trivial for clearly different
mechanisms and ``nan`` for a zero tensor; the pyrocko tensor round-trips; the NED matrix is the
up-south-east matrix rotated.
"""
import numpy as np
import pytest

from seismo_sbi.evaluation.inference import recovered_mt_samples
from seismo_sbi.moment_tensor.comparison import kagan
from seismo_sbi.moment_tensor.comparison import pyrocko_mt

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
    # m6 = [Mrr,Mtt,Mpp,Mrt,Mrp,Mtp] is standard GCMT up-south-east and maps 1:1 to
    # pyrocko's m_up_south_east with NO sign flip (the old spurious Mrp/Mtp negation
    # mirrored the mechanism — see moment_tensor.py history note).
    m6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    M = pyrocko_mt(m6).m_up_south_east()
    assert M[0, 0] == pytest.approx(1.0)   # Mrr
    assert M[1, 1] == pytest.approx(2.0)   # Mtt
    assert M[2, 2] == pytest.approx(3.0)   # Mpp
    assert M[0, 1] == pytest.approx(4.0)   # Mrt
    assert M[0, 2] == pytest.approx(5.0)   # Mrp (no flip — GCMT USE direct)
    assert M[1, 2] == pytest.approx(6.0)   # Mtp


def test_pyrocko_mt_roundtrips_known_mechanism():
    # The decisive proof the no-flip convention is correct: a known mechanism's TRUE
    # USE 6-vector must recover its own strike/dip/rake.
    from pyrocko import moment_tensor as pmt
    for s, d, r in [(0.0, 90.0, 0.0), (30.0, 60.0, -90.0), (115.0, 50.0, -70.0)]:
        M = pmt.MomentTensor(strike=s, dip=d, rake=r).m_up_south_east()
        m6 = np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])
        planes = pyrocko_mt(m6).both_strike_dip_rake()
        ok = any(np.allclose([round(a) % 360, round(b), (round(c) + 180) % 360 - 180],
                             [s % 360, d, r], atol=2) for a, b, c in planes)
        assert ok, f"sdr {(s, d, r)} not recovered; got {planes}"


def test_recovered_mt_samples_slices_first_six():
    class _Dummy:
        samples = np.arange(8 * 9, dtype=float).reshape(8, 9)
    out = recovered_mt_samples(_Dummy())
    assert out.shape == (8, 6)
    assert np.array_equal(out, _Dummy.samples[:, :6])


# ---------------------------------------------------------------------------
# Batched eigen-frame primitives: kagan_batch and mt_axes.
#
# These exist purely as a fast path (pyrocko's per-call MomentTensor
# construction costs ~2.2 ms, which makes posterior-wide orientation statistics
# an 8-minute job).  They are therefore CHARACTERISATION tests against the
# pyrocko implementations they replace, not independent re-derivations.
# ---------------------------------------------------------------------------

def _random_m6(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 6)) * 1e16


def test_kagan_batch_matches_pyrocko_kagan():
    from seismo_sbi.moment_tensor.comparison import kagan_batch
    A, B = _random_m6(200, seed=1), _random_m6(200, seed=2)
    ref = np.array([kagan(a, b) for a, b in zip(A, B)])
    got = kagan_batch(A, B)
    assert np.nanmax(np.abs(ref - got)) < 1e-6


def test_kagan_batch_self_zero_symmetric_and_scale_invariant():
    # Tolerance note: the angle comes out of arccos(qmax) with qmax -> 1 for
    # identical tensors, where arccos has infinite derivative, so a 1e-16
    # rounding error in qmax lifts the angle to ~2e-6 deg.  That floor is
    # inherent to the quaternion form (pyrocko's kagan_angle shares it) and is
    # ~7 orders below any orientation difference of interest.
    from seismo_sbi.moment_tensor.comparison import kagan_batch
    A, B = _random_m6(50, seed=3), _random_m6(50, seed=4)
    assert np.allclose(kagan_batch(A, A), 0.0, atol=1e-4)
    assert np.allclose(kagan_batch(A, B), kagan_batch(B, A), atol=1e-6)
    assert np.allclose(kagan_batch(A, 7.3 * A), 0.0, atol=1e-4)
    assert np.all(kagan_batch(A, B) <= 120.0 + 1e-6)


def test_kagan_batch_broadcasts_single_tensor_either_side():
    from seismo_sbi.moment_tensor.comparison import kagan_batch
    A, one = _random_m6(20, seed=5), _random_m6(1, seed=6)
    assert kagan_batch(A, one).shape == (20,)
    assert kagan_batch(one, A).shape == (20,)
    assert np.allclose(kagan_batch(A, one), kagan_batch(one, A), atol=1e-9)
    # a bare (6,) tensor is accepted on either side too
    assert np.allclose(kagan_batch(A, one.reshape(6)), kagan_batch(A, one), atol=1e-12)


def test_kagan_batch_rejects_mismatched_batches():
    from seismo_sbi.moment_tensor.comparison import kagan_batch
    with pytest.raises(ValueError):
        kagan_batch(_random_m6(5, seed=7), _random_m6(3, seed=8))


def test_mt_axes_matches_mt_features_axes():
    """P/T/N axes must agree with the per-tensor pyrocko readout.

    Axes are sign-ambiguous, so the comparison is on the axis DIRECTION up to
    sign; azimuth itself is only compared where the axis is not near-horizontal
    (a horizontal axis is azimuth-ambiguous by exactly 180°, and either
    convention is correct).
    """
    from seismo_sbi.moment_tensor.comparison import mt_axes
    from pyrocko import moment_tensor as pmt

    M6 = _random_m6(100, seed=9)
    got = mt_axes(M6)
    for i, m6 in enumerate(M6):
        mt = pyrocko_mt(m6)
        for key, vec in (("p", mt.p_axis()), ("t", mt.t_axis()), ("n", mt.null_axis())):
            v = np.asarray(vec, float).ravel()
            v = v / np.linalg.norm(v)
            az, pl = np.radians(got[f"{key}_az"][i]), np.radians(got[f"{key}_plunge"][i])
            w = np.array([np.cos(pl) * np.cos(az), np.cos(pl) * np.sin(az), np.sin(pl)])
            assert abs(float(np.dot(v, w))) > 1.0 - 1e-9, f"{key} axis differs on tensor {i}"
        assert np.all(got[f"{key}_plunge"] >= -1e-9)      # lower hemisphere
    assert np.all((got["p_az"] >= 0.0) & (got["p_az"] < 360.0))
    _ = pmt  # (imported to assert pyrocko availability for the readout above)


def test_mt_axes_pure_strike_slip_axes_are_horizontal():
    # A vertical strike-slip fault has horizontal P and T axes and a vertical null axis.
    from seismo_sbi.moment_tensor.comparison import mt_axes
    from pyrocko import moment_tensor as pmt
    M = pmt.MomentTensor(strike=0.0, dip=90.0, rake=0.0).m_up_south_east()
    m6 = np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])
    ax = mt_axes(m6)
    assert abs(ax["p_plunge"][0]) < 1.0
    assert abs(ax["t_plunge"][0]) < 1.0
    assert abs(ax["n_plunge"][0] - 90.0) < 1.0


def test_from_pyrocko_inverts_pyrocko_mt():
    from seismo_sbi.moment_tensor.comparison import from_pyrocko, pyrocko_mt

    m6 = np.random.default_rng(3).normal(size=6) * 1e16

    np.testing.assert_allclose(from_pyrocko(pyrocko_mt(m6)), m6, rtol=1e-9)


def test_the_ned_matrix_is_the_use_matrix_rotated():
    from seismo_sbi.moment_tensor.comparison import _USE_TO_NED, m6_to_matrix_ned
    from seismo_sbi.moment_tensor.lune_angles import m6_to_matrix

    m6 = np.random.default_rng(4).normal(size=(5, 6)) * 1e16

    np.testing.assert_allclose(m6_to_matrix_ned(m6), _USE_TO_NED @ m6_to_matrix(m6) @ _USE_TO_NED.T, rtol=1e-12)
