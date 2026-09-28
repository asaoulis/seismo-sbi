"""The one moment magnitude: IASPEI (2013) Mw = (2/3) (log10 M0 - 9.1), M0 the full-tensor moment in N.m."""
import numpy as np
import pytest

from seismo_sbi.moment_tensor.conventions import moment_magnitude

#: The scalar moment in N.m of an Mw 4.0 event.
MW4_MOMENT_NM = 10.0 ** (1.5 * 4.0 + 9.1)


def test_a_double_couple_of_known_moment_has_its_iaspei_magnitude():
    strike_slip = [0.0, MW4_MOMENT_NM, -MW4_MOMENT_NM, 0.0, 0.0, 0.0]
    assert moment_magnitude(strike_slip) == pytest.approx(4.0, abs=1e-12)
    rotated_strike_slip = [0.0, 0.0, 0.0, 0.0, 0.0, 1e17]
    assert moment_magnitude(rotated_strike_slip) == pytest.approx((2.0 / 3.0) * (17.0 - 9.1), abs=1e-12)


def test_a_cloud_gives_one_magnitude_per_tensor():
    cloud = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, MW4_MOMENT_NM],
                      [0.0, 0.0, 0.0, 10 * MW4_MOMENT_NM, 0.0, 0.0]])
    np.testing.assert_allclose(moment_magnitude(cloud), [4.0, 4.0 + 2.0 / 3.0], atol=1e-12)
