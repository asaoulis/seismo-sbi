"""Standard sources on the Tape and Tape (2012) lune, and the scalar moment against pyrocko's."""
import numpy as np
import pytest

from seismo_sbi.moment_tensor.comparison import pyrocko_mt
from seismo_sbi.moment_tensor.conventions import scalar_moment
from seismo_sbi.moment_tensor.lune_angles import mts6_to_gamma_delta

#: ``m6`` of each standard source and its (gamma, delta) in degrees; None where gamma is undefined.
STANDARD_SOURCES = {
    "double couple": ([0, 1, -1, 0, 0, 0], (0.0, 0.0)),
    "explosion": ([1, 1, 1, 0, 0, 0], (None, 90.0)),
    "implosion": ([-1, -1, -1, 0, 0, 0], (None, -90.0)),
    "CLVD (2, -1, -1)": ([2, -1, -1, 0, 0, 0], (-30.0, 0.0)),
    "CLVD (-2, 1, 1)": ([-2, 1, 1, 0, 0, 0], (30.0, 0.0)),
    "tensile crack (3, 1, 1)": ([3, 1, 1, 0, 0, 0], (-30.0, 60.5)),
}


@pytest.mark.parametrize("name", list(STANDARD_SOURCES))
def test_standard_sources_sit_where_tape_and_tape_put_them(name):
    m6, (expected_gamma_deg, expected_delta_deg) = STANDARD_SOURCES[name]
    gamma_deg, delta_deg = (float(angle[0]) for angle in mts6_to_gamma_delta(np.array(m6, dtype=float)[None]))
    assert delta_deg == pytest.approx(expected_delta_deg, abs=0.1)
    if expected_gamma_deg is not None:
        assert gamma_deg == pytest.approx(expected_gamma_deg, abs=0.1)


@pytest.mark.parametrize("m6", [[0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 0, 0],
                                [3e17, -1e17, 2e16, 4e16, -5e16, 6e16]])
def test_the_scalar_moment_is_pyrockos(m6):
    assert scalar_moment(m6) == pytest.approx(pyrocko_mt(m6).scalar_moment(), rel=1e-12)
