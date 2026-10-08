"""Smooth perturbations of a layered velocity model draw from numpy's global random state."""
from pathlib import Path

import numpy as np

from seismo_sbi.simulators.cps.compatibility import load_velocity_model
from seismo_sbi.simulators.cps.smooth_perturbations import perturb_cps_model

PREM_LIKE_MODEL = Path(__file__).resolve().parents[1] / "fixtures" / "prem_like_cps_model.txt"


def test_a_seeded_smooth_perturbation_is_reproducible():
    model = load_velocity_model(str(PREM_LIKE_MODEL))
    np.random.seed(1)
    first = perturb_cps_model(model)
    np.random.seed(1)
    again = perturb_cps_model(model)
    np.random.seed(2)
    other = perturb_cps_model(model)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
