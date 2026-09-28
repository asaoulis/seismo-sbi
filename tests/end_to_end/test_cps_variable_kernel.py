"""CPS Green's functions computed for the velocity model each simulation is handed."""
import os
from pathlib import Path

import numpy as np
import pytest

from seismo_sbi.simulators.cps.CPS import perturb_model
from seismo_sbi.simulators.cps.compatibility import load_velocity_model
from seismo_sbi.simulators.cps.simulator import CPSVariableKernelSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers

CPS_PATH = os.environ.get("CPS_PATH", "")
REPO = Path(__file__).resolve().parents[2]
PROCESSING = {"filter": {"type": "bandpass", "freqmin": 0.02, "freqmax": 0.08, "corners": 4, "zerophase": False},
              "sampling_rate": 1.0, "filter_sampling_rate": 5.0}
SOURCE = {"source_location": [37.636, -118.936, 10.0, 0.0],
          "moment_tensor": [1.0e17, -0.6e17, -0.4e17, 0.3e17, -0.5e17, 0.2e17]}

pytestmark = [pytest.mark.slow, pytest.mark.requires_data,
              pytest.mark.skipif(not Path(CPS_PATH, "hprep96").exists(), reason="CPS_PATH has no hprep96")]


def test_perturbed_model_changes_the_waveforms_and_the_reference_is_reproducible(tmp_path):
    receivers = Receivers(receivers=[Receiver(35.945171, -120.541603, "BK", "PKD", ["Z", "E", "N"])])
    simulator = CPSVariableKernelSimulator(
        components="ZEN", receivers=receivers, seismogram_duration_in_s=200, synthetics_processing=PROCESSING,
        gf_storage_root=str(tmp_path / "cps"), cps_path=CPS_PATH)
    reference_model = load_velocity_model(str(REPO / "examples" / "configs" / "SoCal.plain.txt"))
    np.random.seed(0)
    perturbed_model = perturb_model(reference_model, kappa=5)

    reference = simulator.run_simulation({**SOURCE, "velocity_model": reference_model})[1]["PKD"]["Z"]
    repeat = simulator.run_simulation({**SOURCE, "velocity_model": reference_model})[1]["PKD"]["Z"]
    perturbed = simulator.run_simulation({**SOURCE, "velocity_model": perturbed_model})[1]["PKD"]["Z"]

    np.testing.assert_array_equal(repeat, reference)
    assert np.max(np.abs(perturbed - reference)) > 0.05 * np.max(np.abs(reference))
