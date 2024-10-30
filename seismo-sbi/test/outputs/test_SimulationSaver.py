import os
import sys
import unittest

from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from src.axisem_simulator.wrapper.outputs.simulation_saver import SimulationSaver
from src.axisem_simulator.wrapper.input_settings.types import InputTypes

class Testing_SimulationSaver(unittest.TestCase):

    def setUp(self):

        self.path = Path(__file__).parent / "../stubs/fake_data/sim.h5"

    def test_save_simple_inputs(self):

        inputs = {InputTypes.MESHER: {"mesh_param_1": 1, "mesh_param_2": "abc"},
                 InputTypes.BASIC_SOLVER: {"solver_param": 2, "solver_param_2": True},}

        outputs = {}

        output_collector = SimulationSaver(inputs, outputs)
        output_collector.dump_data_as_hdf5(str(self.path.resolve()))

        saved_sim = h5py.File(self.path)
        mesher_input_attrs = dict(saved_sim["inputs"]["MESHER"].attrs)
        solver_input_attrs = dict(saved_sim["inputs"]["BASIC_SOLVER"].attrs)

        self.assertEqual(inputs[InputTypes.MESHER], mesher_input_attrs)
        self.assertEqual(inputs[InputTypes.BASIC_SOLVER], solver_input_attrs)

    def test_save_seismogram_outputs(self):

        inputs = {}

        outputs = {"seismogram_arrays": {"Z":np.ones((50,600)), "E":np.zeros((50,600))}, 
                    "times": np.ones((600))}

        output_collector = SimulationSaver(inputs, outputs)
        output_collector.dump_data_as_hdf5(str(self.path.resolve()))

        saved_sim = h5py.File(self.path)
        Z_seismograms = np.array(saved_sim["outputs"]["seismogram_arrays"]["Z"])
        E_seismograms = np.array(saved_sim["outputs"]["seismogram_arrays"]["E"])
        times = np.array(saved_sim["outputs"]["times"])

        self.assertTrue(np.allclose(outputs["seismogram_arrays"]["Z"], Z_seismograms))
        self.assertTrue(np.allclose(outputs["seismogram_arrays"]["E"], E_seismograms))

        self.assertTrue(np.allclose(outputs["times"], times))

    def tearDown(self):

        self.path.unlink()
        self.path.parent.rmdir()

if __name__ == '__main__':
    unittest.main()