import os
import sys
import unittest

from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from src.axisem_simulator.wrapper.outputs.output_collector import OutputCollector
from src.axisem_simulator.wrapper.outputs.output_modifiers import shift_seismograms

class Testing_OutputCollector(unittest.TestCase):

    def test_load_all_seismograms_Z(self):

        station_order = ["UP01", "UP02", "UP10"]
        output_collector = OutputCollector(station_order, components=["Z"])

        seismograms_path = Path(__file__).parent / "../stubs/seismogram_data"
        seismograms, times = output_collector._load_all_seismograms(str(seismograms_path.resolve()))
        exp_times = np.arange(10)
        exp_Z_seismograms = np.array([[0, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.1],
                                      [0, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.2],
                                      [0, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 1  ]])

        self.assertTrue(np.allclose(seismograms['Z'], exp_Z_seismograms))
        self.assertTrue(np.allclose(times, exp_times))
        self.assertEqual(len(seismograms), 1)

    def test_load_all_seismograms_multicomponent(self):

        station_order = ["UP01", "UP02", "UP10"]
        output_collector = OutputCollector(station_order)

        seismograms_path = Path(__file__).parent / "../stubs/seismogram_data"
        seismograms, _ = output_collector._load_all_seismograms(str(seismograms_path.resolve()))

        exp_Z_seismograms = np.array([[0, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.1],
                                      [0, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.2],
                                      [0, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 1  ]])
        
        exp_E_seismograms = np.array([[0.1, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.1],
                                      [0.1, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.2],
                                      [0.1, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 1  ]])

        exp_N_seismograms = np.array([[0.01, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.1],
                                      [0.01, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.2],
                                      [0.01, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 1  ]])

        pairs = [('Z', exp_Z_seismograms), ('E', exp_E_seismograms), (('N', exp_N_seismograms))]
        for comp, exp_seismograms in pairs:
            self.assertTrue(np.allclose(seismograms[comp], exp_seismograms))


    def test_load_all_seismograms_multicomponent_order(self):

        station_order = ["UP02", "UP01"]
        output_collector = OutputCollector(station_order, components=['Z', 'N'])

        seismograms_path = Path(__file__).parent / "../stubs/seismogram_data"
        seismograms, _ = output_collector._load_all_seismograms(str(seismograms_path.resolve()))

        exp_Z_seismograms = np.array([[0, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.2],
                                      [0, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.1]])

        exp_N_seismograms = np.array([[0.01, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.2],
                                      [0.01, 0.1, 0.2, 0.1, 0, 0.001, -0.1, -0.2, -0.1, 0.1]])

        pairs = [('Z', exp_Z_seismograms), (('N', exp_N_seismograms))]
        for comp, exp_seismograms in pairs:
            self.assertTrue(np.allclose(seismograms[comp], exp_seismograms))

    def test_load_all_seismograms_processing_func(self):

        station_order = ["UP02", "UP01"]
        seismogram_processing_func = lambda x: x + np.ones_like(x)
        output_collector = OutputCollector(station_order, components=['Z', 'N'],
                             seismogram_processing_function=seismogram_processing_func)

        seismograms_path = Path(__file__).parent / "../stubs/seismogram_data"
        seismograms, _ = output_collector._load_all_seismograms(str(seismograms_path.resolve()))

        exp_Z_seismograms = np.array([[1, 1.1, 1.2, 1.1, 1, 1.001, 0.9, 0.8, 0.9, 1.2],
                                      [1, 1.1, 1.2, 1.1, 1, 1.001, 0.9, 0.8, 0.9, 1.1]])

        exp_N_seismograms = np.array([[1.01, 1.1, 1.2, 1.1, 1, 1.001, 0.9, 0.8, 0.9, 1.2],
                                      [1.01, 1.1, 1.2, 1.1, 1, 1.001, 0.9, 0.8, 0.9, 1.1]])

        pairs = [('Z', exp_Z_seismograms), (('N', exp_N_seismograms))]
        for comp, exp_seismograms in pairs:
            self.assertTrue(np.allclose(seismograms[comp], exp_seismograms))

    def test_shift_seismograms(self):

        station_order = ["UP02", "UP01"]
        exp_times = np.arange(10)
        seismogram_processing_func = lambda x: shift_seismograms(x, exp_times, 2, 0) + 1
        output_collector = OutputCollector(station_order, components=['Z', 'N'],
                             seismogram_processing_function=seismogram_processing_func)

        seismograms_path = Path(__file__).parent / "../stubs/seismogram_data"
        seismograms, _ = output_collector._load_all_seismograms(str(seismograms_path.resolve()))

        exp_Z_seismograms = np.array([[1.2, 1.1, 1, 1.001, 0.9, 0.8, 0.9, 1.2, 1, 1],
                                      [1.2, 1.1, 1, 1.001, 0.9, 0.8, 0.9, 1.1, 1, 1]])

        exp_N_seismograms = np.array([[1.2, 1.1, 1, 1.001, 0.9, 0.8, 0.9, 1.2, 1, 1],
                                      [1.2, 1.1, 1, 1.001, 0.9, 0.8, 0.9, 1.1, 1, 1]])

        pairs = [('Z', exp_Z_seismograms), (('N', exp_N_seismograms))]
        for comp, exp_seismograms in pairs:
            self.assertTrue(np.allclose(seismograms[comp], exp_seismograms))


if __name__ == '__main__':
    unittest.main()