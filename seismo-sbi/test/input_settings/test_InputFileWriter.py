import os
import sys
import unittest

from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from src.axisem_simulator.wrapper.input_settings.input_file_writer import InputFileWriter, ConfigurationError, InputSettingsType
from src.axisem_simulator.wrapper.input_settings.presets import MesherSettings, SolverMomentTensorSettings

class Testing_InputFileWriter(unittest.TestCase):

    def test_create_inparam_text_1(self):
        meshWrapper = InputFileWriter(MesherSettings)
        default_text = meshWrapper.create_inparam_text()
        
        exp_default_text = \
            ["BACKGROUND_MODEL\tprem_iso",
            "DOMINANT_PERIOD\t10",
            "NTHETA_SLICES\t1",
            "NRADIAL_SLICES\t1",
            "ONLY_SUGGEST_NTHETA\tfalse", 
            "WRITE_VTK\tfalse",
            "WRITE_1DMODEL\tfalse",
            "COARSENING_LAYERS\t3"]
        self.assertEqual(default_text, exp_default_text)

    def test_create_inparam_text_2(self):
        meshWrapper = InputFileWriter(MesherSettings)
        extra_settings = {"BACKGROUND_MODEL": "prem_20s",
                          "NPOL": 30,
                          "WRITE_VTK": True}
        default_text = meshWrapper.create_inparam_text(extra_settings)
        
        exp_default_text = \
            ["BACKGROUND_MODEL\tprem_20s",
            "DOMINANT_PERIOD\t10",
            "NTHETA_SLICES\t1",
            "NRADIAL_SLICES\t1",
            "ONLY_SUGGEST_NTHETA\tfalse", 
            "WRITE_VTK\ttrue",
            "WRITE_1DMODEL\tfalse",
            "COARSENING_LAYERS\t3",
            "NPOL\t30"]
        self.assertEqual(default_text, exp_default_text)

    def test_create_inparam_text_3(self):
        meshWrapper = InputFileWriter(MesherSettings)
        extra_settings = {"BACKGROUND_MODEL": "prem_20s",
                          "NPOL": 30,
                          "WRITE_VTK": False,
                          "INVALID_SETTING": None}

        self.assertRaises(ConfigurationError, meshWrapper.create_inparam_text, extra_settings)

    def test_create_yaml_inparam_text(self):
        meshWrapper = InputFileWriter(SolverMomentTensorSettings, "solve_name")
        default_text = meshWrapper.create_inparam_text()

        exp_default_text =     ["CMTSolution event info placeholder",
        "event name: \t201108231751A"  ,
        'time shift: \t1.11',
        'half duration: \t1.8',
        'latitude: \t37.91',
        'longitude: \t-77.93',
        'depth: \t12.0',
        'Mrr: \t4.71e+24',
        'Mtt: \t3.81e+22',
        'Mpp: \t-4.74e+24',
        'Mrt: \t3.99e+23',
        'Mrp: \t-8.05e+23',
        'Mtp: \t-1.23e+24']

        self.assertEqual(default_text, exp_default_text)

    def test_missing_solve_name(self):
        self.assertRaises(ConfigurationError, InputFileWriter, SolverMomentTensorSettings)

    def test_create_inparam_mesh_file(self):
        output_path = str((Path(__file__).parent / "../stubs/test_inparam_mesh").resolve())
        newMesherSettings = InputSettingsType(output_path, MesherSettings.required_values_defaults, 
                                                MesherSettings.allowed_keys, use_custom_filename = False)
        meshWrapper = InputFileWriter(newMesherSettings)
        extra_settings = {"BACKGROUND_MODEL": "prem",
                          "NPOL": 20,
                          "WRITE_VTK": True,
                          "NTHETA_SLICES": 2}
        
        meshWrapper.create_inparam_file(extra_settings)

        with open(output_path, "r") as f:
            inparam_file = f.read().splitlines() 
        
        exp_inparam_lines = \
            ["BACKGROUND_MODEL\tprem",
            "DOMINANT_PERIOD\t10",
            "NTHETA_SLICES\t2",
            "NRADIAL_SLICES\t1",
            "ONLY_SUGGEST_NTHETA\tfalse", 
            "WRITE_VTK\ttrue",
            "WRITE_1DMODEL\tfalse",
            "COARSENING_LAYERS\t3",
            "NPOL\t20"]

        self.assertEqual(inparam_file, exp_inparam_lines)

        os.remove(output_path)


if __name__ == '__main__':
    unittest.main()