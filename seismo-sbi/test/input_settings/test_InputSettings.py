import os
import sys
import unittest

from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from src.axisem_simulator.wrapper.input_settings.input_settings import InputSettings
from src.axisem_simulator.wrapper.input_settings.types import InputTypes
from src.axisem_simulator.wrapper.input_settings.presets import MesherSettings, SolverBasicSettings, SolverAdvancedSettings,\
                                                                    SolverSourceSettings, SolverMomentTensorSettings, InputSettingsType

class Testing_InputSettings(unittest.TestCase):

    def test_iterate_over_default_settings(self):
        input_settings = InputSettings()
        exp_settings = [MesherSettings, SolverBasicSettings, SolverAdvancedSettings,
                            SolverSourceSettings, SolverMomentTensorSettings]
        
        for i, (settings, user_values) in enumerate(input_settings.iterate_over_settings()):
            self.assertEqual(settings, exp_settings[i])
            self.assertEqual(user_values, exp_settings[i].required_values_defaults)

    def test_iterate_over_user_settings(self):
        input_settings = InputSettings()

        input_settings.overwrite_user_defined_settings({InputTypes.BASIC_SOLVER: {"LAT_HETEROGENEITY":  True}})
        exp_settings = [MesherSettings, SolverBasicSettings, SolverAdvancedSettings,
                            SolverSourceSettings, SolverMomentTensorSettings]
        
        for i, (_, user_values) in enumerate(input_settings.iterate_over_settings()):
            exp_user_values = exp_settings[i].required_values_defaults
            if i == 1: # solver basic settings
                exp_user_values["LAT_HETEROGENEITY"] = True
            self.assertEqual(user_values, exp_user_values)

    def test_get_user_settings(self):
        input_settings = InputSettings()

        input_settings.overwrite_user_defined_settings({InputTypes.BASIC_SOLVER: {"SIMULATION_TYPE":  "moment"}})

        setting = input_settings.get_setting(InputTypes.BASIC_SOLVER, "SIMULATION_TYPE")
        exp_setting = "moment"

        self.assertEqual(setting, exp_setting)

if __name__ == '__main__':
    unittest.main()