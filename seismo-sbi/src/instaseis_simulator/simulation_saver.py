from pathlib import Path
import h5py

from .wrapper import GenericPointSource

class SimulationSaver:

    def __init__(self, simulation_inputs: GenericPointSource = None, output_data: dict = None, misc_data : dict = None):
        self.simulation_inputs = simulation_inputs
        self.output_data = output_data
        self.misc_data = misc_data

    def dump_data_as_hdf5(self, filepath):
        """Dump the inputs and outputs of the simulation to a single HDF5 file.

        Args:
            filepath (str): Filepath to dump the HDF5 to.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        instance_file = h5py.File(filepath, 'w')

        if self.simulation_inputs is not None:
            inputs_group = instance_file.create_group('inputs')
            self._save_inputs(inputs_group)

        if self.output_data is not None:
            outputs_group = instance_file.create_group('outputs')
            self._save_outputs(outputs_group, self.output_data)
        
        if self.misc_data is not None:
            misc_group = instance_file.create_group('misc')
            self._save_outputs(misc_group, self.misc_data)

    def _save_inputs(self, inputs_group: h5py.Group):
        for input_type, input_settings in self.simulation_inputs._asdict().items():
            input_types_group = inputs_group.create_group(input_type)
            for input_name, input_value in input_settings._asdict().items():
                input_types_group.attrs[input_name] = input_value

    def _save_outputs(self, outputs_group: h5py.Group, data: dict):
        for output_name, output_value in data.items():
            if not isinstance(output_value, dict):
                outputs_group.create_dataset(output_name, data=output_value)
            else:
                outputs_nested_group = outputs_group.create_group(output_name)
                for nested_output_name, nested_output_value in output_value.items():
                    outputs_nested_group.create_dataset(
                        nested_output_name, data=nested_output_value)
