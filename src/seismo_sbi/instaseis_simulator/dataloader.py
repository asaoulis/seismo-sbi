import h5py
import numpy as np

from pathlib import Path

from .receivers import Receivers
from .wrapper import GenericPointSource


class SimulationDataLoader():

    def __init__(self,components : str,
                        receivers : Receivers,
                        data_length = None,):

        self.components = components
        self.receivers = receivers
        self.data_length = data_length

    def load_input_data(self, sim_name):
        
        input_data = {}
        with h5py.File(sim_name, 'r') as simulation_data_file:
            simulation_data_file = h5py.File(sim_name, 'r')

            inputs = simulation_data_file["inputs"]
            for input_type in GenericPointSource._fields:

                input_data[input_type] = dict(inputs[input_type].attrs)
            
        return input_data

    def load_flattened_simulation_vector(self, sim_name, *args, **kwargs):
        # make sure this is returning things in the order we want
        return self.load_simulation_data_array(sim_name, *args, **kwargs)

    def load_simulation_data_array(self, sim_name, *args, **kwargs):
        # context manager to close file
        with h5py.File(sim_name, 'r') as simulation_data_map:
            return self.convert_sim_data_to_array(simulation_data_map, *args, **kwargs)

    def convert_sim_data_to_array(self, simulation_data_map, scale_dict=None, stacked=False):
        """Convert simulation data map into seismogram array.

        Args:
            simulation_data_map (dict): Mapping of simulation outputs.
            scale_dict (dict, optional): Nested dict {station: {component: scale_factor}}.
            stacked (bool, optional): If True, returns shape 
                (num_stations, num_components, trace_length). Otherwise returns flat array.
                Defaults to False.

        Returns:
            np.ndarray: Seismogram data.
        """
        seismogram_array_length = self._get_seismogram_array_length(simulation_data_map)
        if self.data_length is not None:
            seismogram_array_length = min(seismogram_array_length, self.data_length)

        station_data = []

        for receiver in self.receivers.iterate():
            receiver_name = receiver.station_name
            components = receiver.components
            comp_data = []

            for component in components:
                # Handle component name remapping
                alt_component = component.replace('E', '1').replace('N', '2')

                # Get scale factor
                factor = 1.0
                if scale_dict is not None:
                    factor = scale_dict.get(receiver_name, {}).get(component)
                    if factor is None:
                        factor = scale_dict.get(receiver_name, {}).get(alt_component, 1.0)

                # Get trace data
                outputs = simulation_data_map["outputs"][receiver_name]
                trace_data = outputs.get(component)
                if trace_data is None:
                    trace_data = outputs.get(alt_component)

                if trace_data is None:
                    raise KeyError(f"No data found for {receiver_name}:{component}")

                trace_data_vector = trace_data[:seismogram_array_length] / np.sqrt(factor)
                comp_data.append(trace_data_vector)

            station_data.append(comp_data)

        if stacked:
            # shape (num_stations, num_components, trace_length)
            return np.array(station_data)
        else:
            # Flatten into single long vector
            return np.concatenate([comp for comps in station_data for comp in comps])

    def load_misc_data(self, sim_name):
        with h5py.File(sim_name, 'r') as simulation_data_map:
            misc_data = {}
            misc_group = simulation_data_map["misc"]
            for receiver in self.receivers.iterate():
                receiver_name = receiver.station_name
                components = receiver.components
                misc_data[receiver_name] = {}
                for  component in components:
                    try:
                        misc_data[receiver_name][component] = misc_group[receiver_name][component][()]
                    except KeyError:
                        component = component.replace('E', '1').replace('N', '2')
                        misc_data[receiver_name][component] = misc_group[receiver_name][component][()]
            return misc_data

    def _get_seismogram_array_length(self, simulation_data_file):
        dummy_receiver = self.receivers.receivers[0]
        return len(simulation_data_file["outputs"][dummy_receiver.station_name][dummy_receiver.components[0]])