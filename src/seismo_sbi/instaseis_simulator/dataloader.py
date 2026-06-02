import h5py
import numpy as np

from pathlib import Path

from .receivers import Receivers
from .utils import apply_station_time_shifts
from .wrapper import GenericPointSource
def to_numpy(obj):
    # 1) If Dataset → load to NumPy
    if isinstance(obj, h5py.Dataset):
        return obj[...]  # or obj[:]
    
    # 2) If Group or dict → recurse
    if isinstance(obj, (h5py.Group, dict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    
    # 3) If it's already a numpy array or other object → return as is
    return obj
class SimulationDataLoader():

    def __init__(self,components : str,
                        receivers : Receivers,
                        data_length = None,):

        self.components = components
        self.receivers = receivers
        self.data_length = data_length

    @staticmethod
    def _read_input_dict(simulation_data_file):
        """Read the ``inputs`` group of an open h5 file into ``{input_type: {attr: val}}``."""
        inputs = simulation_data_file["inputs"]
        return {
            input_type: dict(inputs[input_type].attrs)
            for input_type in GenericPointSource._fields
        }

    def load_input_data(self, sim_name):
        with h5py.File(sim_name, 'r') as simulation_data_file:
            return self._read_input_dict(simulation_data_file)

    def load_flattened_simulation_vector(self, sim_name, *args, **kwargs):
        # make sure this is returning things in the order we want
        return self.load_simulation_data_array(sim_name, *args, **kwargs)

    def load_simulation_data_array(self, sim_name, *args, **kwargs):
        # context manager to close file
        with h5py.File(sim_name, 'r') as simulation_data_map:
            return self.convert_sim_data_to_array(simulation_data_map, *args, **kwargs)

    def load_input_and_data_array(self, sim_name, *args, **kwargs):
        """Open the h5 file ONCE and return ``(input_data_dict, data_array)``.

        Equivalent to calling :meth:`load_input_data` followed by
        :meth:`load_simulation_data_array`, but with a single ``h5py.File`` open
        instead of two — the per-sample hot path in the ML dataloader needs both
        theta (from ``inputs``) and the seismogram array (from ``outputs``), and
        opening the file twice per ``__getitem__`` is a measurable cost.
        """
        with h5py.File(sim_name, 'r') as simulation_data_file:
            input_data = self._read_input_dict(simulation_data_file)
            data = self.convert_sim_data_to_array(simulation_data_file, *args, **kwargs)
        return input_data, data

    def load_simulation_data_array_with_shifts(self, sim_name, shift_dict, *args, **kwargs):
        # context manager to close file
        self.receivers.set_time_shifts(shift_dict)
        with h5py.File(sim_name, 'r') as simulation_data_map:
            shifted_map = {"outputs": apply_station_time_shifts(self.receivers, to_numpy(simulation_data_map["outputs"]))}
            return self.convert_sim_data_to_array(shifted_map, *args, **kwargs)

    def load_event_subset(self, sim_name, subset_station_names, stacked=True):
        """Load an event/simulation H5 restricted to a SUBSET of stations.

        Because the H5 ``outputs`` group is keyed by station name, selecting a subset is a
        load-time operation: we temporarily restrict ``self.receivers`` to the requested
        stations (preserving the order of ``subset_station_names``) and read only those.
        Used for variable-station inference, where a trained model is applied to a subset of
        its master station set.

        Returns
        -------
        (data, coords) : data is ``(N, C, T)`` when ``stacked`` else a flat vector; coords is
        ``(N, 2)`` of ``(latitude, longitude)``, ordered to match ``subset_station_names``.
        """
        name_to_rec = {rec.station_name: rec for rec in self.receivers.iterate()}
        missing = [n for n in subset_station_names if n not in name_to_rec]
        if missing:
            raise KeyError(
                f"Requested stations not in the master receiver set: {missing}. "
                f"Available: {list(name_to_rec)}"
            )
        subset = [name_to_rec[n] for n in subset_station_names]
        coords = np.array([[rec.latitude, rec.longitude] for rec in subset], dtype=float)

        saved = self.receivers.receivers
        try:
            self.receivers.receivers = subset
            data = self.load_simulation_data_array(sim_name, stacked=stacked)
        finally:
            self.receivers.receivers = saved
        return data, coords

    def convert_sim_data_to_array(self, simulation_data_map, scale_dict=None, stacked=False, fill_unused=False):
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

        # Fetch the outputs group once (not once per (station, component)); reading
        # each per-component dataset is the per-sample dataloader hot path.
        outputs_group = simulation_data_map["outputs"]
        for receiver in self.receivers.iterate():
            receiver_name = receiver.station_name
            rec_components = receiver.components
            station_outputs = outputs_group[receiver_name]
            comp_data = []
            for component in rec_components:
                # Handle component name remapping
                alt_component = component.replace('E', '1').replace('N', '2')

                # Get trace data
                trace_data = station_outputs.get(component)
                if trace_data is None:
                    trace_data = station_outputs.get(alt_component)

                if trace_data is None:
                    raise KeyError(f"No data found for {receiver_name}:{component}")

                trace_data_vector = trace_data[:seismogram_array_length]

                # Apply the scale factor only when a scale_dict is supplied. With no
                # scale_dict every factor is 1.0, so the old `/ np.sqrt(1.0)` just
                # allocated a redundant copy of every trace — skip it.
                if scale_dict is not None:
                    factor = scale_dict.get(receiver_name, {}).get(component)
                    if factor is None:
                        factor = scale_dict.get(receiver_name, {}).get(alt_component, 1.0)
                    if factor != 1.0:
                        trace_data_vector = trace_data_vector / np.sqrt(factor)

                comp_data.append(trace_data_vector)

            station_data.append(comp_data)
        if fill_unused:
            station_data = self.zero_fill_unused_components([comp for station in station_data for comp in station], seismogram_array_length)
        if stacked:
            # shape (num_stations, num_components, trace_length)
            array = np.array(station_data)
        else:
            # Flatten into single long vector
            array= np.concatenate([comp for comps in station_data for comp in comps])
        return array

    def zero_fill_unused_components(self, flattened_list, seismogram_array_length):
        """Zero-fill unused components in the seismogram array.
        """
        all_comps = []
        index = 0
        for receiver in self.receivers.iterate():
            rec_components = receiver.components 
            receiver_components = []
            for component in self.components:
                if component in rec_components:
                    trace_data_vector = flattened_list[index]
                    receiver_components.append(trace_data_vector)
                    index += 1
                else:
                    receiver_components.append(np.zeros(seismogram_array_length))
            all_comps.append(receiver_components)

        return all_comps

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
        try:
            first_component = dummy_receiver.components[0]
            return len(simulation_data_file["outputs"][dummy_receiver.station_name][first_component])
        except KeyError:
            first_component = dummy_receiver.components[0].replace('E', '1').replace('N', '2')
            return len(simulation_data_file["outputs"][dummy_receiver.station_name][first_component])