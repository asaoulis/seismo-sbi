"""Write and read the HDF5 file that holds one simulation.

:class:`SimulationSaver` writes the source parameters under ``inputs`` and the seismograms under
``outputs``; :class:`SimulationDataLoader` reads that layout back and flattens it into the
``(n_traces * n_samples,)`` data vector the inference pipeline consumes, in receiver order.
"""

import h5py
import numpy as np

from pathlib import Path

from .receivers import Receivers
from .sources import GenericPointSource
from seismo_sbi.utils.seismograms import apply_station_time_shifts


class SimulationSaver:

    def __init__(self, simulation_inputs: GenericPointSource = None, output_data: dict = None, misc_data : dict = None):
        self.simulation_inputs = simulation_inputs
        self.output_data = output_data
        self.misc_data = misc_data

    def dump_data_as_hdf5(self, filepath):
        """Write the simulation to ``filepath``, creating its parent directory."""
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


def to_numpy(obj):
    if isinstance(obj, h5py.Dataset):
        return obj[...]
    
    if isinstance(obj, (h5py.Group, dict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    
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
        return self.load_simulation_data_array(sim_name, *args, **kwargs)

    def load_flattened_simulation_vector_with_presence(self, sim_name, *args, **kwargs):
        """``(flat_vector, present_mask)``, tolerating stations absent from the file."""
        with h5py.File(sim_name, 'r') as simulation_data_map:
            return self.convert_sim_data_to_array_with_presence(
                simulation_data_map, *args, **kwargs)

    def load_simulation_data_array(self, sim_name, *args, **kwargs):
        with h5py.File(sim_name, 'r') as simulation_data_map:
            return self.convert_sim_data_to_array(simulation_data_map, *args, **kwargs)

    def load_input_and_data_array(self, sim_name, *args, **kwargs):
        """``(input_data_dict, data_array)`` from a single open of the file.

        The dataloader's per-sample path needs both, and opening the file twice costs.
        """
        with h5py.File(sim_name, 'r') as simulation_data_file:
            input_data = self._read_input_dict(simulation_data_file)
            data = self.convert_sim_data_to_array(simulation_data_file, *args, **kwargs)
        return input_data, data

    def load_simulation_data_array_with_shifts(self, sim_name, shift_dict, *args, **kwargs):
        self.receivers.set_time_shifts(shift_dict)
        with h5py.File(sim_name, 'r') as simulation_data_map:
            shifted_map = {"outputs": apply_station_time_shifts(self.receivers, to_numpy(simulation_data_map["outputs"]))}
            return self.convert_sim_data_to_array(shifted_map, *args, **kwargs)

    def load_event_subset(self, sim_name, subset_station_names, stacked=True):
        """``(data, coords)`` for the named stations only, in the order they are named.

        ``data`` is ``(n_stations, n_components, n_samples)`` when ``stacked``, else a flat
        vector; ``coords`` is ``(n_stations, 2)`` of latitude and longitude in degrees.
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

    def load_event_subset_with_components(self, sim_name, components_map, stacked=True):
        """``(data, coords, kept_stations)`` keeping only the components ``components_map``
        names and zero-filling the rest.

        ``components_map`` is ``{station: [kept components]}``; a station mapped to an empty
        list or absent is dropped entirely, and a partial list keeps those channels and zeroes
        the others, which is the ``component_dropout`` nuisance the model trained on. Channel
        rows always follow each receiver's master component order, never the order inside
        ``components_map``, and ``E``/``N`` match their ``1``/``2`` aliases. ``data`` is
        ``(n_stations, n_components, n_samples)`` when ``stacked``, ``coords`` is
        ``(n_stations, 2)`` of latitude and longitude in degrees.
        """
        name_to_rec = {rec.station_name: rec for rec in self.receivers.iterate()}
        master_order = [rec.station_name for rec in self.receivers.iterate()]
        kept_stations = [s for s in master_order
                         if components_map.get(s) and s in name_to_rec]
        data, coords = self.load_event_subset(sim_name, kept_stations, stacked=True)

        def _aliases(c):
            return {c, c.replace('E', '1').replace('N', '2'),
                    c.replace('1', 'E').replace('2', 'N')}

        for i, s in enumerate(kept_stations):
            kept = set()
            for c in components_map[s]:
                kept |= _aliases(c)
            for ci, comp in enumerate(name_to_rec[s].components):
                if comp not in kept:
                    data[i, ci, :] = 0.0
        if not stacked:
            data = data.reshape(-1)
        return data, coords, kept_stations

    def convert_sim_data_to_array(self, simulation_data_map, scale_dict=None, stacked=False, fill_unused=False):
        """Seismogram array from a simulation map; an absent station raises ``KeyError``."""
        array, _ = self._convert_sim_data(
            simulation_data_map, scale_dict, stacked, fill_unused, allow_missing=False
        )
        return array

    def convert_sim_data_to_array_with_presence(self, simulation_data_map, scale_dict=None,
                                                stacked=False, fill_unused=False):
        """``(array, present_mask)``, tolerating absent stations.

        ``present_mask`` is a boolean over ``receivers`` saying which stations the map
        carried. An absent station is zero-filled so the array keeps its full-station shape;
        those samples are padding, not data, and the caller must mask them out.
        """
        return self._convert_sim_data(
            simulation_data_map, scale_dict, stacked, fill_unused, allow_missing=True
        )

    def _convert_sim_data(self, simulation_data_map, scale_dict=None, stacked=False,
                          fill_unused=False, allow_missing=False):
        """Shared implementation of the two public wrappers above.

        ``scale_dict`` is ``{station: {component: scale factor}}``; ``stacked`` returns
        ``(n_stations, n_components, n_samples)`` instead of a flat vector.
        """
        try:
            seismogram_array_length = self._get_seismogram_array_length(simulation_data_map)
        except KeyError:
            # Only reachable with allow_missing: the probe receiver is the absent one.
            if not allow_missing or self.data_length is None:
                raise
            seismogram_array_length = self.data_length
        if self.data_length is not None:
            seismogram_array_length = min(seismogram_array_length, self.data_length)

        station_data = []
        present = []

        # Fetched once rather than per station and component: this is the dataloader's
        # per-sample path.
        outputs_group = simulation_data_map["outputs"]
        for receiver in self.receivers.iterate():
            receiver_name = receiver.station_name
            rec_components = receiver.components
            if allow_missing:
                station_outputs = outputs_group.get(receiver_name)
                if station_outputs is None:
                    # Zero-filled so the vector keeps its shape; the mask marks it as padding.
                    station_data.append([np.zeros(seismogram_array_length)
                                         for _ in rec_components])
                    present.append(False)
                    continue
            else:
                station_outputs = outputs_group[receiver_name]
            comp_data = []
            for component in rec_components:
                alt_component = component.replace('E', '1').replace('N', '2')

                trace_data = station_outputs.get(component)
                if trace_data is None:
                    trace_data = station_outputs.get(alt_component)

                if trace_data is None:
                    if allow_missing:
                        # A part-present station counts as absent; otherwise it would reach
                        # the model as part-zero data.
                        comp_data = None
                        break
                    raise KeyError(f"No data found for {receiver_name}:{component}")

                trace_data_vector = trace_data[:seismogram_array_length]

                # Skipped without a scale_dict: every factor would be 1.0, and dividing by it
                # copies every trace for nothing.
                if scale_dict is not None:
                    factor = scale_dict.get(receiver_name, {}).get(component)
                    if factor is None:
                        factor = scale_dict.get(receiver_name, {}).get(alt_component, 1.0)
                    if factor != 1.0:
                        trace_data_vector = trace_data_vector / np.sqrt(factor)

                comp_data.append(trace_data_vector)

            if comp_data is None:
                station_data.append([np.zeros(seismogram_array_length) for _ in rec_components])
                present.append(False)
                continue
            station_data.append(comp_data)
            present.append(True)
        if fill_unused:
            station_data = self.zero_fill_unused_components([comp for station in station_data for comp in station], seismogram_array_length)
        if stacked:
            array = np.array(station_data)
        else:
            array= np.concatenate([comp for comps in station_data for comp in comps])
        return array, np.asarray(present, dtype=bool)

    def zero_fill_unused_components(self, flattened_list, seismogram_array_length):
        """Zero-fill unused components in the seismogram array."""
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
