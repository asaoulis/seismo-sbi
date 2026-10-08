"""Write and read the HDF5 file that holds one simulation.

:class:`SimulationSaver` writes the source parameters under ``inputs`` and the seismograms under
``outputs``; :class:`SimulationDataLoader` reads that layout back and flattens it into the
``(n_traces * n_samples,)`` data vector the inference pipeline consumes, in receiver order.
:func:`seismogram_map_to_array` does the same for a simulator's in-memory ``{station: {component:
waveform}}`` map, and :func:`seismogram_array_to_map` builds that map from the traces. Horizontal components may be stored as 1 and 2 rather than E and N;
``component_alias`` maps them.
"""

import h5py
import numpy as np

from pathlib import Path

from .receivers import Receivers
from .sources import GenericPointSource
from seismo_sbi.utils.seismograms import apply_station_time_shifts
from seismo_sbi.utils.errors import InvalidConfiguration


def component_alias(components: str) -> str:
    """``components`` with E and N renamed 1 and 2; one component or a string of them."""
    return components.replace('E', '1').replace('N', '2')


def seismogram_map_to_array(seismogram_map: dict, receivers: Receivers, stacked: bool = False):
    """The data vector of a ``{station: {component: waveform}}`` map, in receiver order.

    Each receiver contributes its own ``components`` in its own order; ``stacked`` returns
    ``(n_stations, n_components, n_samples)`` instead of the flat ``(n_traces * n_samples,)``
    vector. A station or component missing from the map raises ``KeyError``.
    """
    components = "".join(receivers.receivers[0].components)
    loader = SimulationDataLoader(components, receivers)
    return loader.convert_sim_data_to_array({"outputs": seismogram_map}, stacked=stacked)


def seismogram_array_to_map(traces, receivers: Receivers) -> dict:
    """``{station: {component: waveform}}`` of ``traces`` ``(n_traces, n_samples)``, one row per
    receiver component in receiver order; the inverse of :func:`seismogram_map_to_array`.
    """
    seismogram_map = {}
    trace_counter = 0
    for receiver in receivers.iterate():
        seismogram_map[receiver.station_name] = {}
        for component in receiver.components:
            seismogram_map[receiver.station_name][component] = traces[trace_counter]
            trace_counter += 1
    return seismogram_map


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
    """Read simulation HDF5 files for a fixed set of receivers and components.

    Arrays come back as ``(n_stations, n_components, n_samples)`` when ``stacked``, else flattened
    in receiver order; components a receiver lacks are zero-filled when ``fill_unused`` is set.
    """

    def __init__(self,components : str,
                        receivers : Receivers,
                        data_length = None,):
        """``components`` (a string such as ``"ZEN"``, or a list) is the full component layout
        a ``fill_unused`` array is padded to; each receiver's own ``components`` must follow the
        same relative order. ``data_length`` truncates every trace to that many samples.
        """
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

    def load_simulation_data_array(self, sim_name, *, stacked=False, fill_unused=False, data_length=None):
        """The seismograms of the file ``sim_name``, flat in receiver order or ``(n_stations,
        n_components, n_samples)`` when ``stacked``; an absent station raises ``KeyError``.

        ``data_length`` truncates every trace to that many samples; None keeps the loader's own.
        """
        with h5py.File(sim_name, 'r') as simulation_data_map:
            return self.convert_sim_data_to_array(simulation_data_map, stacked=stacked, fill_unused=fill_unused,
                                                  data_length=data_length)

    def load_simulation_data_array_with_presence(self, sim_name, *, stacked=False, fill_unused=False):
        """``(array, present_mask)`` of the file ``sim_name``, tolerating stations absent from it."""
        with h5py.File(sim_name, 'r') as simulation_data_map:
            return self.convert_sim_data_to_array_with_presence(simulation_data_map, stacked=stacked,
                                                                fill_unused=fill_unused)

    def load_input_and_data_array(self, sim_name, *, stacked=False, fill_unused=False):
        """``(input_data_dict, data_array)`` from a single open of the file."""
        with h5py.File(sim_name, 'r') as simulation_data_file:
            input_data = self._read_input_dict(simulation_data_file)
            data = self.convert_sim_data_to_array(simulation_data_file, stacked=stacked, fill_unused=fill_unused)
        return input_data, data

    def load_simulation_data_array_with_shifts(self, sim_name, shift_dict, *, stacked=False, fill_unused=False):
        """Load a simulation with per-station time shifts ``shift_dict`` applied to its traces."""
        self.receivers.set_time_shifts(shift_dict)
        with h5py.File(sim_name, 'r') as simulation_data_map:
            shifted_map = {"outputs": apply_station_time_shifts(self.receivers, to_numpy(simulation_data_map["outputs"]))}
            return self.convert_sim_data_to_array(shifted_map, stacked=stacked, fill_unused=fill_unused)

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

        subset_loader = SimulationDataLoader(self.components, Receivers(receivers=subset), self.data_length)
        data = subset_loader.load_simulation_data_array(sim_name, stacked=stacked)
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
            return {c, component_alias(c),
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

    def convert_sim_data_to_array(self, simulation_data_map, *, stacked=False, fill_unused=False, data_length=None):
        """Seismogram array from a simulation map; an absent station raises ``KeyError``."""
        array, _ = self._convert_sim_data(simulation_data_map, stacked=stacked, fill_unused=fill_unused,
                                          allow_missing=False, data_length=data_length)
        return array

    def convert_sim_data_to_array_with_presence(self, simulation_data_map, *, stacked=False, fill_unused=False):
        """``(array, present_mask)``, tolerating absent stations.

        ``present_mask`` is a boolean over ``receivers`` saying which stations the map
        carried. An absent station is zero-filled so the array keeps its full-station shape;
        those samples are padding, not data, and the caller must mask them out.
        """
        return self._convert_sim_data(simulation_data_map, stacked=stacked, fill_unused=fill_unused,
                                      allow_missing=True)

    def _convert_sim_data(self, simulation_data_map, *, stacked, fill_unused, allow_missing, data_length=None):
        """Shared implementation of the two public wrappers above.

        ``stacked`` returns ``(n_stations, n_components, n_samples)`` instead of a flat vector;
        ``data_length`` (None: the loader's own) truncates every trace.
        """
        if data_length is None:
            data_length = self.data_length
        try:
            seismogram_array_length = self._get_seismogram_array_length(simulation_data_map)
        except KeyError:
            # Only reachable with allow_missing: the probe receiver is the absent one.
            if not allow_missing or data_length is None:
                raise
            seismogram_array_length = data_length
        if data_length is not None:
            seismogram_array_length = min(seismogram_array_length, data_length)

        station_data = []
        present = []

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
                alt_component = component_alias(component)

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

                comp_data.append(trace_data[:seismogram_array_length])

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
            self._check_component_order(receiver)
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

    def _check_component_order(self, receiver):
        """Raise unless ``receiver.components`` follow the loader's component order."""
        layout = [component_alias(component) for component in self.components]
        positions = [layout.index(component_alias(component)) for component in receiver.components
                     if component_alias(component) in layout]
        if positions != sorted(positions):
            raise InvalidConfiguration(
                f"Station {receiver.station_name} records {list(receiver.components)}, not in the "
                f"order of the components {list(self.components)} the data vector is laid out in."
            )

    def load_misc_data(self, sim_name, allow_missing=False):
        """``{station: {component: autocovariance}}`` from the ``misc`` group of ``sim_name``.

        With ``allow_missing`` a station absent from the group, or missing one of its components, is
        left out; otherwise it raises ``KeyError``.
        """
        with h5py.File(sim_name, 'r') as simulation_data_map:
            misc_data = {}
            misc_group = simulation_data_map["misc"]
            for receiver in self.receivers.iterate():
                receiver_name = receiver.station_name
                components = receiver.components
                misc_data[receiver_name] = {}
                try:
                    for  component in components:
                        try:
                            misc_data[receiver_name][component] = misc_group[receiver_name][component][()]
                        except KeyError:
                            component = component_alias(component)
                            misc_data[receiver_name][component] = misc_group[receiver_name][component][()]
                except KeyError:
                    if not allow_missing:
                        raise
                    del misc_data[receiver_name]
            return misc_data

    def _get_seismogram_array_length(self, simulation_data_file):
        dummy_receiver = self.receivers.receivers[0]
        try:
            first_component = dummy_receiver.components[0]
            return len(simulation_data_file["outputs"][dummy_receiver.station_name][first_component])
        except KeyError:
            first_component = component_alias(dummy_receiver.components[0])
            return len(simulation_data_file["outputs"][dummy_receiver.station_name][first_component])
