from pathlib import Path
import h5py

import numpy as np

from abc import ABC, abstractmethod

from .receivers import Receivers
from .simulation_saver import SimulationSaver
from .wrapper import GenericPointSource, InstaseisDBQuerier, SimpleMomentTensor, \
    GeneralMomentTensor, SourceLocation
from seismo_sbi.sbi.configuration import InvalidConfiguration
from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData
from .utils import apply_station_time_shifts
from .post_processing import PostProcessingChain


# Keys consumed directly by run_simulation() / generic_point_source_simulation().
# Any key in source_parameters that is NOT in this set is treated as a
# post-processing nuisance parameter and forwarded to the PostProcessingChain.
_SIMULATOR_KEYS = frozenset({
    "source_location",
    "moment_tensor",
    "earthquake_magnitude",
    "velocity_model",
    "use_fiducial",
    "stf_duration",
})


class Simulator(ABC):

    def __init__(
        self,
        components,
        receivers: Receivers,
        seismogram_duration_in_s,
        synthetics_processing,
        post_processing_effects=None,
    ):
        self.components = components
        self.receivers = receivers
        self.seismogram_length = seismogram_duration_in_s
        self.synthetics_processing = synthetics_processing
        self.post_processing_chain = PostProcessingChain(post_processing_effects or [])

    @abstractmethod
    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        pass

    def execute_sim_and_save_outputs(self, source_params, output_path, **kwargs):

        inputs, outputs = self.run_simulation(source_params, **kwargs)
        self.save_simulation(inputs, outputs, output_path)

    def save_simulation(self, inputs: GenericPointSource, outputs, output_path: str):

        sim_saver = SimulationSaver(
            simulation_inputs=inputs, output_data=outputs)
        sim_saver.dump_data_as_hdf5(output_path)

    def _unpack_source_location_params(self, source_location_params):
        lat = source_location_params[0]
        long = source_location_params[1]
        depth = source_location_params[2]
        time_shift = source_location_params[3]

        source_location = SourceLocation(lat, long, depth, time_shift)
        return source_location

    def run_simulation(self, source_parameters, **kwargs):
        ## combine source parameters and nuisance parameters dictionaries
        ## into one dictionary — copy to avoid mutating the caller's dict
        combined_params = dict(source_parameters)

        # Collect any keys not consumed by the forward model; these are routed
        # to the post-processing chain after the simulation.
        post_proc_params = {
            key: combined_params[key]
            for key in list(combined_params)
            if key not in _SIMULATOR_KEYS
        }

        source_location_params = combined_params["source_location"]
        # Path-dependent effects (distance-scaled scattering, azimuthal anisotropy) need the
        # source position; every effect swallows unknown kwargs, so forwarding it is inert
        # for the others.
        post_proc_params.setdefault("source_location", source_location_params)
        velocity_model_params = combined_params.pop("velocity_model", None)
        stf_duration = combined_params.pop("stf_duration", None)
        use_fiducial = combined_params.pop("use_fiducial", None)
        # add use_fiducial to kwargs if it doesn't exist
        if kwargs.get("use_fiducial") is None:
            kwargs["use_fiducial"] = use_fiducial

        source_location = self._unpack_source_location_params(source_location_params)

        if "earthquake_magnitude" in combined_params.keys():
            moment = SimpleMomentTensor(combined_params["earthquake_magnitude"][0])
        elif "moment_tensor" in combined_params.keys():
            moment = GeneralMomentTensor(combined_params["moment_tensor"])
        else:
            raise InvalidConfiguration(
                f"Source mechanism specified incorrectly. No moment tensor or "
                f"earthquake magnitude specified in {combined_params.keys()}."
            )

        source = GenericPointSource(source_location, moment)
        all_seismograms_map = self.generic_point_source_simulation(
            source,
            velocity_model=velocity_model_params,
            stf_duration=stf_duration,
            **kwargs,
        )
        shifted_seismograms_map = apply_station_time_shifts(self.receivers, all_seismograms_map)

        # Apply post-processing effects (amplitude errors, dropout, etc.)
        processed_seismograms_map = self.post_processing_chain(
            shifted_seismograms_map, self.receivers, post_proc_params
        )
        return source, processed_seismograms_map


class InstaseisSourceSimulator(Simulator):

    def __init__(self, instaseis_model_loc, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instaseis_model_loc = instaseis_model_loc
        self.sampling_rate = float(InstaseisDBQuerier(self.instaseis_model_loc,
                                                      self.synthetics_processing,
                                                       self.seismogram_length).sampling_rate)

    def generic_point_source_simulation(self, source: GenericPointSource, *, stf_duration=None, **kwargs):

        instaseis_db_querier = InstaseisDBQuerier(self.instaseis_model_loc,
                                                  self.synthetics_processing,
                                                    self.seismogram_length)

        all_seismograms_map = {}
        for receiver in self.receivers.iterate():
            all_seismograms_map[receiver.station_name] = {}
            receiver_results = instaseis_db_querier.get_seismograms(
                source, receiver, self.components, stf_duration=stf_duration
            )

            for component in self.components:
                all_seismograms_map[receiver.station_name][component] = receiver_results[component]

        return all_seismograms_map

class FixedLocationKernelSimulator(Simulator):

    def __init__(self, score_compression_data : ScoreCompressionData = None,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # score_compression_data may be None when the simulator is constructed before the
        # kernels are known (e.g. as the initial simulator in a pipeline that will swap in
        # real kernels via use_kernel_simulator_if_possible). Kernels are required before
        # any simulation is actually run.
        if score_compression_data is None:
            self.sensitivity_kernels = None
            self.trace_length = None
        else:
            self.sensitivity_kernels = score_compression_data.data_parameter_gradients
            num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
            self.trace_length = self.sensitivity_kernels.shape[1] // num_traces

    def generic_point_source_simulation(self, source: GenericPointSource, *, stf_duration=None, **kwargs):
        
        all_seismograms_map = {}

        seismograms = self._compute_seismograms_from_kernels(source)

        seismograms = seismograms.reshape(-1, self.trace_length)

        trace_counter = 0
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            all_seismograms_map[receiver.station_name] = {}
            for comp_idx, component in enumerate(receiver.components):
                all_seismograms_map[receiver.station_name][component] = seismograms[trace_counter]
                trace_counter +=1
            
        return all_seismograms_map
    
    def _compute_seismograms_from_kernels(self, source: GenericPointSource):

        moment_tensor_components = source.moment_tensor.components
        seismograms = np.dot(self.sensitivity_kernels.T, moment_tensor_components)
        return seismograms
