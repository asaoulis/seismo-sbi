"""The simulator interface every forward model implements.

:class:`Simulator` turns a source-parameter dictionary into a map of seismograms: it builds the
moment tensor and source location, calls the backend's ``generic_point_source_simulation``,
applies the per-station time shifts and runs the post-processing chain. A backend subclasses it
and implements that one method.
"""

from abc import ABC, abstractmethod

from .receivers import Receivers
from .simulation_io import SimulationSaver
from .sources import GenericPointSource, SimpleMomentTensor, GeneralMomentTensor, SourceLocation
from seismo_sbi.utils.errors import InvalidConfiguration
from seismo_sbi.utils.seismograms import apply_station_time_shifts
from seismo_sbi.nuisance_effects.post_processing import PostProcessingChain


#: Source-parameter keys the forward model consumes; every other key is a nuisance parameter
#: and is forwarded to the post-processing chain.
_SIMULATOR_KEYS = frozenset({
    "source_location",
    "moment_tensor",
    "earthquake_magnitude",
    "velocity_model",
    "use_fiducial",
    "stf_duration",
    "seed",
})


class Simulator(ABC):

    def __init__(
        self,
        components,
        receivers: Receivers,
        seismogram_duration_in_s,
        synthetics_processing,
        post_processing_effects=None,
        source_depth_offset_km: float = 0.0,
    ):
        self.components = components
        self.receivers = receivers
        self.seismogram_length = seismogram_duration_in_s
        self.synthetics_processing = synthetics_processing
        # Applied only where a depth is handed to the backend; zero means the catalogue datum
        # is the model's free surface.
        self.source_depth_offset_km = float(source_depth_offset_km)
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
        # Copied so the caller's dict is not mutated.
        combined_params = dict(source_parameters)

        # Everything the forward model does not consume goes to the post-processing chain.
        post_proc_params = {
            key: combined_params[key]
            for key in list(combined_params)
            if key not in _SIMULATOR_KEYS
        }

        source_location_params = combined_params["source_location"]
        # Path-dependent effects need the source position; the others swallow it.
        post_proc_params.setdefault("source_location", source_location_params)
        velocity_model_params = combined_params.pop("velocity_model", None)
        stf_duration = combined_params.pop("stf_duration", None)
        use_fiducial = combined_params.pop("use_fiducial", None)
        if kwargs.get("use_fiducial") is None:
            kwargs["use_fiducial"] = use_fiducial
        seed = combined_params.pop("seed", None)
        if seed is not None:
            kwargs["seed"] = seed

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

        processed_seismograms_map = self.post_processing_chain(
            shifted_seismograms_map, self.receivers, post_proc_params
        )
        return source, processed_seismograms_map
