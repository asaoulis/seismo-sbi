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

import instaseis


class Simulator(ABC):

    def __init__(self, components, receivers: Receivers, seismogram_duration_in_s, synthetics_processing):
        
        self.components = components
        self.receivers = receivers
        self.seismogram_length = seismogram_duration_in_s
        self.synthetics_processing = synthetics_processing

    @abstractmethod
    def generic_point_source_simulation(self, source: GenericPointSource):
        pass

    def execute_sim_and_save_outputs(self, source_params, output_path):

        inputs, outputs = self.run_simulation(source_params)
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

    def run_simulation(self, source_parameters):
        ## combine source parameters and nuisance parameters dictionaries
        ## into one dictionary
        combined_params = source_parameters
        source_location_params = combined_params["source_location"]

        source_location = self._unpack_source_location_params(source_location_params)

        if "earthquake_magnitude" in combined_params.keys():
            moment = SimpleMomentTensor(combined_params["earthquake_magnitude"][0])
        elif "moment_tensor" in combined_params.keys():
            moment = GeneralMomentTensor(combined_params["moment_tensor"])
        else:
            raise InvalidConfiguration(f"Source mechanism specified incorrectly. No moment tensor or earthquake magnitude specified in {combined_params.keys()}.")

        source = GenericPointSource(source_location, moment)
        return source, self.generic_point_source_simulation(source)


class InstaseisSourceSimulator(Simulator):

    def __init__(self, instaseis_model_loc, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.instaseis_model_loc = instaseis_model_loc
        self.sampling_rate = float(InstaseisDBQuerier(self.instaseis_model_loc,
                                                      self.synthetics_processing,
                                                       self.seismogram_length).sampling_rate)

    def generic_point_source_simulation(self,  source: GenericPointSource):

        instaseis_db_querier = InstaseisDBQuerier(self.instaseis_model_loc,
                                                  self.synthetics_processing,
                                                    self.seismogram_length)

        all_seismograms_map = {}
        for receiver in self.receivers.iterate():
            all_seismograms_map[receiver.station_name] = {}
            receiver_results = instaseis_db_querier.get_seismograms(source,
                                                                         receiver, self.components)
            
            for component in self.components:
                all_seismograms_map[receiver.station_name][component] = receiver_results[component]

        return all_seismograms_map

class FixedLocationKernelSimulator(Simulator):

    def __init__(self, score_compression_data : ScoreCompressionData,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sensitivity_kernels = score_compression_data.data_parameter_gradients
        num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
        self.trace_length = self.sensitivity_kernels.shape[1] // num_traces

    def generic_point_source_simulation(self, source: GenericPointSource):
        
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