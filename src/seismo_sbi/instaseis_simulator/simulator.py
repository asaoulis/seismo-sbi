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
from seismo_sbi.cps_simulator.compatibility import build_objstats
from seismo_sbi.cps_simulator.CPS import update_with_Gtensor
import instaseis


class Simulator(ABC):

    def __init__(self, components, receivers: Receivers, seismogram_duration_in_s, synthetics_processing):
        
        self.components = components
        self.receivers = receivers
        self.seismogram_length = seismogram_duration_in_s
        self.synthetics_processing = synthetics_processing

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
        ## into one dictionary
        combined_params = source_parameters
        source_location_params = combined_params["source_location"]
        velocity_model_params = combined_params.pop("velocity_model", None)
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
            raise InvalidConfiguration(f"Source mechanism specified incorrectly. No moment tensor or earthquake magnitude specified in {combined_params.keys()}.")

        source = GenericPointSource(source_location, moment)
        return source, self.generic_point_source_simulation(source, velocity_model=velocity_model_params, **kwargs)


class InstaseisSourceSimulator(Simulator):

    def __init__(self, instaseis_model_loc, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.instaseis_model_loc = instaseis_model_loc
        self.sampling_rate = float(InstaseisDBQuerier(self.instaseis_model_loc,
                                                      self.synthetics_processing,
                                                       self.seismogram_length).sampling_rate)

    def generic_point_source_simulation(self,  source: GenericPointSource, **kwargs):

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

    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        
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
# convert newtons into dynes
CPS_INPUT_COVERSION = 1.e-7
CPS_OUTPUT_COVERSION = 1.e-2

class CPSSimulator(Simulator):
    
    def __init__(self, gf_storage_root=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitivity_kernels = None
        self.num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
        self.gf_storage_root = gf_storage_root

    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        
        all_seismograms_map = {}
        velocity_model = kwargs.pop('velocity_model', None)
        self.sensitivity_kernels = self.compute_greens_functions(source, velocity_model, **kwargs)

        seismograms = self._compute_seismograms_from_kernels(source)
        trace_length = self.sensitivity_kernels.shape[1] // self.num_traces
        seismograms = seismograms.reshape(-1, trace_length)

        trace_counter = 0
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            all_seismograms_map[receiver.station_name] = {}
            for comp_idx, component in enumerate(receiver.components):
                all_seismograms_map[receiver.station_name][component] = seismograms[trace_counter]
                trace_counter +=1
            
        return all_seismograms_map
    
    def compute_greens_functions(self, source: GenericPointSource, velocity_model, **kwargs):
        objstats = build_objstats(self.receivers, source, self.seismogram_length)
        greens_functions = self.compute_or_load_greens_functions(objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir=self.gf_storage_root, return_gf=True, **kwargs)
        greens_functions = greens_functions.transpose(2, 0, 1, 3)

        trace_counter = 0
        used_greens_functions = []
        comp_ids = ['Z', 'E', 'N']
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            for component in receiver.components:
                idx = comp_ids.index(component)
                used_greens_functions.append(greens_functions[:, rec_idx, idx, :])

        return np.concatenate(used_greens_functions, axis=1)

    def _compute_seismograms_from_kernels(self, source: GenericPointSource):

        moment_tensor_components = source.moment_tensor.components
        seismograms = np.dot(self.sensitivity_kernels.T, np.array(moment_tensor_components) * CPS_INPUT_COVERSION) * CPS_OUTPUT_COVERSION
        return seismograms
    
    @abstractmethod
    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        """
        Abstract method to compute or load Green's functions.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
class CPSVariableKernelSimulator(CPSSimulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # delete everything in gf_storage_root
        if Path(self.gf_storage_root).exists():
            for item in Path(self.gf_storage_root).iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for sub_item in item.iterdir():
                        sub_item.unlink()
                    item.rmdir()

    
    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        return update_with_Gtensor(
            objstats, velocity_model, delta=delta, force_calc=force_calc, verbose=verbose, rootdir=rootdir, return_gf=return_gf, filter_params=self.synthetics_processing['filter'], **kwargs)


from pathlib import Path
class CPSPrecomputedSimulator(CPSSimulator):
    
    def __init__(self, fiducial_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cps_data_path = self.gf_storage_root
        cps_data_path = Path(self.cps_data_path)
        self.fiducial_model_path = Path(fiducial_model_path)
        # first check if GF.mseed file exists
        single_gf_path = Path(cps_data_path) / 'GF.mseed'
        if single_gf_path.exists():
            self.cps_data_folders = [cps_data_path]
        else:
            self.cps_data_folders = [f for f in Path(cps_data_path).iterdir() if f.is_dir()]
        if not self.cps_data_folders:
            raise FileNotFoundError(f"No CPS data folders found in {cps_data_path}")
        self.num_models = len(self.cps_data_folders)

    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        seed = kwargs.get('seed', None)
        use_fiducial = kwargs.pop('use_fiducial', False)
        if seed is not None:
            np.random.seed(seed)
        
        if use_fiducial:
            cps_data_folder = self.fiducial_model_path
        else:
            # randomly  choose a CPS data folder
            cps_data_folder = np.random.choice(self.cps_data_folders)
        if verbose:
            print(f"Using CPS data folder: {cps_data_folder}")
        return update_with_Gtensor(
            objstats, velocity_model, delta=delta, force_calc=False, verbose=verbose, gf_directory=cps_data_folder, return_gf=return_gf, filter_params=self.synthetics_processing['filter'], **kwargs)
