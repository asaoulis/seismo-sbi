from pathlib import Path
import tempfile
from sbi import utils as utils
from sbi import analysis as analysis

from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
from seismo_sbi.sbi.dataset_compressor import DatasetCompressor
from seismo_sbi.sbi.configuration import  ModelParameters
from seismo_sbi.sbi.types.results import  JobData
from seismo_sbi.utils.errors import error_handling_wrapper


class DataManager:

    def __init__(self, data_loader : SimulationDataLoader, dataset_compressor : DatasetCompressor):
        self.data_loader = data_loader
        self.dataset_compressor = dataset_compressor

    def compress_dataset(self, compressor, param_names, simulations_output_path, synthetic_noise_model_sampler = None):
        sim_string = "sim_" # TODO: either remove this glob or make it a constant
        sims_paths = list((Path(simulations_output_path) / 'train').glob(f"{sim_string}*"))
        self.dataset_compressor.load_compressor_and_noise_model(compressor, synthetic_noise_model_sampler)
        raw_compressed_dataset = self.dataset_compressor.compress_dataset(sims_paths, param_names)

        return raw_compressed_dataset

    def create_job_data(self, test_jobs_paths, real_event_jobs, test_noises):

        job_data = []

        job_data += self.create_synthetic_job_data(test_jobs_paths, test_noises)

        job_data += self._create_job_data_from_real_events(real_event_jobs, test_noises)

        return job_data

    def create_synthetic_job_data(self, test_jobs_paths, test_noises):
        synthetic_jobs = []
        for sim_path in test_jobs_paths:
            theta0 = self.load_model_parameter_vector(sim_path)
            D = self.load_simulation_vector(sim_path)
            for test_noise_name, synthetic_noise_sampler in test_noises.items():
                noise = synthetic_noise_sampler(no_rescale=True)
                if isinstance(noise, tuple):
                    noise, covariance_data = noise
                else:
                    covariance_data = None
                synthetic_jobs.append(
                    JobData(sim_path.stem, 
                            test_noise_name,
                            D + noise, 
                            theta0,
                            covariance=covariance_data)
                    )
        return synthetic_jobs

    def _create_job_data_from_real_events(self, real_event_jobs, test_noises):
        real_jobs = []
        for real_event_name, real_event_data in real_event_jobs.items():
            if isinstance(real_event_data, str):
                real_event_path = real_event_data
                priors = (None, None)
            elif isinstance(real_event_data, dict):
                real_event_path = real_event_data['path']
                priors = tuple(real_event_data['priors'])
            self.data_loader.data_length = 901
            D = self.load_simulation_vector(real_event_path)
            covariance_data = self.load_noise_parametrisation_data(real_event_path)
            self.data_loader.data_length = None
            for test_noise_name in test_noises.keys():
                real_jobs.append(
                    JobData(real_event_name,
                            test_noise_name,
                            D, 
                            theta0=None,
                            covariance = covariance_data,
                            priors = priors)
                )
        return real_jobs

    @error_handling_wrapper(num_attempts=3)
    def compute_compression_data_from_stencil(self, model_parameters : ModelParameters):

        with tempfile.TemporaryDirectory() as stencil_outputs_folder:
            
            score_compression_data = self.dataset_compressor.run_derivative_stencil_for_compression_data(
                                            model_parameters,
                                            Path(stencil_outputs_folder)
                                        )
        return score_compression_data
    
    @error_handling_wrapper(num_attempts=3)
    def compute_hessian(self, score_compression_data, model_parameters : ModelParameters):

        with tempfile.TemporaryDirectory() as stencil_outputs_folder:
            hessian_gradients = self.dataset_compressor.run_hessian_stencil(model_parameters, score_compression_data, Path(stencil_outputs_folder))

        return hessian_gradients

    def compute_data_vector_length(self, test_jobs_paths, real_event_jobs_config):
        if len(test_jobs_paths) > 0:
            data_vector_length  = self.load_simulation_vector(test_jobs_paths[0]).shape[0]
        else:
            real_event_data = list(real_event_jobs_config.values())[0]
            if isinstance(real_event_data, str):
                real_event_path = real_event_data
            else:
                real_event_path = real_event_data['path']
            data_vector_length = self.load_simulation_vector(real_event_path).shape[0]

        return data_vector_length
    
    def load_simulation_vector(self, sim_path):
        return self.data_loader.load_flattened_simulation_vector(sim_path)

    def load_model_parameter_vector(self, sim_path):
        return self.data_loader.load_input_data(sim_path)

    def load_noise_parametrisation_data(self, sim_path):
        return self.data_loader.load_misc_data(sim_path)
