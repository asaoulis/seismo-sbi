from pathlib import Path
import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from sbi import utils as utils
from sbi import analysis as analysis
from copy import deepcopy
import time
from typing import List

from .configuration import SBI_Configuration
from .configuration import  ModelParameters, SimulationParameters, PipelineParameters, DatasetGenerationParameters, TestJobs
from .types.results import InversionResult, InversionData, JobResult, InversionConfig, JobData
from .types.fixed_jobs import FixedEventJobs

from .compression.gaussian import GaussianCompressor, MachineLearningCompressor, MultiPointGaussianCompressor, SecondOrderCompressor
from .compression.gaussian import ScoreCompressionData

from .noises.real_noise import RealNoiseSampler
from .noises.covariance_estimation import ScalarEmpiricalCovariance, \
                                                            DiagonalEmpiricalCovariance, \
                                                                BlockDiagonalEmpiricalCovariance

from .inference import SBI_Inference
from . import likelihood as likelihood
from .lsquares.least_squares import IterativeLeastSquaresSolver

from .scalers import FlexibleScaler
from .dataset_compressor import DatasetCompressor

from ..utils.errors import error_handling_wrapper
from ..plotting.results_plotting import SBIPipelinePlotter
from ..instaseis_simulator.dataloader import SimulationDataLoader
from ..instaseis_simulator.dataset_generator import DatasetGenerator

from .data_manager import DataManager
from .simulator_wrapper import GeneralSimulatorWrapper
from .utils import convert_lists_to_arrays

from ..instaseis_simulator.utils import compute_data_vector_length
    

class SBIPipeline:
    

    def __init__(self, pipeline_parameters : PipelineParameters, config_path : str = None):

        self.base_output_path = Path(pipeline_parameters.output_directory)
        self.sbi_run_name = pipeline_parameters.run_name
        self.job_name = pipeline_parameters.job_name
        self.num_parallel_jobs = pipeline_parameters.num_jobs

        simulations_output_path = self.base_output_path / f"./sims/{self.sbi_run_name}/{self.job_name}"
        simulations_output_path.mkdir(parents=True, exist_ok=True)
        self.simulations_output_path = str(simulations_output_path.resolve())

        self.job_outputs_path = self.base_output_path / f"./plots/{self.sbi_run_name}/{self.job_name}"
        self.job_outputs_path.mkdir(parents=True, exist_ok=True)
        if config_path is not None:
            shutil.copy(config_path, self.job_outputs_path)
        
        self.test_jobs_paths = None

        self.num_dim = None
        self.simulation_parameters = None
        self.simulator_wrapper = None
        
        self.ground_truth_scaler = None

        self.data_vector_length = None
        self.trace_length = None
        self.compressors = {}
        self.compression_methods = None

        self.training_noise_sampler = None
        self.test_noises = {}

        self.dataset_generation_samplers = None
        self.empirical_cov_mat = None
        self.adaptive_covariance = None

        self.data_manager = None

    def load_seismo_parameters(self,
                               simulation_parameters : SimulationParameters, 
                               model_parameters : ModelParameters,
                               dataset_parameters : DatasetGenerationParameters,
                               downsampled_length=None):

        self._load_base_pipeline_params(simulation_parameters, model_parameters, dataset_parameters, downsampled_length)

    def _load_base_pipeline_params(self, simulation_parameters, model_parameters, dataset_parameters, downsampled_length):
        self.parameters = model_parameters
        self.simulation_parameters = simulation_parameters

        sampling_method = dataset_parameters.sampling_method
        self.dataset_generation_samplers = DatasetGenerator.create_samplers(self.parameters, sampling_method)

        self.num_dim = model_parameters.parameter_to_vector('theta_fiducial').shape[0]

        data_loader = SimulationDataLoader(simulation_parameters.components, simulation_parameters.receivers)

        self.simulator_wrapper = GeneralSimulatorWrapper(simulation_parameters, self.parameters, data_loader, self.dataset_generation_samplers)

        dataset_compressor = DatasetCompressor(data_loader, self.simulator_wrapper.simulation_save_callable, self.num_parallel_jobs, downsampled_length)
        data_length = compute_data_vector_length(simulation_parameters.seismogram_duration, simulation_parameters.sampling_rate) + 1
        self.data_manager = DataManager(data_loader, dataset_compressor, data_length)

    def compute_data_vector_properties(self, test_jobs_paths, real_event_jobs_config):
        self.data_vector_length = self.data_manager.compute_data_vector_length(test_jobs_paths, real_event_jobs_config)
        num_traces = [component for receiver in self.simulation_parameters.receivers.receivers for component in receiver.components]
        self.trace_length = int(self.data_vector_length// len(num_traces))


    def load_compressors(self, compression_methods : dict, score_compression_data, priors=(None,None), covariance_data=None, hessian_gradients = None):

        for compression_type, options in compression_methods:
            if compression_type == "optimal_score":
                cov_matrix_option, cov_mat_config = list(options.items())[0]
                
                if covariance_data is not None:
                    cov_mat_config = covariance_data
                    self.empirical_cov_mat = self.create_covariance_matrix(cov_matrix_option, cov_mat_config)
                else:
                    sampler = RealNoiseSampler(self.simulation_parameters,
                                                                cov_mat_config,
                                                                self.trace_length,)
                    _, cov_data = sampler(noise_index=0, no_rescale=True)
                    self.empirical_cov_mat = self.create_covariance_matrix(cov_matrix_option, cov_data)
                compressor = GaussianCompressor(score_compression_data, self.empirical_cov_mat, prior=priors)

            elif compression_type == "multi_optimal_score":
                noise_level = options["noise_level"]
                cov_mat = np.diag(noise_level**2 *np.ones((self.data_vector_length)))
                compressor = MultiPointGaussianCompressor(score_compression_data, cov_mat)
            elif compression_type == "second_order_score":
                noise_level = options["noise_level"]
                cov_mat = np.diag(noise_level**2 *np.ones((self.data_vector_length)))
                compressor = SecondOrderCompressor(score_compression_data, hessian_gradients, cov_mat)
            elif compression_type == "ml_compressor":
                compressor = MachineLearningCompressor(**options) # TODO: IMPLEMENT THIS
            self.compressors[compression_type] = compressor

    @error_handling_wrapper(num_attempts=3)
    def create_covariance_matrix(self, cov_matrix_option, cov_mat_config):

        stationwise_covariances = cov_mat_config

        if cov_matrix_option == "empirical_block":
            cov_mat = BlockDiagonalEmpiricalCovariance(stationwise_covariances, self.simulation_parameters.receivers, self.trace_length, num_jobs=self.num_parallel_jobs)
        elif cov_matrix_option == "empirical_diagonal":
            cov_mat = DiagonalEmpiricalCovariance(stationwise_covariances, self.simulation_parameters.receivers, self.trace_length)
        elif cov_matrix_option == "noise_level":
            noise_level = cov_mat_config
            cov_mat = ScalarEmpiricalCovariance(noise_level)
        else:
            raise NotImplementedError(f"covariance matrix option {cov_matrix_option} not implemented")
        return cov_mat
    
    def load_test_noises(self, sbi_noise_model, test_noise_models):
        train_noise_level = sbi_noise_model['noise_level']

        for noise_type, noise_options in test_noise_models:
            if noise_type == "gaussian_noises":
                noise_factor = noise_options
                noise_callable =  self._build_lambda_noiselevel( noise_factor *train_noise_level)
                self.test_noises[f"{noise_type}_x{noise_factor}"] = noise_callable
            elif noise_type == "real_noise":
                noise_catalogue_path = noise_options
                self.test_noises[noise_type] = RealNoiseSampler(self.simulation_parameters,
                                                                noise_catalogue_path,
                                                                self.trace_length,)
            elif noise_type == "empirical_gaussian":
                noise_catalogue_path = noise_options
                self.test_noises[noise_type] = self.empirical_cov_mat.create_sampler()
            
        
        train_noise_type =  sbi_noise_model['type']
        
        if train_noise_type == 'gaussian':
            train_noise_level = sbi_noise_model['noise_level']
            self.training_noise_sampler = lambda : np.random.normal(0, train_noise_level *np.ones((self.data_vector_length)))
        elif train_noise_type == 'real_noise':
            noise_catalogue_path = sbi_noise_model['noise_catalogue_path']

            self.training_noise_sampler = RealNoiseSampler(self.simulation_parameters,
                                                           noise_catalogue_path,
                                                           self.trace_length)
        elif train_noise_type == 'empirical_gaussian':
            self.training_noise_sampler = self.empirical_cov_mat.create_sampler()


    def _build_lambda_noiselevel(self, noise_level):
        return lambda : np.random.normal(0, noise_level *np.ones((self.data_vector_length)))

    def compute_required_compression_data(self, compression_methods, model_parameters : ModelParameters, rerun_if_stencil_exists = True):

        self.compression_methods = compression_methods

        compression_method_details = [compression_method[0] for compression_method in compression_methods]
        score_compression_data = None
        hessian_gradients = None
        if "optimal_score" in compression_method_details or "multi_optimal_score" in compression_method_details:
            score_compression_data = self.data_manager.compute_compression_data_from_stencil(model_parameters)
        if "second_order_score" in compression_method_details:
            if score_compression_data is None:
                score_compression_data = self.data_manager.compute_compression_data_from_stencil(model_parameters)
            hessian_gradients = self.data_manager.compute_hessian(score_compression_data, model_parameters)
        return score_compression_data, hessian_gradients
    
    def use_kernel_simulator_if_possible(self, score_compression_data, sampling_methods : dict):

        only_moment_tensor_variable = all([sampler == 'constant' for param, sampler in sampling_methods.items() if param != 'moment_tensor'])

        if only_moment_tensor_variable:
            self.simulator_wrapper.set_simulation_objects(
                    ('kernel', score_compression_data), self.simulation_parameters, 
                    deepcopy(self.parameters), deepcopy(self.data_manager.data_loader), self.dataset_generation_samplers
                )

    def generate_simulation_data(self, dataset_parameters : DatasetGenerationParameters, simulation_indices = None, priors = (None, None)):

        if simulation_indices is None:
            num_simulations = dataset_parameters.num_simulations
            simulation_indices = (0, num_simulations)
        sampling_method = dataset_parameters.sampling_method

        dataset_generator = DatasetGenerator(self.simulator_wrapper.simulation_save_callable, self.simulations_output_path + '/train', self.num_parallel_jobs)
        dataset_generator.run_and_save_simulations(self.parameters, sampling_method, simulation_indices, priors=priors)
        return dataset_generator

    def simulate_test_jobs(self, dataset_parameters : DatasetGenerationParameters, test_jobs : TestJobs):
        sampling_method = dataset_parameters.sampling_method
        dataset_generator = DatasetGenerator(self.simulator_wrapper.simulation_save_callable, self.simulations_output_path + '/test', self.num_parallel_jobs)

        test_sample_namer = lambda num_samples: (self.simulations_output_path + f"/random_event_{i}.h5" for i in range(num_samples))
        dataset_generator.run_and_save_simulations(self.parameters, sampling_method, test_jobs.random_events, sample_namer=test_sample_namer)

        test_jobs_paths = []
        if len(test_jobs.fixed_events):
            fixed_job_simulation_args = self._generate_fixed_jobs_args(test_jobs.fixed_events)
            dataset_generator.run_parallel_simulations(fixed_job_simulation_args)
            fixed_jobs_sim_paths = [Path(sim_args[1]) for sim_args in fixed_job_simulation_args]
            test_jobs_paths += fixed_jobs_sim_paths
        
        custom_job_args = [(convert_lists_to_arrays(inputs), self.simulations_output_path + f"/{job_name}.h5") for job_name, inputs in test_jobs.custom_events.items()]
        dataset_generator.run_parallel_simulations(custom_job_args)

        custom_job_sim_paths = [Path(job_name) for _, job_name in custom_job_args]
        test_jobs_sim_paths = [Path(sim_args) for sim_args in test_sample_namer(test_jobs.random_events)]

        test_jobs_paths  +=  test_jobs_sim_paths + custom_job_sim_paths

        return test_jobs_paths
    
    def create_job_data(self, test_jobs_paths, real_event_jobs, *args, **kwargs):
        return self.data_manager.create_job_data(test_jobs_paths, real_event_jobs, self.test_noises, *args, **kwargs)
    
    def _generate_fixed_jobs_args(self, fixed_events_list):
        M_0_bounds = self.parameters.bounds["moment_tensor"]
        try:
            M_0_values = [M_0_bounds[1][1]/1.5]
        except TypeError:
            M_0_values = 10**np.linspace(np.log10(M_0_bounds[0]),np.log10(M_0_bounds[1]), 4)[1:-1]
        all_simulation_args = []
        for M_0 in M_0_values:
            fixed_jobs_creator = FixedEventJobs(self.parameters, M_0, self.simulations_output_path)
            all_simulation_args += fixed_jobs_creator.create_simulation_inputs(fixed_events_list)
        
        return all_simulation_args

    def scale_dataset(self, dataset, data_scaler, statistic_scaler):
        if dataset.shape[0] > 0:

            dataset = np.hstack([data_scaler.transform(dataset[:, :self.num_dim]),
                                statistic_scaler.transform(dataset[:, self.num_dim:])])
        return dataset
    
    def plot_results(self, job_results, inversion_results):

        bounds = self.parameters.parameter_to_vector('bounds', only_theta_fiducial=True)
        tqdm_progress_bar = tqdm(zip(inversion_results, job_results), "Plotting inversion results: ", total=len(inversion_results))
        for inversion_result, job_result in tqdm_progress_bar:
            self.plot_result(job_result, inversion_result, bounds)

    def plot_result(self, job_result, inversion_result, bounds):
        job_name, inversion_data, inversion_config = inversion_result
        train_noise, test_noise, method_name = inversion_config

        plotter = SBIPipelinePlotter(self.job_outputs_path / f"{method_name}/{test_noise}", self.parameters)
        flattened_param_info = self.parameters.parameter_to_vector('information')
        plotter.initialise_posterior_plotter(inversion_data.data_scaler, flattened_param_info)

        if job_result is not None:
            compressed_dataset, compressed_x0, _ = job_result
            plotter.plot_compression(compressed_dataset, compressed_x0, job_name = job_name)
        plotter.plot_posterior(job_name, inversion_data, kde=True)

    def plot_comparisons(self, inversion_results, chain_consumer_config, savefig = True):

        flattened_param_info = self.parameters.parameter_to_vector('information')

        plotter = SBIPipelinePlotter(self.job_outputs_path / "./comparisons", self.parameters)
        plotter.initialise_posterior_plotter(self.ground_truth_scaler, flattened_param_info)

        tqdm_progress_bar = tqdm(chain_consumer_config, "Plotting posterior comparisons: ", total=len(chain_consumer_config))
        hashed_results = {hash(inversion_results): inversion_results for inversion_results in inversion_results}

        # find unique job names
        event_names = [inversion_result.event_name for inversion_result in inversion_results]
        event_names = list(set(event_names))
        for job_name in event_names:
            for dict_keys in tqdm_progress_bar:
                try:
                    # find hash match in inversion results list
                    chain_consumer_dict = {f"{compressor}_{test_noise}": 
                                                hashed_results[hash((job_name, "", test_noise, compressor))].inversion_data
                                                        for compressor, test_noise in dict_keys}
                    
                    flattened_dict_keys = [item for sublist in dict_keys for item in sublist]
                    plotter.plot_chain_consumer("_".join(flattened_dict_keys), job_name, chain_consumer_dict, kde=True, savefig=savefig)
                except Exception as e:
                    print(e)
                    print("ChainConsumer failed, skipping plotting of posterior comparisons")

class SingleEventPipeline(SBIPipeline):

    def __init__(self, pipeline_parameters : PipelineParameters, config_path : str = None):

        super().__init__(pipeline_parameters, config_path)

    def load_seismo_parameters(self,
                               simulation_parameters : SimulationParameters, 
                               model_parameters : ModelParameters,
                               dataset_parameters : DatasetGenerationParameters,
                               downsampled_length=None):
        
        self._load_base_pipeline_params(simulation_parameters, model_parameters, dataset_parameters, downsampled_length)

        self.least_squares_solver = IterativeLeastSquaresSolver(simulation_parameters, 
                                                                model_parameters,
                                                                 self.data_manager.data_loader, 
                                                                 self.simulator_wrapper.simulator,
                                                                 dataset_parameters.iterative_least_squares, 
                                                                 self.num_parallel_jobs)
    
    def use_kernel_simulator_if_possible(self, score_compression_data, sampling_methods : dict):

        only_moment_tensor_variable = all([sampler == 'constant' for param, sampler in sampling_methods.items() if param != 'moment_tensor'])

        if only_moment_tensor_variable:
            self.simulator_wrapper.set_simulation_objects(
                    ('kernel', score_compression_data), self.simulation_parameters, 
                    deepcopy(self.parameters), deepcopy(self.data_manager.data_loader), self.dataset_generation_samplers
                )
            self.least_squares_solver.simulator = self.simulator_wrapper.simulator

    
    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details):

        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)

        for single_job in job_data:
            sim_name, test_noise, D, theta0_dict, covariance, priors = single_job
            if covariance is not None:
                self.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

            plotter = SBIPipelinePlotter(self.job_outputs_path / f"{test_noise}", self.parameters)

            theta0, dataset_details  = self.compute_theta0_and_update_dataset(param_names, original_dataset_details, theta0_dict)

            for compressor_name, compressor in self.compressors.items():

                start_time = time.time()
                
                inversion_config = InversionConfig("", test_noise, compressor_name)

                compression_data = self.find_mle_and_set_compressor(D, covariance, priors, dataset_details)

                inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(sbi_method, dataset_details, theta0, compression_data, priors)
                
                print(f"Time taken for {sim_name} with {compressor_name}: {time.time() - start_time}s", flush=True)

                plotter.plot_synthetic_misfits(single_job, self.simulation_parameters.receivers, compression_data.data_fiducial, self.parameters.parameter_to_vector('source_location')[:2], covariance = self.empirical_cov_mat)

                inversion_result = InversionResult(sim_name, inversion_data, inversion_config)

                yield job_result, inversion_result

            
            if likelihood_config["run"]:
                print('Starting likelihood inversions.')
                start_time = time.time()
                yield from self.run_single_gaussian_likelihood_inversion(single_job, likelihood_config, deepcopy(self.parameters), priors)
                print(f"Time taken for likelihood inversions: {time.time() - start_time}s")

    def run_single_sbi_inversion(self, sbi_method, dataset_details, theta0, compression_data, priors):
        
        param_names = self.parameters.names

        self.use_kernel_simulator_if_possible(compression_data, dataset_details.sampling_method)

        if dataset_details.use_fisher_to_constrain_bounds:
            dataset_details = self.use_fisher_to_constrain_bounds(dataset_details, compression_data)

        compressor = self.compressors['optimal_score']
        x_0 = compression_data.theta_fiducial
        
        print('MLE', x_0)
        print('bounds', self.parameters.bounds)
        self.ground_truth_scaler = FlexibleScaler(self.parameters)
        statistic_scaler = self.ground_truth_scaler
        x_0_scaled = statistic_scaler.transform(x_0.reshape(1,-1)).reshape(-1)
        if theta0 is not None:
            theta0_scaled = self.ground_truth_scaler.transform(theta0.reshape(1,-1)).reshape(-1)
        else:
            theta0_scaled = None

        dataset = self.generate_simulation_data(dataset_details, priors=priors)
        raw_compressed_dataset = self.data_manager.compress_dataset(
            compressor, param_names, self.simulations_output_path, self.training_noise_sampler
        )
        dataset.clear_all_outputs()

        train_data = torch.Tensor(self.scale_dataset(raw_compressed_dataset, self.ground_truth_scaler, statistic_scaler))
        train_data, raw_compressed_dataset = self.clean_train_data(train_data, raw_compressed_dataset)
        sbi_model = SBI_Inference(sbi_method, self.num_dim)

        sbi_model.build_amortised_estimator(train_data)

        sample_results, _ = sbi_model.sample_posterior(x_0_scaled, num_samples=10000)

        inversion_data = InversionData(theta0_scaled, sample_results, deepcopy(self.ground_truth_scaler), compression_data)
        job_result = JobResult(raw_compressed_dataset, x_0, deepcopy(self.ground_truth_scaler))
        return inversion_data, job_result, sbi_model

    def clean_train_data(self, train_data, raw_compressed_dataset):
        # if mean relative error is too high, remove the row
        start_length = train_data.shape[0]
        truths = train_data[:, :self.num_dim]
        compressions = train_data[:, self.num_dim:]
        mean_relative_error = torch.mean(torch.abs(compressions - truths)/truths, dim=1)
        train_data = train_data[mean_relative_error < 5]
        raw_compressed_dataset = raw_compressed_dataset[mean_relative_error < 3]
        print(f"Removed {start_length - train_data.shape[0]} rows due to high relative compression error.")
        # count number of rows removed
        return train_data, raw_compressed_dataset

    def set_known_parameters(self, dataset_details, theta_dict):
        for param, sampler in dataset_details.sampling_method.items():
            order = SBI_Configuration.param_names_map[param]
            if sampler == 'uniform known':
                known_theta = theta_dict[param]
                self.parameters.nuisance[param] = [known_theta[param_name] for param_name in order]
                self.parameters.bounds[param] = self.parameters.nuisance[param]
                dataset_details.sampling_method[param] = 'constant'

                self.dataset_generation_samplers = DatasetGenerator.create_samplers(self.parameters, dataset_details.sampling_method)
                
                self.simulator_wrapper.set_simulation_objects(
                    ('instaseis', None), self.simulation_parameters, 
                    deepcopy(self.parameters), deepcopy(self.data_manager.data_loader), self.dataset_generation_samplers
                )
                self.simulator_wrapper.simulation_save_callable = self.simulator_wrapper.generic_simulation_save_callable
                self.data_manager.dataset_compressor.simulator = self.simulator_wrapper.generic_simulation_save_callable
                self.least_squares_solver.simulator = self.simulator_wrapper.simulator
        return dataset_details
                

    def find_mle_and_set_compressor(self, data_vector, covariance_data, priors, dataset_details):
        only_moment_tensor_variable = all([sampler =='constant' for param, sampler in dataset_details.sampling_method.items() if param != 'moment_tensor'])
        single_least_squares_step =  only_moment_tensor_variable
        compression_data = self.data_manager.compute_compression_data_from_stencil(self.parameters)
        self.load_compressors(self.compression_methods, score_compression_data=compression_data, priors=priors, covariance_data=covariance_data)
        compressor = self.compressors['optimal_score']
        compression_data = self.least_squares_solver.solve_least_squares(data_vector, compressor, single_step=single_least_squares_step)
        self.load_compressors(self.compression_methods, score_compression_data=compression_data, priors=priors, covariance_data=covariance_data)

        return compression_data

    def use_fisher_to_constrain_bounds(self, dataset_details, compression_data):
        compressor = self.compressors['optimal_score']
        prior_covariance_matrix = (compressor.Fisher_mat_inverse)
        marginals = np.sqrt(np.diag(prior_covariance_matrix))
        num_sigmas = dataset_details.use_fisher_to_constrain_bounds
        new_bounds = np.vstack([compression_data.theta_fiducial - num_sigmas*marginals,                 
                                compression_data.theta_fiducial + num_sigmas*marginals])

                    
        new_bounds = np.sort(new_bounds, axis=0)
        lower_bound_inputs = self.parameters.vector_to_simulation_inputs(new_bounds[0], only_theta_fiducial=True)
        upper_bound_inputs = self.parameters.vector_to_simulation_inputs(new_bounds[1], only_theta_fiducial=True)
        for parameter in lower_bound_inputs.keys():
            if parameter == 'source_location':
                lower_bound_inputs[parameter][2] = max(lower_bound_inputs[parameter][2], 0)
            self.parameters.bounds[parameter] = np.vstack([lower_bound_inputs[parameter], upper_bound_inputs[parameter]])
            dataset_details.sampling_method[parameter] = 'uniform'

        return dataset_details

    def compute_theta0_and_update_dataset(self, param_names, original_dataset_details, theta0_dict):
        if theta0_dict is not None:
            theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
            dataset_details = self.set_known_parameters(deepcopy(original_dataset_details), theta0_dict)
        else:
            theta0 = None
            dataset_details = deepcopy(original_dataset_details)
        return theta0, dataset_details

    def run_single_gaussian_likelihood_inversion(self, single_job, likelihood_config, parameters, priors=(None,None)):

        param_names = self.parameters.names
        ensemble = likelihood_config.get('ensemble', True)
        covariance = likelihood_config['covariance']
        if covariance == 'empirical':
            if 'optimal_score' in self.compressors.keys():
                covariance = deepcopy(self.compressors['optimal_score'].C)
        elif isinstance(covariance, float):
            covariance = covariance **2
        walker_burn_in = likelihood_config['walker_burn_in']
        num_samples = likelihood_config['num_samples']
        move_size = likelihood_config.get('move_size')
        nsamples_per_walker = num_samples//self.num_parallel_jobs


        scaler = FlexibleScaler(parameters)

        sim_name, test_noise, D, theta0_dict, *unused_data = single_job
        inversion_config = InversionConfig("", test_noise, 'gaussian_likelihood')
        
        if theta0_dict is not None:
            theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
            theta0_scaled = scaler.transform(theta0.reshape(1,-1)).reshape(-1)
        else:
            theta0_scaled = None
        # multiprocessing and joblib use two different backends
        if ensemble:
            covariance_loss_callable = covariance.generic_loss_callable
        else:
            covariance_loss_callable = covariance.create_loss_callable(covariance.C_inverse, covariance.data_vector_length)

        simulator_likelihood = likelihood.GaussianLikelihoodEvaluator(D, self.simulator_wrapper.simulation_callable, scaler, loss_callable=covariance_loss_callable, priors=priors)

        samples = likelihood.generate_samples(simulator_likelihood.log_probability, ensemble,
                                                        self.num_dim,
                                                        nsamples_per_walker=nsamples_per_walker, nwalkers=self.num_parallel_jobs, 
                                                        burn_in=walker_burn_in, num_processes=self.num_parallel_jobs, theta0=theta0_scaled, move_size=move_size)
        inversion_data = InversionData(theta0_scaled, samples, scaler)

        inversion_result = InversionResult(sim_name, inversion_data, inversion_config)
        yield None, inversion_result

class MultiEventPipeline(SingleEventPipeline):
    def __init__(self, pipeline_parameters : PipelineParameters, compression_methods, config_path : str = None):
            
        super().__init__(pipeline_parameters, compression_methods, config_path)

    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details):

        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)
        compressed_dataset = None
        for i, single_job in enumerate(job_data):
            sim_name, test_noise, D, theta0_dict, covariance, priors = single_job
            if covariance is not None:
                self.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

            plotter = SBIPipelinePlotter(self.job_outputs_path / f"{test_noise}", self.parameters)

            if theta0_dict is not None:
                theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
                dataset_details = self.set_known_parameters(deepcopy(original_dataset_details), theta0_dict)
            else:
                theta0 = None

            for compressor_name, compressor in self.compressors.items():

                start_time = time.time()
                
                inversion_config = InversionConfig("", test_noise, compressor_name)

                if i == 0:
                    compression_data = self.find_mle_and_set_compressor(D, covariance, priors, dataset_details)
                    inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(sbi_method, dataset_details, theta0, compression_data, priors)
                    compressed_dataset = job_result.compressed_dataset
                else:
                    x_0 = compressor.compress_data_vector(D)
                    x_0_scaled = self.ground_truth_scaler.transform(x_0.reshape(1,-1)).reshape(-1)
                    # if np.abs(x_0_scaled - 0.5).max() > 0.7:
                    #     print('x_0 problem found', np.abs(x_0_scaled - 0.5).max(), i)
                    #     x_0_scaled = x_0_scaled * 0 + 0.5
                    sample_results, _ = sbi_model.sample_posterior(x_0_scaled, num_samples=10000)
                    theta0_scaled = self.ground_truth_scaler.transform(theta0.reshape(1,-1)).reshape(-1)

                    inversion_data = InversionData(theta0_scaled, sample_results, deepcopy(self.ground_truth_scaler), compression_data)
                    job_result = JobResult(compressed_dataset, x_0, deepcopy(self.ground_truth_scaler))

                    compression_data = ScoreCompressionData(x_0, D, compression_data.data_parameter_gradients, None)
                    
                print(f"Time taken for {sim_name} with {compressor_name}: {time.time() - start_time}s", flush=True)

                inversion_result = InversionResult(sim_name, inversion_data, inversion_config)

                yield job_result, inversion_result

            
                if likelihood_config["run"]:
                    print('Starting likelihood inversions.')
                    start_time = time.time()
                    yield from self.run_single_gaussian_likelihood_inversion(single_job, likelihood_config, deepcopy(self.parameters), priors)
                    print(f"Time taken for likelihood inversions: {time.time() - start_time}s")

    def create_job_data(self, test_jobs_paths, real_event_jobs):

        job_data = {test_noise_name:{} for test_noise_name in self.test_noises.keys()}
        job_data = []

        for i, sim_path in enumerate(test_jobs_paths):
            theta0 = self.data_manager.load_model_parameter_vector(sim_path)
            D = self.data_manager.load_simulation_vector(sim_path)
            for test_noise_name, synthetic_noise_sampler in self.test_noises.items():
                if i == 0:
                    noise = synthetic_noise_sampler(no_rescale=True)
                    if isinstance(noise, tuple):
                        noise, covariance_data = noise
                        self.test_noises[test_noise_name].set_adaptive_covariance_with_misc_data(covariance_data)
                    else:
                        covariance_data = None

                else:
                    noise = synthetic_noise_sampler(no_rescale=False)
                    if isinstance(noise, tuple):
                        noise, covariance_data = noise
                    else:
                        covariance_data = None

                job_data.append(
                    JobData(sim_path.stem, 
                            test_noise_name,
                            D + noise, 
                            theta0,
                            covariance=covariance_data)
                    )

        for real_event_name, real_event_data in real_event_jobs.items():
            if isinstance(real_event_data, str):
                real_event_path = real_event_data
                priors = (None, None)
            elif isinstance(real_event_data, dict):
                real_event_path = real_event_data['path']
                priors = tuple(real_event_data['priors'])
            self.data_loader.data_length = 901
            D = self.data_loader.load_flattened_simulation_vector(real_event_path)
            covariance_data = self.data_loader.load_misc_data(real_event_path)
            self.data_loader.data_length = None
            for test_noise_name in self.test_noises.keys():
                job_data.append(
                    JobData(real_event_name,
                            test_noise_name,
                            D, 
                            theta0=None,
                            covariance = covariance_data,
                            priors = priors)
                )

        return job_data


class VaryDatasetSizeEventPipeline(MultiEventPipeline):
    def __init__(self, pipeline_parameters : PipelineParameters, compression_methods, config_path : str = None):
            
        super().__init__(pipeline_parameters, compression_methods, config_path)

    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details):

        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)
        compressed_dataset = None
        for repeat in range(3):

            for num_sims in original_dataset_details.num_simulations:

                for i, single_job in enumerate(job_data):

                    sim_name, test_noise, D, theta0_dict, covariance, priors = single_job
                    if covariance is not None:
                        self.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

                    plotter = SBIPipelinePlotter(self.job_outputs_path / f"{test_noise}", self.parameters)

                    if theta0_dict is not None:
                        theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
                        dataset_details = self.set_known_parameters(deepcopy(original_dataset_details), theta0_dict)
                    else:
                        theta0 = None

                    for compressor_name, compressor in self.compressors.items():
                        
                        start_time = time.time()
                        
                        inversion_config = InversionConfig("", test_noise, compressor_name)
                        if i == 0:
                            dataset_details = dataset_details._replace(num_simulations=num_sims)
                            compression_data = self.find_mle_and_set_compressor(D, covariance, priors, dataset_details)
                            inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(sbi_method, dataset_details, theta0, compression_data, compressor_name)
                            compressed_dataset = job_result.compressed_dataset
                        else:
                            x_0 = compressor.compress_data_vector(D)
                            x_0_scaled = self.ground_truth_scaler.transform(x_0.reshape(1,-1)).reshape(-1)
                            if np.abs(x_0_scaled - 0.5).max() > 0.5:
                                print('x_0 problem found', np.abs(x_0_scaled - 0.5).max(), i)
                                x_0_scaled = np.clip(x_0_scaled, 0, 1.)
                            sample_results, _ = sbi_model.sample_posterior(x_0_scaled, num_samples=10000)
                            theta0_scaled = self.ground_truth_scaler.transform(theta0.reshape(1,-1)).reshape(-1)

                            inversion_data = InversionData(theta0_scaled, sample_results, deepcopy(self.ground_truth_scaler), compression_data)
                            job_result = JobResult(compressed_dataset, x_0, deepcopy(self.ground_truth_scaler))

                            compression_data = ScoreCompressionData(x_0, D, compression_data.data_parameter_gradients, None)
                            
                        print(f"Time taken for {sim_name} with {compressor_name}: {time.time() - start_time}s", flush=True)

                        # plotter.plot_synthetic_misfits(single_job, self.simulation_parameters.receivers, compression_data.data_fiducial, "", covariance = self.empirical_cov_mat)

                        inversion_result = InversionResult(sim_name+f'_{num_sims}_{repeat}', inversion_data, inversion_config)

                        yield job_result, inversion_result

                    
                        if likelihood_config["run"] and num_sims == 10000 and repeat == 0:
                            print('Starting likelihood inversions.')
                            start_time = time.time()
                            yield from self.run_single_gaussian_likelihood_inversion(single_job, likelihood_config, deepcopy(self.parameters), priors)
                            print(f"Time taken for likelihood inversions: {time.time() - start_time}s")

