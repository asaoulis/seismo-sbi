from functools import partial
from pathlib import Path
import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from sbi import utils as utils
from sbi import analysis as analysis
import itertools
from copy import deepcopy
import time
from typing import List


from src.instaseis_simulator.receivers import Receivers
from src.sbi.noises.real_noise import RealNoiseSampler
from src.sbi.configuration import SBI_Configuration

from src.instaseis_simulator.simulator import InstaseisSourceSimulator, FixedLocationKernelSimulator
from src.instaseis_simulator.dataloader import SimulationDataLoader
from src.instaseis_simulator.dataset_generator import DatasetGenerator

from src.sbi.compression.gaussian import GaussianCompressor, MachineLearningCompressor, MultiPointGaussianCompressor, SecondOrderCompressor

from src.sbi.dataset_compressor import DatasetCompressor
from src.sbi.compression.gaussian import ScoreCompressionData
from src.sbi.noises.covariance_estimation import EmpiricalCovarianceEstimator, ScalarEmpiricalCovariance, \
                                                        DiagonalEmpiricalCovariance, BlockDiagonalEmpiricalCovariance

from src.plotting.seismo_plots import plot_stacked_waveforms, MisfitsPlotting
from src.plotting.distributions import PosteriorPlotter, MomentTensorReparametrised
from src.sbi.inference import SBI_Inference
from src.sbi.fixed_jobs import FixedEventJobs
import src.sbi.likelihood as likelihood
from src.sbi.lsquares.least_squares import IterativeLeastSquaresSolver

from src.sbi.scalers import FlexibleScaler, GeneralScaler

from src.sbi.configuration import  ModelParameters, SimulationParameters, \
    PipelineParameters, DatasetGenerationParameters, TestJobs

from src.sbi.results import InversionResult, InversionData, JobResult, InversionConfig, JobData
from src.utils.errors import error_handling_wrapper



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

        self.simulation_callable = None
        self.simulation_save_callable = None

        self.generic_simulation_callable = None
        self.generic_simulation_save_callable = None
        
        self.dataset_compressor = None
        self.data_loader = None

        self.ground_truth_scaler = None

        self.data_vector_length = None
        self.trace_length = None
        self.compressors = {}

        self.training_noise_sampler = None
        self.test_noises = {}

        self.samplers = None
        self.empirical_cov_mat = None
        self.adaptive_covariance = None

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
        self.samplers = DatasetGenerator.create_samplers(self.parameters, sampling_method)

        self.num_dim = model_parameters.parameter_to_vector('theta_fiducial').shape[0]

        self.simulator = InstaseisSourceSimulator(simulation_parameters.syngine_address, 
                                    components= simulation_parameters.components, 
                                    receivers = simulation_parameters.receivers,
                                    seismogram_duration_in_s = simulation_parameters.seismogram_duration,
                                    synthetics_processing = simulation_parameters.processing)

        self.simulation_save_callable = self.simulator.execute_sim_and_save_outputs
        self.dataset_compressor, self.data_loader = self._load_compressor_and_data_loader(simulation_parameters.components, simulation_parameters.receivers, downsampled_length)

        parallelisable_simulator = likelihood.LightweightSimulator(deepcopy(self.parameters), deepcopy(self.data_loader), self.samplers)

        self.simulation_callable = partial(parallelisable_simulator.input_output_simulation, self.simulator )

        self.generic_simulation_callable = deepcopy(self.simulation_callable)
        self.generic_simulation_save_callable = deepcopy(self.simulation_save_callable)
    
    def _load_compressor_and_data_loader(self, components, receivers, downsampled_length):
        data_loader = SimulationDataLoader(components, receivers)
        dataset_compressor = DatasetCompressor(data_loader, self.simulation_save_callable, self.num_parallel_jobs, downsampled_length)
        return dataset_compressor, data_loader

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

        compression_method_details = [compression_method[0] for compression_method in compression_methods]
        score_compression_data = None
        hessian_gradients = None
        if "optimal_score" in compression_method_details or "multi_optimal_score" in compression_method_details:
            score_compression_data = self.compute_compression_data_from_stencil(model_parameters, rerun_if_stencil_exists)
        if "second_order_score" in compression_method_details:
            if score_compression_data is None:
                score_compression_data = self.compute_compression_data_from_stencil(model_parameters, rerun_if_stencil_exists)
            hessian_gradients = self.compute_hessian(score_compression_data, model_parameters, rerun_if_stencil_exists)
        return score_compression_data, hessian_gradients

    @error_handling_wrapper(num_attempts=3)
    def compute_compression_data_from_stencil(self, model_parameters : ModelParameters, rerun_if_stencil_exists = True):

        stencil_outputs_folder = Path(self.simulations_output_path) / "./stencil"
        
        stencil_hash = model_parameters.theta_fiducial_hash()
        score_compression_data_path =str((stencil_outputs_folder / f"./score_compresson_data-{stencil_hash}.npy").resolve())

        if not os.path.isfile(score_compression_data_path) or rerun_if_stencil_exists:
            score_compression_data = self.dataset_compressor.run_derivative_stencil_for_compression_data(
                                        model_parameters,
                                        stencil_outputs_folder,
                                        score_compression_data_path
                                    )
        else:
            score_compression_data = ScoreCompressionData(**np.load(score_compression_data_path, 
                                                                        allow_pickle=True).item())

        self.data_vector_length = score_compression_data.data_parameter_gradients.shape[1]
        num_traces = [component for receiver in self.simulation_parameters.receivers.receivers for component in receiver.components]
        self.trace_length = int(self.data_vector_length// len(num_traces))

        return score_compression_data
    
    def compute_hessian(self, score_compression_data, model_parameters : ModelParameters, rerun_if_stencil_exists = True):
        stencil_outputs_folder = Path(self.simulations_output_path) / "./stencil"


        stencil_hash = model_parameters.theta_fiducial_hash()
        hessian_gradients_path =str((stencil_outputs_folder / f"./hessian_gradients-{stencil_hash}.npy").resolve())

        if not os.path.isfile(hessian_gradients_path) or rerun_if_stencil_exists:

            hessian_gradients = self.dataset_compressor.run_hessian_stencil(model_parameters, score_compression_data, stencil_outputs_folder)
            np.save(hessian_gradients_path, hessian_gradients)
        else:
            hessian_gradients = np.load(hessian_gradients_path,  allow_pickle=True)

        return hessian_gradients
    
    def use_kernel_simulator_if_possible(self, score_compression_data, sampling_methods : dict):

        only_moment_tensor_variable = all([sampler == 'constant' for param, sampler in sampling_methods.items() if param != 'moment_tensor'])

        if only_moment_tensor_variable:
            sim = FixedLocationKernelSimulator(score_compression_data,
                                               self.trace_length,
                            components= self.simulation_parameters.components, 
                            receivers = self.simulation_parameters.receivers,
                            seismogram_duration_in_s = self.simulation_parameters.seismogram_duration,
                            synthetics_processing = self.simulation_parameters.processing)

                
            self.simulation_save_callable = sim.execute_sim_and_save_outputs
            
            parallelisable_simulator = likelihood.LightweightSimulator(deepcopy(self.parameters), deepcopy(self.data_loader), self.samplers)

            self.simulation_callable = partial(parallelisable_simulator.input_output_simulation, sim)

    def generate_simulation_data(self, dataset_parameters : DatasetGenerationParameters, simulation_indices = None, priors = (None, None)):

        if simulation_indices is None:
            num_simulations = dataset_parameters.num_simulations
            simulation_indices = (0, num_simulations)
        sampling_method = dataset_parameters.sampling_method

        dataset_generator = DatasetGenerator(self.simulation_save_callable, self.simulations_output_path + '/train', self.num_parallel_jobs)
        dataset_generator.run_and_save_simulations(self.parameters, sampling_method, simulation_indices, priors=priors)
        return dataset_generator

    def simulate_test_jobs(self, dataset_parameters : DatasetGenerationParameters, test_jobs : TestJobs):
        sampling_method = dataset_parameters.sampling_method
        dataset_generator = DatasetGenerator(self.simulation_save_callable, self.simulations_output_path + '/test', self.num_parallel_jobs)

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
    
    def compress_dataset(self, compressor, param_names, synthetic_noise_model_sampler = None, sorted_sims_paths = None, indices = slice(None)):
        if sorted_sims_paths is None:
            sorted_sims_paths = self.get_sorted_simulation_paths(indices)

        self.dataset_compressor.load_compressor_and_noise_model(compressor, synthetic_noise_model_sampler)
        raw_compressed_dataset = self.dataset_compressor.compress_dataset(sorted_sims_paths, param_names)

        return raw_compressed_dataset

    def get_sorted_simulation_paths(self, indices = slice(None)):
        sim_string = "sim_"
        sims_paths = (Path(self.simulations_output_path) / 'train').glob(f"{sim_string}*")
        sorted_sims_paths = list(sorted(sims_paths, key = lambda path: int(str(path)[str(path).find(sim_string) + len(sim_string):-3]) ))
        return sorted_sims_paths[indices]
    
    def scale_dataset(self, dataset, data_scaler, statistic_scaler):
        if dataset.shape[0] > 0:

            dataset = np.hstack([data_scaler.transform(dataset[:, :self.num_dim]),
                                statistic_scaler.transform(dataset[:, self.num_dim:])])
        return dataset
    
    def create_job_data(self, test_jobs_paths, real_event_jobs):

        job_data = {test_noise_name:{} for test_noise_name in self.test_noises.keys()}
        job_data = []

        for sim_path in test_jobs_paths:
            theta0 = self.data_loader.load_input_data(sim_path)
            D = self.data_loader.load_flattened_simulation_vector(sim_path)
            for test_noise_name, synthetic_noise_sampler in self.test_noises.items():
                noise = synthetic_noise_sampler(no_rescale=True)
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
    
    def run_compressions_and_inversions(self, job_data, sbi_method, likelihood_config):

        param_names = self.parameters.names

        for compressor_name, compressor in self.compressors.items():
            
            job_results[compressor_name] = {}
            inversion_results[compressor_name] = {}

            raw_compressed_dataset = self.compress_dataset(compressor, param_names, self.training_noise_sampler)
            if self.ground_truth_scaler is None:
                self.ground_truth_scaler = FlexibleScaler(self.parameters, raw_compressed_dataset[:, :self.num_dim])

            if compressor_name == "second_order_score":
                statistic_scaler = GeneralScaler(raw_compressed_dataset[:, self.num_dim:])
            else:
                statistic_scaler = self.ground_truth_scaler

            train_data = torch.Tensor(self.scale_dataset(raw_compressed_dataset, self.ground_truth_scaler, statistic_scaler))

            sbi_model = SBI_Inference(sbi_method, self.num_dim)
            sbi_model.build_amortised_estimator(train_data)

            ### Compression and inversion
            for test_noise_name, synthetic_noise_sampler in self.test_noises.items():
                inversion_results[compressor_name][test_noise_name] = {}
                train_compressed_dataset = self.compress_dataset(compressor, param_names, synthetic_noise_sampler)
                sim_names, compressed_job_data = self._compress_jobs(job_data, test_noise_name, param_names)

                test_data = torch.Tensor(self.scale_dataset(compressed_job_data, self.ground_truth_scaler, statistic_scaler))

                for sim_name, test_datapoint in zip(sim_names, test_data):
                    theta0, x0 = test_datapoint[:self.num_dim], test_datapoint[self.num_dim:]
                    sample_results, _ = sbi_model.sample_posterior(x0, num_samples=10000) 
                    inversion_results[compressor_name][test_noise_name][sim_name] = (sample_results, theta0) # throwing away probabilities atm

                job_results[compressor_name][test_noise_name] = (train_compressed_dataset)

        if likelihood_config["run"]:
            yield self.run_gaussian_likelihood_inversions(job_data, likelihood_config, )

        return job_data, job_results, inversion_results

    def _compress_jobs(self, job_data, test_noise_name, parameter_name_map):

        sim_names = []
        results = []

        for sim_name, (theta0_dict, D) in job_data[test_noise_name].items():
            fixed_keys = dict((param_type, param_names) if param_names != ["earthquake_magnitude"] else ("moment_tensor",["earthquake_magnitude"])\
                          for param_type, param_names in parameter_name_map.items())
            theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in fixed_keys.items()])
            
            result = np.concatenate([theta0, self.dataset_compressor.compressor.compress_data_vector(D)])
            sim_names.append(sim_name)
            results.append(result)

        return sim_names, np.stack(results)

    def run_gaussian_likelihood_inversions(self, job_data, likelihood_config, parameters_list):

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


        for single_job, parameters in zip(job_data, parameters_list):
            scaler = FlexibleScaler(parameters)

            sim_name, test_noise, D, theta0_dict, _ = single_job
            inversion_config = InversionConfig("", test_noise, 'gaussian_likelihood')
            
            if theta0_dict is not None:
                theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
                theta0_scaled = scaler.transform(theta0.reshape(1,-1)).reshape(-1)
            else:
                theta0_scaled = None

            covariance_loss_callable = covariance.create_loss_callable()
            simulator_likelihood = likelihood.GaussianLikelihoodEvaluator(D, self.simulation_callable, scaler, loss_callable=covariance_loss_callable)

            samples = likelihood.generate_samples(simulator_likelihood.log_probability, ensemble,
                                                            self.num_dim,
                                                            nsamples_per_walker=nsamples_per_walker, nwalkers=self.num_parallel_jobs, 
                                                            burn_in=walker_burn_in, num_processes=self.num_parallel_jobs, theta0=theta0_scaled, move_size=move_size)
            inversion_data = InversionData(theta0_scaled, samples, scaler)

            inversion_result = InversionResult(sim_name, inversion_data, inversion_config)
            yield None, inversion_result
        yield None

    
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

    def plot_comparisons(self, inversion_results, chain_consumer_config):

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
                    plotter.plot_chain_consumer("_".join(flattened_dict_keys), job_name, chain_consumer_dict, kde=True)
                except Exception as e:
                    print(e)
                    print("ChainConsumer failed, skipping plotting of posterior comparisons")

from sbi.inference import SNLE, SNPE
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi import utils as utils

class SingleEventPipeline(SBIPipeline):

    def __init__(self, pipeline_parameters : PipelineParameters, compression_methods, config_path : str = None):

        super().__init__(pipeline_parameters, config_path)
    
        self.compression_methods = compression_methods

    def load_seismo_parameters(self,
                               simulation_parameters : SimulationParameters, 
                               model_parameters : ModelParameters,
                               dataset_parameters : DatasetGenerationParameters,
                               downsampled_length=None):
        
        self._load_base_pipeline_params(simulation_parameters, model_parameters, dataset_parameters, downsampled_length)

        self.least_squares_solver = IterativeLeastSquaresSolver(simulation_parameters, 
                                                                model_parameters,
                                                                 self.data_loader, 
                                                                 self.simulator, 
                                                                 self.num_parallel_jobs)
    
    def use_kernel_simulator_if_possible(self, score_compression_data, sampling_methods : dict):

        only_moment_tensor_variable = all([sampler == 'constant' for param, sampler in sampling_methods.items() if param != 'moment_tensor'])

        if only_moment_tensor_variable:
            sim = FixedLocationKernelSimulator(score_compression_data,
                                               self.trace_length,
                            components= self.simulation_parameters.components, 
                            receivers = self.simulation_parameters.receivers,
                            seismogram_duration_in_s = self.simulation_parameters.seismogram_duration,
                            synthetics_processing = self.simulation_parameters.processing)

                
            self.simulation_save_callable = sim.execute_sim_and_save_outputs
            
            parallelisable_simulator = likelihood.LightweightSimulator(deepcopy(self.parameters), deepcopy(self.data_loader), self.samplers)

            self.simulation_callable = partial(parallelisable_simulator.input_output_simulation, sim)

            self.least_squares_solver.simulator = sim

    
    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details):

        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)

        for single_job in job_data:
            sim_name, test_noise, D, theta0_dict, covariance, priors = single_job
            if covariance is not None:
                self.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

            plotter = SBIPipelinePlotter(self.job_outputs_path / f"{test_noise}", self.parameters)
            # plotter.plot_all_stacked_waveforms(single_job, self.simulation_parameters.receivers)

            if theta0_dict is not None:
                theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
                dataset_details = self.set_known_parameters(deepcopy(original_dataset_details), theta0_dict)
            else:
                theta0 = None

            for compressor_name, compressor in self.compressors.items():

                start_time = time.time()
                
                inversion_config = InversionConfig("", test_noise, compressor_name)

                compression_data = self.find_mle_and_set_compressor(D, covariance, priors, dataset_details)

                inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(sbi_method, dataset_details, theta0, compression_data, priors)
                
                print(f"Time taken for {sim_name} with {compressor_name}: {time.time() - start_time}s", flush=True)

                # plotter.plot_synthetic_misfits(single_job, self.simulation_parameters.receivers, compression_data.data_fiducial, "", covariance = self.empirical_cov_mat)

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
        
        print('x0', x_0)
        print('bounds', self.parameters.bounds)
        self.ground_truth_scaler = FlexibleScaler(self.parameters)
        statistic_scaler = self.ground_truth_scaler
        x_0_scaled = statistic_scaler.transform(x_0.reshape(1,-1)).reshape(-1)
        if theta0 is not None:
            theta0_scaled = self.ground_truth_scaler.transform(theta0.reshape(1,-1)).reshape(-1)
        else:
            theta0_scaled = None

        dataset = self.generate_simulation_data(dataset_details, priors=priors)
        raw_compressed_dataset = self.compress_dataset(compressor, param_names, self.training_noise_sampler, indices = slice(0, dataset_details.num_simulations))
        dataset.clear_all_outputs()

        train_data = torch.Tensor(self.scale_dataset(raw_compressed_dataset, self.ground_truth_scaler, statistic_scaler))
        sbi_model = SBI_Inference(sbi_method, self.num_dim)
        sbi_model.build_amortised_estimator(train_data)

        sample_results, _ = sbi_model.sample_posterior(x_0_scaled, num_samples=10000)

        inversion_data = InversionData(theta0_scaled, sample_results, deepcopy(self.ground_truth_scaler), compression_data)
        job_result = JobResult(raw_compressed_dataset, x_0, deepcopy(self.ground_truth_scaler))
        return inversion_data, job_result, sbi_model

    def set_known_parameters(self, dataset_details, theta_dict):
        for param, sampler in dataset_details.sampling_method.items():
            order = SBI_Configuration.param_names_map[param]
            if sampler == 'uniform known':
                known_theta = theta_dict[param]
                self.parameters.nuisance[param] = [known_theta[param_name] for param_name in order]
                self.parameters.bounds[param] = self.parameters.nuisance[param]
                dataset_details.sampling_method[param] = 'constant'

                self.samplers = DatasetGenerator.create_samplers(self.parameters, dataset_details.sampling_method)
                
                parallelisable_simulator = likelihood.LightweightSimulator(deepcopy(self.parameters), deepcopy(self.data_loader), self.samplers)
                self.simulation_callable = partial(parallelisable_simulator.input_output_simulation, self.simulator )
                self.simulation_save_callable = self.generic_simulation_save_callable
                self.dataset_compressor.simulator = self.generic_simulation_save_callable
                self.least_squares_solver.simulator = self.simulator
        return dataset_details
                

    def find_mle_and_set_compressor(self, data_vector, covariance_data, priors, dataset_details):
        only_moment_tensor_variable = all([sampler =='constant' for param, sampler in dataset_details.sampling_method.items() if param != 'moment_tensor'])
        num_iterations = 10 if not only_moment_tensor_variable else 1
        compression_data = self.compute_compression_data_from_stencil(self.parameters, rerun_if_stencil_exists = True)
        self.load_compressors(self.compression_methods, score_compression_data=compression_data, priors=priors, covariance_data=covariance_data)
        compressor = self.compressors['optimal_score']
        if num_iterations > 0:
            compression_data = self.least_squares_solver.solve_least_squares(data_vector, compressor, iterations=num_iterations)
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

        simulator_likelihood = likelihood.GaussianLikelihoodEvaluator(D, self.simulation_callable, scaler, loss_callable=covariance_loss_callable, priors=priors)

        samples = likelihood.generate_samples(simulator_likelihood.log_probability, ensemble,
                                                        self.num_dim,
                                                        nsamples_per_walker=nsamples_per_walker, nwalkers=self.num_parallel_jobs, 
                                                        burn_in=walker_burn_in, num_processes=self.num_parallel_jobs, theta0=theta0_scaled, move_size=move_size)
        inversion_data = InversionData(theta0_scaled, samples, scaler)

        inversion_result = InversionResult(sim_name, inversion_data, inversion_config)
        yield None, inversion_result
        

    def perform_SNPE(self, dataset_details, param_names, test_noise, sim_name, D, theta0, statistic_scaler, prior, compressor_name, compressor, start_index):
        x_0 = compressor.compress_data_vector(D)

        inference = SNPE(prior)
        proposal = prior
        for iteration, num_simulations in enumerate([500, 500, 500]):
            end_index = start_index + num_simulations
            indices = (start_index, end_index) 
            if iteration == 0:
                self.generate_simulation_data(dataset_details, indices, priors=prior)
            else:
                theta = proposal.sample((num_simulations,))
                scaled_thetas = self.ground_truth_scaler.inverse_transform(theta)
                dataset_generator = DatasetGenerator(self.simulation_save_callable, self.simulations_output_path, self.num_parallel_jobs)
                dataset_generator.run_predefined_batch(scaled_thetas, indices, self.parameters)
    
            raw_compressed_dataset = self.compress_dataset(compressor, param_names, self.training_noise_sampler, indices = slice(start_index, end_index))
            if self.ground_truth_scaler is None:
                self.ground_truth_scaler = FlexibleScaler(self.parameters)
            if statistic_scaler is None:
                if compressor_name == "second_order_score":
                    statistic_scaler = GeneralScaler(raw_compressed_dataset[:, self.num_dim:])
                else:
                    statistic_scaler = self.ground_truth_scaler
                x_0_scaled = statistic_scaler.transform(x_0.reshape(1,-1)).reshape(-1)


            plotter = SBIPipelinePlotter(self.job_outputs_path / f"{compressor_name}/{test_noise}", self.parameters)
            plotter.initialise_posterior_plotter(self.ground_truth_scaler, self.parameters.parameter_to_vector('information'))

            plotter.plot_compression(raw_compressed_dataset)

            train_data = torch.Tensor(self.scale_dataset(raw_compressed_dataset, self.ground_truth_scaler, statistic_scaler))

            _ = inference.append_simulations(train_data[:, :self.num_dim], train_data[:, self.num_dim:], proposal).train()
            posterior = inference.build_posterior().set_default_x(x_0_scaled)

  
            plotter.plot_posterior(sim_name, theta0, self.parameters.parameter_to_vector('bounds', only_theta_fiducial=True), posterior.sample((2000,), x=x_0_scaled), kde=True)

            proposal = posterior

            start_index = end_index

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

                # plotter.plot_synthetic_misfits(single_job, self.simulation_parameters.receivers, compression_data.data_fiducial, "", covariance = self.empirical_cov_mat)

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
            theta0 = self.data_loader.load_input_data(sim_path)
            D = self.data_loader.load_flattened_simulation_vector(sim_path)
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

                        plotter.plot_synthetic_misfits(single_job, self.simulation_parameters.receivers, compression_data.data_fiducial, "", covariance = self.empirical_cov_mat)

                        inversion_result = InversionResult(sim_name+f'_{num_sims}_{repeat}', inversion_data, inversion_config)

                        yield job_result, inversion_result

                    
                        if likelihood_config["run"] and num_sims == 10000 and repeat == 0:
                            print('Starting likelihood inversions.')
                            start_time = time.time()
                            yield from self.run_single_gaussian_likelihood_inversion(single_job, likelihood_config, deepcopy(self.parameters), priors)
                            print(f"Time taken for likelihood inversions: {time.time() - start_time}s")

class SBIPipelinePlotter:

    def __init__(self, base_output_path, parameters : ModelParameters):

        self.base_output_path = Path(base_output_path)
        self.parameters = parameters
        self.num_dim = parameters.parameter_to_vector('theta_fiducial').shape[0]

        self.posterior_plotter = None
        self.reparametrised_plotter = None

    def initialise_posterior_plotter(self, data_scaler, parameters_info):

        self.posterior_plotter = PosteriorPlotter(data_scaler, parameters_info, self.parameters)

        if "moment_tensor" in self.parameters.names.keys():
            self.reparametrised_plotter = MomentTensorReparametrised(data_scaler, self.parameters)

    def plot_all_stacked_waveforms(self, single_job : JobData, receivers : Receivers):

        figure_path = self.base_output_path / "./seismograms"
        figure_path.mkdir(parents=True, exist_ok=True)

        plot_path = figure_path / f"./{single_job.job_name}.png"
        plot_stacked_waveforms(receivers.receivers, single_job.data_vector, figname=plot_path)
    
    def plot_synthetic_misfits(self, single_job : JobData, receivers : Receivers, synthetics : np.ndarray, event_location, covariance = None):
            
        figure_path = self.base_output_path / "./misfits"
        figure_path.mkdir(parents=True, exist_ok=True)

        misfits_plotter = MisfitsPlotting(receivers, 1, covariance)
        data_vector = single_job.data_vector

        plot_path = figure_path / f"./raw_{single_job.job_name}.png"
        misfits_plotter.raw_synthetic_misfits(data_vector, synthetics, figname=plot_path)

        plot_path = figure_path / f"./arrival_{single_job.job_name}.png"
        misfits_plotter.arrival_synthetic_misfits(data_vector, synthetics, (39.9267,  -29.9392, 20), figname=plot_path)

    
    def plot_posterior(self, test_name, inversion_data, kde=True):

        figure_path = self.base_output_path / "./inversions"
        figure_path.mkdir(parents=True, exist_ok=True)

        self.plot_chain_consumer(f"inversions", test_name, {"":inversion_data}, kde=kde)

        if "moment_tensor" in self.parameters.names.keys():
            plot_path = self.base_output_path / f"./beachballs/{test_name}" 
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            self.posterior_plotter.plot_beachball_samples(inversion_data, plot_path=plot_path)

    def plot_chain_consumer(self, base_figure_path, test_name, inversion_data_dict, kde=True):
        plot_path = self.base_output_path / f"./{base_figure_path}" / f"./{test_name}.png" 
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        self.posterior_plotter.plot_chain_consumer(inversion_data_dict, kde=kde, figsave=plot_path)

        if "moment_tensor" in self.parameters.names.keys():
            plot_path = self.base_output_path / f"./{base_figure_path}" / f"./nodal_params_{test_name}.png" 
            self.reparametrised_plotter.plot_chain_consumer(inversion_data_dict, kde=kde, inverse=True, figsave=plot_path)

    def plot_compression(self, raw_compressed_dataset, compressed_estimate = None, job_name=None):
        plotting_base_output_path = self.base_output_path
        if job_name is not None:
            plotting_base_output_path = plotting_base_output_path / 'compression'
        plotting_base_output_path.mkdir(exist_ok=True, parents=True)

        figname = (plotting_base_output_path / f"./{job_name}.png").resolve()
        self.posterior_plotter.plot_compression_errors(raw_compressed_dataset[:, :2*self.num_dim], compressed_estimate, figname=figname)
    

def convert_lists_to_arrays(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_lists_to_arrays(value)
        elif isinstance(value, list):
            if type(value[0]) is str:
                value = list(map(float, value))
            d[key] = np.array(value)
    return d
