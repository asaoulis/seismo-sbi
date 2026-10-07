"""The inference pipeline: simulate, compress, and invert one event.

:class:`SBIPipeline` loads the parameters and the simulator, builds the compressors and
covariances, generates the training simulations and the test jobs, and runs the Gaussian
likelihood and SBI inversions; :class:`SingleEventPipeline` specialises it to one event with a
fixed receiver set.
"""

import logging
from pathlib import Path
import shutil
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
import time
from typing import List
from functools import partial

from .configuration import SBI_Configuration
from .configuration import  ModelParameters, SimulationParameters, PipelineParameters, DatasetGenerationParameters, TestJobs
from .types.results import InversionResult, InversionData, JobResult, InversionConfig, JobData
from .types.fixed_jobs import FixedEventJobs

from .compression.gaussian import GaussianCompressor, MultiPointGaussianCompressor, SecondOrderCompressor

from .noises.noise_model import build_noise_sampler, build_test_noise_samplers
from .noises.real_noise import RealNoiseSampler
from .noises.diagonal_covariances import ScalarEmpiricalCovariance, DiagonalEmpiricalCovariance
from .noises.toeplitz_covariances import (
    BlockDiagonalEmpiricalCovariance,
    BlockDiagonalFilteredCovariance,
    BlockDiagonalKolbCovariance,
)
from .noises.theory_block_covariance import TheoryBlockDiagonalEmpiricalCovariance
from .noises.covariance_estimator import build_cov_sigma2_dict

from .inversion.sbi_inference import SBI_Inference
from .inversion import likelihood as likelihood
from .inversion.least_squares import IterativeLeastSquaresSolver

from .scalers import FlexibleScaler
from .datasets.dataset_compressor import DatasetCompressor

from seismo_sbi.simulators.simulation_io import SimulationDataLoader
from seismo_sbi.priors.parameter_sampler import ParameterSampler
from seismo_sbi.sbi.datasets.dataset_generator import DatasetGenerator

from .data_manager import DataManager
from .simulator_wrapper import GeneralSimulatorWrapper
from .job_runners import convert_lists_to_arrays

from seismo_sbi.utils.seismograms import compute_data_vector_length
from seismo_sbi.utils.errors import InvalidConfiguration

logger = logging.getLogger(__name__)


def likelihood_covariance(option, compressor_covariance, data_vector_length):
    """The noise covariance the Gaussian likelihood samples with: ``'empirical'`` copies the
    compressor's, a number is a white-noise standard deviation in the data units."""
    if option == 'empirical':
        return deepcopy(compressor_covariance)
    if isinstance(option, (int, float)):
        return ScalarEmpiricalCovariance(float(option), data_vector_length)
    raise InvalidConfiguration(
        f"inference.likelihood.covariance must be 'empirical' or a noise standard deviation, not {option!r}.")


class SBIPipeline:
    """The simulate-compress-invert method for one run, configured from an ``SBI_Configuration``.

    Holds the parameters, simulator wrapper, data manager, compressors, covariances and noise
    samplers that the steps share; outputs go under ``<output_directory>/<run_name>/<job_name>``
    and simulations under ``<output_directory>/sims/<run_name>/<job_name>``.
    """

    def __init__(self, pipeline_parameters : PipelineParameters, config_path : str = None):

        self.base_output_path = Path(pipeline_parameters.output_directory)
        self.sbi_run_name = pipeline_parameters.run_name
        self.job_name = pipeline_parameters.job_name
        self.num_parallel_jobs = pipeline_parameters.num_jobs

        simulations_output_path = self.base_output_path / f"./sims/{self.sbi_run_name}/{self.job_name}"
        simulations_output_path.mkdir(parents=True, exist_ok=True)
        self.simulations_output_path = str(simulations_output_path.resolve())

        # Where trained models live, one subdirectory per training run.
        self.models_output_path = self.base_output_path / self.sbi_run_name / self.job_name

        self.job_outputs_path = self.base_output_path / f"./plots/{self.sbi_run_name}/{self.job_name}"
        self.job_outputs_path.mkdir(parents=True, exist_ok=True)
        if config_path is not None:
            shutil.copy(config_path, self.job_outputs_path)
        
        self.test_jobs_paths = None

        self.num_dim = None
        self.parameters = None
        self.simulation_parameters = None
        #: The configured receiver time shifts, ``{station: samples}``, kept before the
        #: theory-covariance estimator zeroes them on the shared receivers.
        self.default_receiver_time_shifts = {}
        self.simulator_wrapper = None
        
        self.ground_truth_scaler = None

        self.data_vector_length = None
        self.trace_length = None
        self.compressor_keys = []
        self.compressors = {}
        self.compression_methods = None
        #: Seeds the SBI leg (covariance realisations, MLE chains, training set, NPE training); None leaves it unseeded.
        self.seed = None
        self.score_compression_data = None
        self.extra_gradients = None

        self.training_noise_sampler = None
        #: Whether the training noise is rescaled to each event's pre-event noise: False for white
        #: ``gaussian`` noise and for ``real_noise`` with ``rescale: false``.
        self.training_noise_follows_event = False
        self.test_noises = {}

        self.parameter_sampler = None
        self.data_cov_mat = None
        self.empirical_cov_mat = None

        self.data_manager = None

    def load_configuration(self, config):
        """Take the compression methods, the seed (``inference.sbi.seed``) and the seismic,
        model and dataset parameters of a parsed
        :class:`~seismo_sbi.sbi.configuration.SBI_Configuration`.
        """
        self.compression_methods = config.compression_methods
        self.seed = config.sbi_seed
        self.load_seismo_parameters(config.sim_parameters, config.model_parameters,
                                    config.dataset_parameters)

    def load_seismo_parameters(self,
                               simulation_parameters : SimulationParameters, 
                               model_parameters : ModelParameters,
                               dataset_parameters : DatasetGenerationParameters,
                               downsampled_length=None):
        """Set the parameters, samplers, simulator wrapper and data manager for this run."""

        self._load_base_pipeline_params(simulation_parameters, model_parameters, dataset_parameters, downsampled_length)

    def _load_base_pipeline_params(self, simulation_parameters, model_parameters, dataset_parameters, downsampled_length):
        self.parameters = model_parameters
        self.simulation_parameters = simulation_parameters

        sampling_method = dataset_parameters.sampling_method
        self.parameter_sampler = ParameterSampler.from_configuration(self.parameters, sampling_method)

        self.num_dim = model_parameters.parameter_to_vector('theta_fiducial').shape[0]

        data_loader = SimulationDataLoader(simulation_parameters.components, simulation_parameters.receivers)

        self.simulator_wrapper = GeneralSimulatorWrapper(simulation_parameters, self.parameters, data_loader, self.parameter_sampler)

        dataset_compressor = DatasetCompressor(data_loader, self.simulator_wrapper.simulation_save_callable, self.num_parallel_jobs, downsampled_length)
        data_length = compute_data_vector_length(simulation_parameters.seismogram_duration, simulation_parameters.sampling_rate) + 1
        self.data_manager = DataManager(data_loader, dataset_compressor, data_length)
        self.default_receiver_time_shifts = dict(simulation_parameters.receivers.receiver_time_shifts_map)

    def compute_data_vector_properties(self, test_jobs_paths, real_event_jobs_config):
        """Set ``data_vector_length`` and ``trace_length`` from the test and real-event jobs."""
        self.data_vector_length = self.data_manager.compute_data_vector_length(test_jobs_paths, real_event_jobs_config)
        num_traces = [component for receiver in self.simulation_parameters.receivers.receivers for component in receiver.components]
        self.trace_length = int(self.data_vector_length// len(num_traces))

    def load_compressors(self, compression_methods : dict, score_compression_data, prior=None, covariance_data=None, extra_gradients = None, freeze=False):

        """Build every compressor of ``compression_methods``, a list of ``(name, options)`` pairs where
        ``name`` is the final compressor name (e.g. ``'optimal_score_filtered_block'``).
        """
        for full_key, options in compression_methods:
            compressor, key = self._build_single_compressor(
                full_key,
                options,
                score_compression_data,
                prior,
                covariance_data,
                extra_gradients,
            )
            self.compressors[key] = compressor

        if freeze:
            self.compressor_keys = list(self.compressors.keys())

    def _build_single_compressor(
        self,
        full_key: str,
        options: dict,
        score_compression_data,
        prior,
        covariance_data,
        extra_gradients,
    ):
        """Build a single compressor instance for the given full_key.

        full_key is the compressor name used everywhere, e.g.
        'optimal_score_filtered_block' or 'theory_optimal_score'.
        options is the dict stored in SBI_Configuration.compression_methods for
        this key.

        Returns (compressor, full_key).
        """
        ctype = options.type

        if ctype == "optimal_score":
            cov_matrix_option = options.covariance
            cov_mat_config = options.path

            # reset and build covariance
            self.empirical_cov_mat = None
            if covariance_data is not None:
                cov_mat_config = covariance_data
                self.empirical_cov_mat = self.create_covariance_matrix(cov_matrix_option, cov_mat_config)
            else:
                sampler = RealNoiseSampler(
                    self.simulation_parameters,
                    cov_mat_config,
                    self.trace_length,
                )
                cov_data = sampler.draw_with_covariance(window_index=0).covariance_data
                self.empirical_cov_mat = self.create_covariance_matrix(cov_matrix_option, cov_data)

            compressor = GaussianCompressor(score_compression_data, self.empirical_cov_mat, prior=prior)

        elif ctype == "theory_optimal_score":
            theory_covariance = extra_gradients
            noise_level = options.noise_level
            data_cov_option = options.data_covariance
            if covariance_data is not None and noise_level is None:
                noise_level = build_cov_sigma2_dict(covariance_data)
            elif covariance_data is None and noise_level is None:
                # TEMP NOISE LEVEL
                logger.info("using temp noise level 1.0")
                noise_level = 1.0

            diag_regularisation_magnitude = options.diag_regularisation_magnitude
            cov_mat_options = (theory_covariance, diag_regularisation_magnitude)
            cov_mat_config = "theory_block"
            self.data_cov_mat = self.create_covariance_matrix(data_cov_option, noise_level)
            self.empirical_cov_mat = self.create_covariance_matrix(cov_mat_config, cov_mat_options)
            compressor = GaussianCompressor(score_compression_data, self.empirical_cov_mat, prior=prior)

        elif ctype == "multi_optimal_score":
            noise_level = options.noise_level
            cov_mat = np.diag(noise_level**2 * np.ones((self.data_vector_length)))
            compressor = MultiPointGaussianCompressor(score_compression_data, cov_mat)

        elif ctype == "second_order_score":
            noise_level = options.noise_level
            cov_mat = np.diag(noise_level**2 * np.ones((self.data_vector_length)))
            compressor = SecondOrderCompressor(score_compression_data, extra_gradients, cov_mat)

        else:
            raise NotImplementedError(f"Unknown compression type {ctype} for compressor '{full_key}'")

        return compressor, full_key

    def prepare_single_compressor(
        self,
        compressor_name: str,
        prior=None,
        covariance_data=None,
        dataset_details=None,
        compression_data_extras=None
    ):
        """Compute compression data and (re)build a single compressor.

        compressor_name must match one of the full keys from
        ``self.compression_methods`` (e.g. 'optimal_score_filtered_block').

        Returns (key, compressor, compression_data, extra_gradients).
        """

        # compression_methods is a list of (full_key, options)
        methods_dict = dict(self.compression_methods)
        if compressor_name not in methods_dict:
            raise KeyError(
                f"Compressor '{compressor_name}' not found in compression_methods. "
                f"Available: {list(methods_dict.keys())}"
            )

        options = methods_dict[compressor_name]

        # Compute compression data for this single compressor using the full key
        if compression_data_extras is None:
            compression_data, extra_gradients = self.compute_required_compression_data(
                [(compressor_name, options)],
                self.parameters,
            )
        else:
            compression_data, extra_gradients = compression_data_extras

        # Build just this compressor with the same full key
        compressor, key = self._build_single_compressor(
            compressor_name,
            options,
            compression_data,
            prior,
            covariance_data,
            extra_gradients,
        )

        self.compressors[key] = compressor
        if key not in self.compressor_keys:
            self.compressor_keys.append(key)
        return key, compressor, compression_data, extra_gradients

    def create_covariance_matrix(self, cov_matrix_option, cov_mat_config):

        stationwise_covariances = deepcopy(cov_mat_config)
        if cov_matrix_option == "empirical_block":
            cov_mat = BlockDiagonalEmpiricalCovariance(stationwise_covariances, self.simulation_parameters.receivers, self.trace_length, num_jobs=self.num_parallel_jobs)
        elif cov_matrix_option == "theory_block":
            covariance_blocks, diag_reg_magnitude = stationwise_covariances
            cov_mat = TheoryBlockDiagonalEmpiricalCovariance(covariance_blocks, self.data_cov_mat.covariance_matrix_arrays, self.simulation_parameters.receivers, self.trace_length, diag_regularisation=diag_reg_magnitude, num_jobs=self.num_parallel_jobs)
        elif cov_matrix_option == "filtered_block":
            noise_level = stationwise_covariances
            logger.info('Initialising filtered block covariance with noise level: %s', type(noise_level))
            cov_mat = BlockDiagonalFilteredCovariance(noise_level, self.simulation_parameters.processing['filter'], self.simulation_parameters.receivers, self.trace_length, num_jobs=self.num_parallel_jobs)
        elif cov_matrix_option == "kolb":
            noise_level = stationwise_covariances
            cov_mat = BlockDiagonalKolbCovariance(noise_level,receivers=self.simulation_parameters.receivers, data_vector_length=self.trace_length, num_jobs=self.num_parallel_jobs)
        elif cov_matrix_option == "empirical_diagonal":
            cov_mat = DiagonalEmpiricalCovariance(stationwise_covariances, self.simulation_parameters.receivers, self.trace_length)
        elif cov_matrix_option == "noise_level":
            noise_level = stationwise_covariances
            cov_mat = ScalarEmpiricalCovariance(noise_level, data_vector_length=self.data_vector_length)
        else:
            raise NotImplementedError(f"covariance matrix option {cov_matrix_option} not implemented")
        return cov_mat
    
    def load_test_noises(self, sbi_noise_model, test_noise_models):
        """Build the test-noise samplers (``test_noises``) and the training noise sampler from the
        training noise model ``sbi_noise_model`` (:class:`NoiseModelConfiguration`)."""
        covariances = dict(data_covariance=self.data_cov_mat, empirical_covariance=self.empirical_cov_mat)
        self.test_noises.update(build_test_noise_samplers(test_noise_models, sbi_noise_model, self.simulation_parameters,
                                                          self.trace_length, self.data_vector_length, **covariances))
        self.training_noise_follows_event = sbi_noise_model.follows_event
        self.training_noise_sampler = build_noise_sampler(sbi_noise_model, self.simulation_parameters,
                                                          self.trace_length, self.data_vector_length, **covariances)

    def rescale_training_noise(self, covariance_data):
        """Rescale the training noise to one event's pre-event noise ``covariance_data``
        ``{station: {component: autocovariance}}``, unless the noise model keeps its own level."""
        if self.training_noise_follows_event:
            self.training_noise_sampler.rescale_to(covariance_data)

    def compute_required_compression_data(self, compression_methods, model_parameters : ModelParameters, rerun_if_stencil_exists = True):
        """Run the derivative stencils the compression methods need; returns the compression data."""
        return self.data_manager.compute_required_compression_data(model_parameters, compression_methods, self.simulator_wrapper, self.simulation_parameters, seed=self.seed)
    
    def use_kernel_simulator_if_possible(self, score_compression_data, sampling_methods : dict):

        only_moment_tensor_variable = all([sampler == 'constant' for param, sampler in sampling_methods.items() if param != 'moment_tensor'])

        if only_moment_tensor_variable:
            self.simulator_wrapper.set_simulation_objects(
                    'kernel', self.simulation_parameters,
                    deepcopy(self.parameters), deepcopy(self.data_manager.data_loader), self.parameter_sampler,
                    score_compression_data=score_compression_data
                )

    def generate_simulation_data(self, dataset_parameters : DatasetGenerationParameters):
        """Simulate ``num_simulations`` training sources drawn from the current bounds with
        ``dataset_parameters.sampling_method``, written to ``<simulations_output_path>/train/sim_<i>.h5``."""
        num_simulations = dataset_parameters.num_simulations
        parameter_sampler = ParameterSampler.from_configuration(self.parameters, dataset_parameters.sampling_method)
        output_paths = [self.simulations_output_path + f'/train/sim_{i}.h5' for i in range(num_simulations)]

        dataset_generator = DatasetGenerator(self.simulator_wrapper.simulation_save_callable, self.num_parallel_jobs,
                                             seed=self.seed)
        dataset_generator.run_and_save_simulations(parameter_sampler.draw_simulation_inputs(num_simulations), output_paths)

    def simulate_test_jobs(self, dataset_parameters : DatasetGenerationParameters, test_jobs : TestJobs):
        """Simulate the random, fixed-mechanism and custom test events; returns their HDF5 paths."""
        parameter_sampler = ParameterSampler.from_configuration(self.parameters, dataset_parameters.sampling_method)
        dataset_generator = DatasetGenerator(self.simulator_wrapper.simulation_save_callable, self.num_parallel_jobs)

        random_event_paths = [self.simulations_output_path + f"/random_event_{i}.h5" for i in range(test_jobs.random_events)]
        dataset_generator.run_and_save_simulations(parameter_sampler.draw_simulation_inputs(test_jobs.random_events), random_event_paths)

        test_jobs_paths = []
        if len(test_jobs.fixed_events):
            fixed_job_simulation_args = self._generate_fixed_jobs_args(test_jobs.fixed_events)
            dataset_generator.run_parallel_simulations(fixed_job_simulation_args)
            fixed_jobs_sim_paths = [Path(sim_args[1]) for sim_args in fixed_job_simulation_args]
            test_jobs_paths += fixed_jobs_sim_paths
        
        custom_job_args = [(convert_lists_to_arrays(inputs), self.simulations_output_path + f"/{job_name}.h5") for job_name, inputs in test_jobs.custom_events.items()]
        dataset_generator.run_parallel_simulations(custom_job_args)

        custom_job_sim_paths = [Path(job_name) for _, job_name in custom_job_args]
        test_jobs_sim_paths = [Path(path) for path in random_event_paths]

        test_jobs_paths  +=  test_jobs_sim_paths + custom_job_sim_paths

        return test_jobs_paths
    
    def create_job_data(self, test_jobs_paths, real_event_jobs, *args, **kwargs):
        """``JobData`` for every test simulation under each test noise, and for each real event."""
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

    def plot_result(self, job_result, inversion_result, output = True):
        from ..plotting.results_plotting import SBIPipelinePlotter
        job_name, inversion_data, inversion_config = inversion_result
        train_noise, test_noise, method_name = inversion_config
        if not output:
            job_name = None

        plotter = SBIPipelinePlotter(self.job_outputs_path / f"{method_name}/{test_noise}", self.parameters)
        flattened_param_info = self.parameters.parameter_to_vector('information')[:6]
        plotter.initialise_posterior_plotter(inversion_data.data_scaler, flattened_param_info)

        if job_result is not None:
            compressed_dataset, compressed_x0, _ = job_result
            plotter.plot_compression(compressed_dataset, compressed_x0, job_name = job_name)
        plotter.plot_posterior(job_name, inversion_data, kde=True)

    def plot_comparisons(self, inversion_results, chain_consumer_config, savefig = True):

        from ..plotting.results_plotting import SBIPipelinePlotter
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
                    import traceback
                    traceback.print_exc()
                    logger.warning("ChainConsumer failed, skipping plotting of posterior comparisons: %s", e)

class SingleEventPipeline(SBIPipeline):
    """The pipeline for one event: an iterative least-squares MLE, then the Gaussian-likelihood
    and SBI inversions around it.
    """

    def __init__(self, pipeline_parameters : PipelineParameters, config_path : str = None):

        super().__init__(pipeline_parameters, config_path)
        self.least_squares_solver = None
        self.mcmc_chain_for_mle = None

    def load_seismo_parameters(self,
                               simulation_parameters : SimulationParameters, 
                               model_parameters : ModelParameters,
                               dataset_parameters : DatasetGenerationParameters,
                               downsampled_length=None):
        """As the base method, plus the iterative least-squares solver for the MLE."""
        
        self._load_base_pipeline_params(simulation_parameters, model_parameters, dataset_parameters, downsampled_length)

        self.least_squares_solver = IterativeLeastSquaresSolver(simulation_parameters, 
                                                                self.compression_methods,
                                                                model_parameters,
                                                                 self.data_manager, 
                                                                 self.simulator_wrapper,
                                                                 dataset_parameters.iterative_least_squares, 
                                                                 self.num_parallel_jobs)
    
        self.mcmc_chain_for_mle = dataset_parameters.iterative_least_squares.mcmc_chain_for_mle
    
    def use_kernel_simulator_if_possible(self, score_compression_data, sampling_methods : dict):

        only_moment_tensor_variable = all([sampler == 'constant' for param, sampler in sampling_methods.items() if param != 'moment_tensor'])

        if only_moment_tensor_variable:
            self.simulator_wrapper.set_simulation_objects(
                    'kernel', self.simulation_parameters,
                    deepcopy(self.parameters), deepcopy(self.data_manager.data_loader), self.parameter_sampler,
                    score_compression_data=score_compression_data
                )
            self.least_squares_solver.simulator = self.simulator_wrapper.simulator

    
    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details, do_plots = True):

        from ..plotting.results_plotting import SBIPipelinePlotter
        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)
        original_parameters = deepcopy(self.parameters)

        for single_job in job_data:
            self.parameters = deepcopy(original_parameters)
            if single_job.covariance is not None:
                self.rescale_training_noise(single_job.covariance)

            plotter = SBIPipelinePlotter(self.job_outputs_path / f"{single_job.noise_type}", self.parameters)

            theta0, dataset_details  = self.compute_theta0_and_update_dataset(param_names, original_dataset_details, single_job.theta0)

            for compressor_name in self.compressor_keys:

                start_time = time.time()
                logger.info("Starting on simulation: %s with compressor: %s", single_job.job_name, compressor_name)
                inversion_config = InversionConfig("", single_job.noise_type, compressor_name)
                if self.seed is not None:
                    np.random.seed(self.seed)
                    torch.manual_seed(self.seed)

                compression_data = self.find_mle_and_set_compressor(single_job.data_vector, single_job.covariance, single_job.prior, dataset_details, compressor_name=compressor_name)
                for _ in range(self.mcmc_chain_for_mle):
                    compression_data = self.find_mle_with_mcmc_and_set_compressor(likelihood_config, single_job, single_job.covariance, single_job.prior, mle_start=compression_data.theta_fiducial)

                inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(sbi_method, dataset_details, theta0, compression_data, single_job.prior, compressor_name=compressor_name)
                
                logger.info(f"Time taken for {single_job.job_name} with {compressor_name}: {time.time() - start_time}s")
                if do_plots:
                    plotter.plot_synthetic_misfits(single_job, self.simulation_parameters.receivers, compression_data.data_fiducial, self.parameters.get_parameter_values('source_location')[:2], covariance = self.empirical_cov_mat)

                inversion_result = InversionResult(single_job.job_name, inversion_data, inversion_config)

                yield job_result, inversion_result

            
                if likelihood_config["run"]:
                    logger.info('Starting likelihood inversions.')
                    start_time = time.time()
                    for result in self.run_single_gaussian_likelihood_inversion(
                        single_job, likelihood_config, compressor_name,
                        deepcopy(self.parameters), single_job.prior
                    ):
                        yield job_result, result[1]
                    logger.info(f"Time taken for likelihood inversions: {time.time() - start_time}s")

    def find_mle_with_mcmc_and_set_compressor(self, likelihood_config, single_job, covariance, prior, mle_start = None):
        MLE_likelihood_config = deepcopy(likelihood_config)
        use_best = MLE_likelihood_config.get('mle_use_best', False)
        logger.info('Finding MLE with MCMC, use_best: %s', use_best)
        if use_best:
            MLE_likelihood_config['walker_burn_in'] = 30
            MLE_likelihood_config['num_samples'] = self.num_parallel_jobs * 400
        else:
            MLE_likelihood_config['walker_burn_in'] = 300
            MLE_likelihood_config['num_samples'] = self.num_parallel_jobs * 400
        MLE_likelihood_config['ensemble'] = False
        MLE_likelihood_config['return_log_prob'] = bool(use_best)
        compressor_name = "theory_optimal_score"
        result = next(iter(self.run_single_gaussian_likelihood_inversion(
            single_job,
            MLE_likelihood_config,
            compressor_name,
            deepcopy(self.parameters),
            prior,
            mle_start=mle_start,
            seed=self.seed,
        )))
        if len(result) == 3:
            _, res, logps = result
        else:
            _, res = result
            logps = None

        samples = res.inversion_data.samples
        if use_best and logps is not None:
            best_idx = int(np.argmax(logps))
            logger.info('Best chi2 value: %s', -logps[best_idx])
            logger.info('Worst chi2 value: %s', -logps[np.argmin(logps)])
            mcmc_MLE = samples[best_idx]
        else:
            mcmc_MLE = np.mean(samples, axis=0)
        logger.info('MCMC MLE %s', mcmc_MLE)
        # compute chi2 of MLE
        
        self.parameters.theta_fiducial = self.parameters.vector_to_parameters(mcmc_MLE, 'theta_fiducial')
        _, _, compression_data, _ = self.prepare_single_compressor(
            compressor_name,
            prior=prior,
            covariance_data=covariance,
        )

        score_compression_data, extra_gradients = self.data_manager.compute_required_compression_data(
            self.parameters,
            *self.least_squares_solver.stencil_args,
            seed=self.seed,
        )
        compression_data = score_compression_data
        _, _, _, _ = self.prepare_single_compressor(
            compressor_name,
            prior=prior,
            covariance_data=covariance,
            compression_data_extras=(compression_data, extra_gradients)
        )
        chi2_mle = self.compressors[compressor_name].compute_misfit(single_job.data_vector)
        logger.info(f"chi^2 at MCMC MLE: {chi2_mle:.5f}")
        return compression_data

    def run_single_sbi_inversion(self, sbi_method, dataset_details, theta0, compression_data, prior, compressor_name: str = None):
        """Train an NPE on the compressed simulations and sample it at the data; returns
        ``(inversion_data, job_result, sbi_model)``.
        """
        
        param_names = self.parameters.names

        self.use_kernel_simulator_if_possible(compression_data, dataset_details.sampling_method)

        if dataset_details.use_fisher_to_constrain_bounds:
            dataset_details = self.use_fisher_to_constrain_bounds(compressor_name, dataset_details, compression_data)

        compressor = self.compressors[compressor_name]
        x_0 = compression_data.theta_fiducial
        
        logger.info('MLE %s', x_0)
        logger.info('theta0 %s', theta0)
        logger.info('bounds %s', self.parameters.bounds)
        self.ground_truth_scaler = FlexibleScaler(self.parameters)
        statistic_scaler = self.ground_truth_scaler
        x_0_scaled = statistic_scaler.transform(x_0.reshape(1,-1)).reshape(-1)

        self.generate_simulation_data(dataset_details)
        raw_compressed_dataset = self.data_manager.compress_dataset(
            compressor, param_names, self.simulations_output_path, self.training_noise_sampler,
            seed=self.seed
        )

        train_data = torch.Tensor(self.scale_dataset(raw_compressed_dataset, self.ground_truth_scaler, statistic_scaler))
        train_data, raw_compressed_dataset = self.clean_train_data(train_data, raw_compressed_dataset)
        sbi_model = SBI_Inference(sbi_method, self.num_dim)

        sbi_model.build_amortised_estimator(train_data)

        sample_results, _ = sbi_model.sample_posterior(x_0_scaled, num_samples=10000)
        # unscale the results
        sample_results = self.ground_truth_scaler.inverse_transform(sample_results)

        inversion_data = InversionData(theta0, sample_results, deepcopy(self.ground_truth_scaler), compression_data)
        job_result = JobResult(raw_compressed_dataset, x_0, deepcopy(self.ground_truth_scaler))
        return inversion_data, job_result, sbi_model

    def clean_train_data(self, train_data, raw_compressed_dataset, factor=100):
        # if mean relative error is too high, remove the row
        start_length = train_data.shape[0]
        truths = train_data[:, :self.num_dim]
        compressions = train_data[:, self.num_dim:]
        mean_relative_error = torch.mean(torch.abs(compressions - truths)/truths, dim=1)
        train_data = train_data[mean_relative_error < factor]
        raw_compressed_dataset = raw_compressed_dataset[mean_relative_error < factor]
        logger.info(f"Removed {start_length - train_data.shape[0]} rows due to high relative compression error.")
        # count number of rows removed
        return train_data, raw_compressed_dataset

    def find_mle_and_set_compressor(self, data_vector, covariance_data, prior, dataset_details, extra_gradients=None, compressor_name: str = None):
        """Find the MLE by iterative least squares and re-centre the compressor on it; returns the
        compression data at the MLE.
        """
        logger.info("Starting MLE")
        # choose a compressor name if not provided (needed to decide single- vs multi-step below)
        if compressor_name is None:
            if not self.compressor_keys:
                # default to first configured method name
                compressor_name = next(iter(dict(self.compression_methods).keys()))
            else:
                compressor_name = self.compressor_keys[0]
        only_moment_tensor_variable = all([sampler =='constant' for param, sampler in dataset_details.sampling_method.items() if param != 'moment_tensor'])
        # One linearised Gauss-Newton step is exact only for a linear problem: moment tensor
        # alone AND a constant covariance, which a theory-error covariance is not.
        theory_error_covariance = str(compressor_name).startswith("theory")
        single_least_squares_step = only_moment_tensor_variable and not theory_error_covariance
        # build / refresh this specific compressor with its own compression data
        key, compressor, compression_data, extra_gradients = self.prepare_single_compressor(
            compressor_name,
            prior=prior,
            covariance_data=covariance_data,
            dataset_details=dataset_details,
        )
        self.least_squares_solver.seed = self.seed
        compression_data, extra_gradients = self.least_squares_solver.solve_least_squares(
            data_vector,
            compressor,
            single_step=single_least_squares_step,
        )
        self.parameters.theta_fiducial = self.parameters.vector_to_parameters(compression_data.theta_fiducial, 'theta_fiducial')
        _, _, _, _ = self.prepare_single_compressor(
            compressor_name,
            prior=prior,
            covariance_data=covariance_data,
            dataset_details=dataset_details,
            compression_data_extras=(compression_data, extra_gradients)
        )

        return compression_data

    def use_fisher_to_constrain_bounds(self, compressor_name, dataset_details, compression_data):
        compressor = self.compressors[compressor_name]
        prior_covariance_matrix = (compressor.Fisher_mat_inverse)
        marginals = np.sqrt(np.diag(prior_covariance_matrix))
        num_sigmas = dataset_details.use_fisher_to_constrain_bounds
        new_bounds = np.vstack([compression_data.theta_fiducial - num_sigmas*marginals,                 
                                compression_data.theta_fiducial + num_sigmas*marginals])

                    
        new_bounds = np.sort(new_bounds, axis=0)
        lower_bound_inputs = self.parameters.vector_to_simulation_inputs(new_bounds[0], only_theta_fiducial=True)
        upper_bound_inputs = self.parameters.vector_to_simulation_inputs(new_bounds[1], only_theta_fiducial=True)
        # The Fisher box never reaches outside each parameter's configured bounds.
        lower_bound_inputs = {param: np.maximum(lower_bound_inputs[param], np.asarray(self.parameters.bounds[param][0], dtype=float))
                              for param in self.parameters.names}
        upper_bound_inputs = {param: np.minimum(upper_bound_inputs[param], np.asarray(self.parameters.bounds[param][1], dtype=float))
                              for param in self.parameters.names}

        for parameter in lower_bound_inputs.keys():
            if parameter == 'source_location':
                lower_bound_inputs[parameter][2] = max(lower_bound_inputs[parameter][2], 0)
            self.parameters.bounds[parameter] = np.vstack([lower_bound_inputs[parameter], upper_bound_inputs[parameter]])
            dataset_details.sampling_method[parameter] = 'uniform'

        return dataset_details

    def compute_theta0_and_update_dataset(self, param_names, original_dataset_details, theta0_dict):
        """``(theta0, dataset_details)``: the truth vector, or None when there is no truth, and a
        copy of the dataset settings.
        """
        if theta0_dict is not None:
            theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
        else:
            theta0 = None
        return theta0, deepcopy(original_dataset_details)

    def run_single_gaussian_likelihood_inversion(self, single_job, likelihood_config, compressor_name, parameters, prior=None, mle_start = None, seed = None):
        """Sample the Gaussian-likelihood posterior of one job with emcee; yields
        ``(None, inversion_result)``, with the log-probabilities when ``return_log_prob`` is set.
        """

        param_names = self.parameters.names
        ensemble = likelihood_config.get('ensemble', True)
        covariance = likelihood_config['covariance']

        covariance = likelihood_covariance(covariance, self.compressors[compressor_name].C, self.data_vector_length)
        walker_burn_in = likelihood_config['walker_burn_in']
        num_samples = likelihood_config['num_samples']
        move_size = likelihood_config.get('move_size')
        # Decoupled from num_jobs: with the kernel simulator each log-prob is a cheap
        # mat-vec, so the process fan-out costs more than it saves and can deadlock on HDF5.
        num_processes = likelihood_config.get('num_processes') or self.num_parallel_jobs
        nsamples_per_walker = num_samples//num_processes
        return_log_prob = bool(likelihood_config.get('return_log_prob', False))


        scaler = FlexibleScaler(parameters)

        inversion_config = InversionConfig("", single_job.noise_type, f'gaussian_likelihood_{compressor_name}')
        
        if single_job.theta0 is not None:
            theta0 = np.concatenate([[single_job.theta0[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
        else:
            theta0 = None
        if mle_start is not None:
            mle_start = scaler.transform(mle_start.reshape(1,-1)).reshape(-1)

        if ensemble:
            covariance_loss_callable = covariance.generic_loss_callable
        else:
            covariance_loss_callable = covariance.create_loss_callable(covariance.inverse_metadata, covariance.data_vector_length)

        simulator_likelihood = likelihood.GaussianLikelihoodEvaluator(single_job.data_vector, partial(self.simulator_wrapper.simulation_callable, use_fiducial=True) , scaler, loss_callable=covariance_loss_callable, prior=prior)

        if return_log_prob:
            samples_scaled, logps = likelihood.generate_samples(simulator_likelihood.log_probability, ensemble,
                                                        self.num_dim,
                                                        nsamples_per_walker=nsamples_per_walker, nwalkers=num_processes,
                                                        burn_in=walker_burn_in, num_processes=num_processes, theta0=theta0, move_size=move_size, mle_start = mle_start, return_log_prob=True, seed=seed)
        else:
            samples_scaled = likelihood.generate_samples(simulator_likelihood.log_probability, ensemble,
                                                        self.num_dim,
                                                        nsamples_per_walker=nsamples_per_walker, nwalkers=num_processes,
                                                        burn_in=walker_burn_in, num_processes=num_processes, theta0=theta0, move_size=move_size, mle_start = mle_start, seed=seed)
        logger.info("Finished MCMC chains.")
        samples = scaler.inverse_transform(samples_scaled)
        inversion_data = InversionData(theta0, samples, scaler)

        inversion_result = InversionResult(single_job.job_name, inversion_data, inversion_config)
        if return_log_prob:
            yield None, inversion_result, logps
        else:
            yield None, inversion_result
