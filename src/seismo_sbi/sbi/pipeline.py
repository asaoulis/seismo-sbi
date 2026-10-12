"""The shared state of an inversion run, held by :class:`SBIPipeline`.

:class:`SBIPipeline` loads the parameters and the simulator, builds the compressors and
covariances, and generates the training simulations and the test jobs. The pipelines that run
the Gaussian-likelihood and SBI inversions on it are in :mod:`seismo_sbi.sbi.event_pipelines`.
"""

import logging
import traceback
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .configuration import SBI_Configuration
from .configuration import  ModelParameters, SimulationParameters, PipelineParameters, DatasetGenerationParameters, TestJobs
from .types.fixed_jobs import FixedEventJobs

from .compression.compressors import build_compressor

from .noises.noise_model import build_noise_sampler, build_test_noise_samplers
from .noises.covariances import CovarianceLayout

from .datasets.dataset_compressor import DatasetCompressor

from seismo_sbi.simulators.simulation_io import SimulationDataLoader
from seismo_sbi.priors.parameter_sampler import ParameterSampler
from seismo_sbi.sbi.datasets.dataset_generator import DatasetGenerator

from .data_manager import DataManager
from .simulator_wrapper import GeneralSimulatorWrapper
from .job_runners import convert_lists_to_arrays

from seismo_sbi.utils.seismograms import compute_data_vector_length
from seismo_sbi.utils.errors import PipelineStateError

logger = logging.getLogger(__name__)


class SBIPipeline:
    """The simulate-compress-invert method for one run, configured from an ``SBI_Configuration``.

    Holds the parameters, simulator wrapper, data manager, compressors, covariances and noise
    samplers that the steps share. Outputs go under ``<output_directory>/<run_name>/<job_name>``
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
        self.compressors = {}
        self.compression_methods = None
        #: Seeds the SBI leg (covariance realisations, MLE chains, training set, NPE training).
        #: None leaves it unseeded.
        self.seed = None
        self.score_compression_data = None
        self.extra_gradients = None

        self.training_noise_sampler = None
        #: Whether the training noise is rescaled to each event's pre-event noise: False for white
        #: ``gaussian`` noise and for ``real_noise`` with ``rescale: false``.
        self.training_noise_follows_event = False
        self.test_noises = {}

        self.parameter_sampler = None

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

    def load_compressors(self, compression_methods : dict, score_compression_data, prior=None, covariance_data=None, extra_gradients = None):

        """Build every compressor of ``compression_methods``, a list of ``(name, options)`` pairs where
        ``name`` is the final compressor name (for example ``'optimal_score_filtered_block'``).
        """
        for full_key, options in compression_methods:
            self.compressors[full_key] = build_compressor(
                options, score_compression_data, self.simulation_parameters, self.covariance_layout(),
                covariance_data=covariance_data, extra_gradients=extra_gradients, prior=prior)

    def compression_data_at_fiducial(self, compressor_name):
        """``(compression_data, extra_gradients)`` of the configured compressor ``compressor_name`` at the
        current fiducial source (``parameters.theta_fiducial``)."""
        return self.compute_required_compression_data([(compressor_name, self.compressor_options(compressor_name))],
                                                      self.parameters)

    def compressor_options(self, compressor_name):
        """The options of the configured compressor ``compressor_name``."""
        methods_dict = dict(self.compression_methods)
        if compressor_name not in methods_dict:
            raise KeyError(
                f"Compressor '{compressor_name}' not found in compression_methods. "
                f"Available: {list(methods_dict.keys())}"
            )
        return methods_dict[compressor_name]

    def set_compressor(self, compressor_name, compression_data, extra_gradients=None, covariance_data=None, prior=None):
        """Build the configured compressor ``compressor_name`` around ``compression_data``, store it in
        ``compressors`` and return it. ``covariance_data`` is the job's noise, None for the
        configured one."""
        compressor = build_compressor(self.compressor_options(compressor_name), compression_data,
                                      self.simulation_parameters, self.covariance_layout(),
                                      covariance_data=covariance_data, extra_gradients=extra_gradients, prior=prior)
        self.compressors[compressor_name] = compressor
        return compressor

    def check_compressor_is_current(self, compressor_name):
        """Raise :class:`PipelineStateError` unless the compressor ``compressor_name`` is built and was built
        at the current fiducial source (``parameters.theta_fiducial``)."""
        if compressor_name not in self.compressors:
            raise PipelineStateError(f"The compressor {compressor_name!r} is not built: call load_compressors or "
                                     "find_mle_and_set_compressor first.")
        fiducial = self.parameters.parameter_to_vector('theta_fiducial', True)
        if not np.allclose(self.compressors[compressor_name].theta_fiducial, fiducial, rtol=1e-9, atol=0):
            raise PipelineStateError(f"The compressor {compressor_name!r} was built at a fiducial source other than "
                                     "the current one: rebuild it (find_mle_and_set_compressor) after changing "
                                     "theta_fiducial.")

    def covariance_layout(self):
        """The :class:`~seismo_sbi.sbi.noises.covariances.CovarianceLayout` of this run's data vector."""
        return CovarianceLayout(self.simulation_parameters.receivers, self.trace_length, self.data_vector_length,
                                (self.simulation_parameters.processing or {}).get('filter'), self.num_parallel_jobs)

    def load_test_noises(self, sbi_noise_model, test_noise_models):
        """Build the test-noise samplers (``test_noises``) and the training noise sampler from the
        training noise model ``sbi_noise_model`` (:class:`NoiseModelConfiguration`)."""
        covariances = self.noise_covariances()
        self.test_noises.update(build_test_noise_samplers(test_noise_models, sbi_noise_model, self.simulation_parameters,
                                                          self.trace_length, self.data_vector_length, **covariances))
        self.training_noise_follows_event = sbi_noise_model.follows_event
        self.training_noise_sampler = build_noise_sampler(sbi_noise_model, self.simulation_parameters,
                                                          self.trace_length, self.data_vector_length, **covariances)

    def noise_covariances(self):
        """``dict(data_covariance=, empirical_covariance=)`` for the noise models drawn from a compressor's
        covariance: the data covariance of the last built theory compressor and the covariance of the last
        built compressor (None before any is built)."""
        built = [self.compressors[name].C for name, _ in self.compression_methods or [] if name in self.compressors]
        data_covariances = [covariance.data_covariance for covariance in built if hasattr(covariance, "data_covariance")]
        return dict(data_covariance=data_covariances[-1] if data_covariances else None,
                    empirical_covariance=built[-1] if built else None)

    def rescale_training_noise(self, covariance_data):
        """Rescale the training noise to one event's pre-event noise ``covariance_data``
        ``{station: {component: autocovariance}}``, unless the noise model keeps its own level."""
        if self.training_noise_follows_event:
            self.training_noise_sampler.rescale_to(covariance_data)

    def compute_required_compression_data(self, compression_methods, model_parameters : ModelParameters, rerun_if_stencil_exists = True):
        """Run the derivative stencils the compression methods need. Returns the compression
        data."""
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
        """Simulate the random, fixed-mechanism and custom test events. Returns their HDF5 paths."""
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
                    traceback.print_exc()
                    logger.warning("ChainConsumer failed, skipping plotting of posterior comparisons: %s", e)
