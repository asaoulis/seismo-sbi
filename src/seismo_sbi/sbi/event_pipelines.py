"""Inversion pipelines built on :class:`SBIPipeline`, one class per ``inference.sbi.pipeline`` value.

:class:`SingleEventPipeline` finds an event's maximum-likelihood source by iterative least squares,
then runs the Gaussian-likelihood and SBI inversions around it. :class:`MultiEventPipeline` builds the
compressor and trains once, on the first job, then runs the Gaussian-likelihood inversion for every
job. :class:`VaryDatasetSizeEventPipeline` repeats the SBI inversion over training-set sizes, and
:class:`MLEEstimatePipeline` runs only the maximum-likelihood search for each job.
"""

import logging
import time
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import List

import numpy as np
import torch

from .compression.gaussian import ScoreCompressionData
from .configuration import ModelParameters, SimulationParameters, PipelineParameters, DatasetGenerationParameters
from .types.results import InversionResult, InversionData, JobResult, InversionConfig, JobData
from .noises.diagonal_covariances import ScalarEmpiricalCovariance
from .inversion.sbi_inference import SBI_Inference
from .inversion import likelihood as likelihood
from .inversion.least_squares import IterativeLeastSquaresSolver
from .pipeline import SBIPipeline
from .scalers import FlexibleScaler, ZeroOneScaler

from seismo_sbi.utils.errors import InvalidConfiguration, PipelineStateError

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


def observed_summary_and_scaler(compressor, compression_data, data_vector, training_summaries, parameter_scaler):
    """``(x_0, statistic_scaler)``: the observation's summary and the scaler of every summary.

    A compressor whose summary estimates the parameters gives the fiducial source the MLE search
    converged to, scaled like the parameters (``parameter_scaler``). Any other compresses the
    observation ``data_vector`` and scales each summary component to [0, 1] over the
    ``training_summaries`` ``(n_simulations, n_summary)``.
    """
    if compressor.summary_is_parameter_estimate:
        return compression_data.theta_fiducial, parameter_scaler
    if data_vector is None:
        raise PipelineStateError("This compressor's summary is not a parameter estimate, so the observed "
                                 "summary is the compressed observation: pass data_vector.")
    lower, upper = training_summaries.min(axis=0), training_summaries.max(axis=0)
    upper = np.where(upper > lower, upper, lower + 1.0)
    return compressor.compress_data_vector(data_vector), ZeroOneScaler(np.vstack([lower, upper]))


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

            for compressor_name, _ in self.compression_methods:

                start_time = time.time()
                logger.info("Starting on simulation: %s with compressor: %s", single_job.job_name, compressor_name)
                inversion_config = InversionConfig("", single_job.noise_type, compressor_name)
                if self.seed is not None:
                    np.random.seed(self.seed)
                    torch.manual_seed(self.seed)

                compression_data = self.find_mle_and_set_compressor(single_job.data_vector, single_job.covariance, single_job.prior, dataset_details, compressor_name=compressor_name)
                for _ in range(self.mcmc_chain_for_mle):
                    compression_data = self.find_mle_with_mcmc_and_set_compressor(likelihood_config, single_job, single_job.covariance, single_job.prior, mle_start=compression_data.theta_fiducial,
                                                                                compressor_name=compressor_name)

                inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(sbi_method, dataset_details, theta0, compression_data, single_job.prior, compressor_name=compressor_name,
                                                                                         data_vector=single_job.data_vector)
                
                logger.info(f"Time taken for {single_job.job_name} with {compressor_name}: {time.time() - start_time}s")
                if do_plots:
                    plotter.plot_synthetic_misfits(single_job, self.simulation_parameters.receivers, compression_data.data_fiducial, self.parameters.get_parameter_values('source_location')[:3], self.simulation_parameters.sampling_rate, covariance = self.compressors[compressor_name].C)

                inversion_result = InversionResult(single_job.job_name, inversion_data, inversion_config)

                yield job_result, inversion_result

            
                if likelihood_config.run:
                    logger.info('Starting likelihood inversions.')
                    start_time = time.time()
                    for result in self.run_single_gaussian_likelihood_inversion(
                        single_job, likelihood_config, compressor_name,
                        deepcopy(self.parameters), single_job.prior
                    ):
                        yield job_result, result[1]
                    logger.info(f"Time taken for likelihood inversions: {time.time() - start_time}s")

    def find_mle_with_mcmc_and_set_compressor(self, likelihood_config, single_job, covariance, prior, mle_start = None,
                                              compressor_name="theory_optimal_score"):
        """Refine the MLE with a Gaussian-likelihood chain of the compressor ``compressor_name`` and re-centre
        the compressor on it; returns the compression data there."""
        use_best = likelihood_config.mle_use_best
        logger.info('Finding MLE with MCMC, use_best: %s', use_best)
        MLE_likelihood_config = replace(likelihood_config, walker_burn_in=30 if use_best else 300,
                                        num_samples=self.num_parallel_jobs * 400, ensemble=False)
        result = next(iter(self.run_single_gaussian_likelihood_inversion(
            single_job,
            MLE_likelihood_config,
            compressor_name,
            deepcopy(self.parameters),
            prior,
            mle_start=mle_start,
            seed=self.seed,
            return_log_prob=bool(use_best),
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
        compression_data, extra_gradients = self.compression_data_at_fiducial(compressor_name)
        self.set_compressor(compressor_name, compression_data, extra_gradients, covariance, prior)
        chi2_mle = self.compressors[compressor_name].compute_misfit(single_job.data_vector)
        logger.info(f"chi^2 at MCMC MLE: {chi2_mle:.5f}")
        return compression_data

    def run_single_sbi_inversion(self, sbi_method, dataset_details, theta0, compression_data, prior, compressor_name: str = None,
                                 data_vector=None):
        """Train an NPE on the compressed simulations and sample it at the data; returns
        ``(inversion_data, job_result, sbi_model)``. ``data_vector`` is the observation, needed when
        the compressor's summary is not a parameter estimate.
        """
        
        param_names = self.parameters.names
        self.check_compressor_is_current(compressor_name)
        if self.training_noise_sampler is None:
            raise PipelineStateError("The SBI inversion adds training noise to every simulation, and no training "
                                     "noise sampler is loaded: call load_test_noises first.")

        self.use_kernel_simulator_if_possible(compression_data, dataset_details.sampling_method)

        if dataset_details.use_fisher_to_constrain_bounds:
            dataset_details = self.use_fisher_to_constrain_bounds(compressor_name, dataset_details, compression_data)

        compressor = self.compressors[compressor_name]
        logger.info('theta0 %s', theta0)
        logger.info('bounds %s', self.parameters.bounds)
        self.ground_truth_scaler = FlexibleScaler(self.parameters)

        self.generate_simulation_data(dataset_details)
        raw_compressed_dataset = self.data_manager.compress_dataset(
            compressor, param_names, self.simulations_output_path, self.training_noise_sampler,
            seed=self.seed
        )
        x_0, statistic_scaler = observed_summary_and_scaler(compressor, compression_data, data_vector,
                                                            raw_compressed_dataset[:, self.num_dim:],
                                                            self.ground_truth_scaler)
        logger.info('Observed summary %s', x_0)
        x_0_scaled = statistic_scaler.transform(x_0.reshape(1,-1)).reshape(-1)

        train_data = torch.Tensor(self.scale_dataset(raw_compressed_dataset, self.ground_truth_scaler, statistic_scaler))
        if compressor.summary_is_parameter_estimate:
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
        if compressor_name is None:
            compressor_name = self.compression_methods[0][0]
        only_moment_tensor_variable = all([sampler =='constant' for param, sampler in dataset_details.sampling_method.items() if param != 'moment_tensor'])
        # One linearised Gauss-Newton step is exact only for a linear problem: moment tensor
        # alone AND a constant covariance, which a theory-error covariance is not.
        theory_error_covariance = str(compressor_name).startswith("theory")
        single_least_squares_step = only_moment_tensor_variable and not theory_error_covariance
        compression_data, extra_gradients = self.compression_data_at_fiducial(compressor_name)
        compressor = self.set_compressor(compressor_name, compression_data, extra_gradients, covariance_data, prior)
        self.least_squares_solver.seed = self.seed
        compression_data, extra_gradients = self.least_squares_solver.solve_least_squares(
            data_vector,
            compressor,
            single_step=single_least_squares_step,
            compression_methods=[(compressor_name, self.compressor_options(compressor_name))],
        )
        self.parameters.theta_fiducial = self.parameters.vector_to_parameters(compression_data.theta_fiducial, 'theta_fiducial')
        self.set_compressor(compressor_name, compression_data, extra_gradients, covariance_data, prior)
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

    def run_single_gaussian_likelihood_inversion(self, single_job, likelihood_config, compressor_name, parameters, prior=None, mle_start = None, seed = None,
                                                 return_log_prob=False):
        """Sample the Gaussian-likelihood posterior of one job with emcee, as ``likelihood_config``
        (:class:`~seismo_sbi.sbi.configuration.LikelihoodConfiguration`) sets; yields
        ``(None, inversion_result)``, with the log-probabilities when ``return_log_prob`` is set.
        """

        param_names = self.parameters.names
        ensemble = likelihood_config.ensemble
        covariance = likelihood_config.covariance

        self.check_compressor_is_current(compressor_name)
        covariance = likelihood_covariance(covariance, self.compressors[compressor_name].C, self.data_vector_length)
        walker_burn_in = likelihood_config.walker_burn_in
        num_samples = likelihood_config.num_samples
        move_size = likelihood_config.move_size
        # Decoupled from num_jobs: with the kernel simulator each log-prob is a cheap
        # mat-vec, so the process fan-out costs more than it saves and can deadlock on HDF5.
        num_processes = likelihood_config.num_processes or self.num_parallel_jobs
        nsamples_per_walker = num_samples//num_processes


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


class MultiEventPipeline(SingleEventPipeline):
    """Build the compressor and train once on the first job, then invert every job with it."""

    def __init__(self, pipeline_parameters : PipelineParameters, config_path : str = None):
            
        super().__init__(pipeline_parameters, config_path)

    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details, do_plots = True):

        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)
        for i, single_job in enumerate(job_data):
            if single_job.covariance is not None:
                self.rescale_training_noise(single_job.covariance)

            if single_job.theta0 is not None:
                theta0 = np.concatenate([[single_job.theta0[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
                dataset_details = deepcopy(original_dataset_details)
            else:
                theta0 = None
            for compressor_name, _ in self.compression_methods:
                start_time = time.time()

                if i == 0:
                    # find MLE and build this compressor using per-compressor API
                    compression_data = self.find_mle_and_set_compressor(
                        single_job.data_vector,
                        single_job.covariance,
                        single_job.prior,
                        dataset_details,
                        compressor_name=compressor_name,
                    )
                    inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(
                        sbi_method,
                        dataset_details,
                        theta0,
                        compression_data,
                        single_job.prior,
                        compressor_name=compressor_name,
                        data_vector=single_job.data_vector,
                    )
                else:
                    pass

                job_compression_data, job_extra_gradients = self.compression_data_at_fiducial(compressor_name)
                self.set_compressor(compressor_name, job_compression_data, job_extra_gradients,
                                    covariance_data=single_job.covariance, prior=single_job.prior)
                job_result = None
                if likelihood_config.run:
                    logger.info('Starting likelihood inversions.')
                    start_time = time.time()
                    for result in self.run_single_gaussian_likelihood_inversion(
                        single_job, likelihood_config, compressor_name,
                        deepcopy(self.parameters), single_job.prior
                    ):
                        yield job_result, result[1]
                    logger.info(f"Time taken for likelihood inversions: {time.time() - start_time}s")

    def create_job_data(self, test_jobs_paths, real_event_jobs):

        job_data = []
        # Every test event's noise is rescaled to the first event's, so every job carries its covariance.
        first_event_covariances = {}

        for i, sim_path in enumerate(test_jobs_paths):
            theta0 = self.data_manager.data_loader.load_input_data(sim_path)
            D = self.data_manager.data_loader.load_simulation_data_array(sim_path)
            for test_noise_name, synthetic_noise_sampler in self.test_noises.items():
                if i == 0:
                    noise = synthetic_noise_sampler.draw_with_covariance()
                    first_event_covariances[test_noise_name] = noise.covariance_data
                    if noise.covariance_data is not None:
                        synthetic_noise_sampler.rescale_to(noise.covariance_data)
                else:
                    noise = synthetic_noise_sampler.draw()

                job_data.append(
                    JobData(sim_path.stem, 
                            test_noise_name,
                            D + noise.noise, 
                            theta0,
                            covariance=first_event_covariances[test_noise_name])
                    )

        job_data += self.data_manager._create_job_data_from_real_events(
            real_event_jobs, self.test_noises, data_length=self.data_manager.data_length)

        return job_data


class VaryDatasetSizeEventPipeline(MultiEventPipeline):
    """Repeat the SBI inversion over a list of training-set sizes."""

    def __init__(self, pipeline_parameters : PipelineParameters, config_path : str = None):
            
        super().__init__(pipeline_parameters, config_path)

    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details):

        from ..plotting.results_plotting import SBIPipelinePlotter
        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)
        compressed_dataset = None
        for repeat in range(3):

            for num_sims in original_dataset_details.num_simulations:

                for i, single_job in enumerate(job_data):

                    if single_job.covariance is not None:
                        self.rescale_training_noise(single_job.covariance)

                    SBIPipelinePlotter(self.job_outputs_path / f"{single_job.noise_type}", self.parameters)

                    if single_job.theta0 is not None:
                        theta0 = np.concatenate([[single_job.theta0[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
                        dataset_details = deepcopy(original_dataset_details)
                    else:
                        theta0 = None

                    for compressor_name, compressor in self.compressors.items():
                        
                        start_time = time.time()
                        
                        inversion_config = InversionConfig("", single_job.noise_type, compressor_name)
                        if i == 0:
                            dataset_details = dataset_details._replace(num_simulations=num_sims)
                            # use per-compressor MLE/compressor API
                            compression_data = self.find_mle_and_set_compressor(
                                single_job.data_vector,
                                single_job.covariance,
                                single_job.prior,
                                dataset_details,
                                compressor_name=compressor_name,
                            )
                            inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(
                                sbi_method,
                                dataset_details,
                                theta0,
                                compression_data,
                                single_job.prior,
                                compressor_name=compressor_name,
                                data_vector=single_job.data_vector,
                            )
                            compressed_dataset = job_result.compressed_dataset
                        else:
                            # reuse existing compressor and SBI model
                            x_0 = compressor.compress_data_vector(single_job.data_vector)
                            x_0_scaled = self.ground_truth_scaler.transform(x_0.reshape(1,-1)).reshape(-1)
                            if np.abs(x_0_scaled - 0.5).max() > 0.5:
                                logger.warning('x_0 problem found %s %s', np.abs(x_0_scaled - 0.5).max(), i)
                                x_0_scaled = np.clip(x_0_scaled, 0, 1.)
                            sample_results, _ = sbi_model.sample_posterior(x_0_scaled, num_samples=10000)
                            theta0_scaled = self.ground_truth_scaler.transform(theta0.reshape(1,-1)).reshape(-1)
                            inversion_data = InversionData(theta0_scaled, sample_results, deepcopy(self.ground_truth_scaler), compression_data)
                            job_result = JobResult(compressed_dataset, x_0, deepcopy(self.ground_truth_scaler))
                            compression_data = ScoreCompressionData(x_0, single_job.data_vector, compression_data.data_parameter_gradients, None)

                        logger.info(f"Time taken for {single_job.job_name} with {compressor_name}: {time.time() - start_time}s")
                        inversion_result = InversionResult(single_job.job_name+f'_{num_sims}_{repeat}', inversion_data, inversion_config)
                        yield job_result, inversion_result

                        if likelihood_config.run and num_sims == 10000 and repeat == 0:
                            logger.info('Starting likelihood inversions.')
                            start_time = time.time()
                            for result in self.run_single_gaussian_likelihood_inversion(
                                single_job, likelihood_config, compressor_name,
                                deepcopy(self.parameters), single_job.prior
                            ):
                                yield result
                            logger.info(f"Time taken for likelihood inversions: {time.time() - start_time}s")


class MLEEstimatePipeline(SingleEventPipeline):
    """Run only the maximum-likelihood search for each job."""


    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details, plot = True):

        from ..plotting.results_plotting import SBIPipelinePlotter
        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)

        for single_job in job_data:
            if single_job.covariance is not None:
                self.rescale_training_noise(single_job.covariance)

            plotter = SBIPipelinePlotter(self.job_outputs_path / f"{single_job.noise_type}", self.parameters)

            theta0, dataset_details  = self.compute_theta0_and_update_dataset(param_names, original_dataset_details, single_job.theta0)

            for compressor_name, _ in self.compression_methods:

                
                inversion_config = InversionConfig("", single_job.noise_type, compressor_name)

                # use per-compressor API for MLE and compressor setup
                compression_data = self.find_mle_and_set_compressor(
                    single_job.data_vector,
                    single_job.covariance,
                    single_job.prior,
                    dataset_details,
                    compressor_name=compressor_name,
                )
                if plot:
                    plotter.plot_synthetic_misfits(
                        single_job,
                        self.simulation_parameters.receivers,
                        compression_data.data_fiducial,
                        self.parameters.get_parameter_values('source_location')[:3],
                        self.simulation_parameters.sampling_rate,
                        covariance=self.compressors[compressor_name].C,
                    )
                inversion_data = InversionData(theta0, None, None, compression_data)
                inversion_result = InversionResult(single_job.job_name, inversion_data, inversion_config)

                yield None, inversion_result


#: The pipeline class for each value of ``inference.sbi.pipeline``.
PIPELINE_CLASSES = {
    "single_event": SingleEventPipeline,
    "multi_event": MultiEventPipeline,
    "vary_dataset_size": VaryDatasetSizeEventPipeline,
    "mle_estimate": MLEEstimatePipeline,
}
