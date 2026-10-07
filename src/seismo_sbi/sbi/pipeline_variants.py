"""Pipeline variants that run the inversion loop differently from :class:`SingleEventPipeline`.

:class:`MultiEventPipeline` builds the compressor and trains once, on the first job, then runs the
Gaussian-likelihood inversion for every job; it is the default for training configurations.
:class:`VaryDatasetSizeEventPipeline` repeats the SBI inversion over a list of training-set sizes,
and :class:`MLEEstimatePipeline` runs only the maximum-likelihood search for each job.
"""

import logging
import time
from copy import deepcopy
from typing import List

import numpy as np

from .compression.gaussian import ScoreCompressionData
from .configuration import PipelineParameters
from .pipeline import SingleEventPipeline
from .types.results import InversionResult, InversionData, JobResult, InversionConfig, JobData

logger = logging.getLogger(__name__)


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
            for compressor_name in self.compressor_keys:
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
                    )
                else:
                    pass

                self.prepare_single_compressor(
                    compressor_name,
                    prior=single_job.prior,
                    covariance_data=single_job.covariance,
                    dataset_details=dataset_details,
                )
                job_result = None
                if likelihood_config["run"]:
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

                        if likelihood_config["run"] and num_sims == 10000 and repeat == 0:
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

            for compressor_name in self.compressor_keys:

                
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
                        self.parameters.get_parameter_values('source_location')[:2],
                        covariance=self.empirical_cov_mat,
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
