"""Pipeline variants that run the inversion loop differently from :class:`SingleEventPipeline`.

:class:`MultiEventPipeline` builds the compressor and trains once, on the first job, then runs the
Gaussian-likelihood inversion for every job; it is the default for training configurations.
:class:`VaryDatasetSizeEventPipeline` repeats the SBI inversion over a list of training-set sizes,
and :class:`MLEEstimatePipeline` runs only the maximum-likelihood search for each job.
"""

import time
from copy import deepcopy
from typing import List

import numpy as np

from .compression.gaussian import ScoreCompressionData
from .configuration import PipelineParameters
from .pipeline import SingleEventPipeline
from .types.results import InversionResult, InversionData, JobResult, InversionConfig, JobData


class MultiEventPipeline(SingleEventPipeline):
    def __init__(self, pipeline_parameters : PipelineParameters, config_path : str = None):
            
        super().__init__(pipeline_parameters, config_path)

    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details, do_plots = True):

        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)
        for i, single_job in enumerate(job_data):
            sim_name, test_noise, D, theta0_dict, covariance, priors = single_job
            if covariance is not None:
                self.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

            if theta0_dict is not None:
                theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
                dataset_details = self.set_known_parameters(deepcopy(original_dataset_details), theta0_dict)
            else:
                theta0 = None
            for compressor_name in self.compressor_keys:
                start_time = time.time()

                if i == 0:
                    # find MLE and build this compressor using per-compressor API
                    compression_data = self.find_mle_and_set_compressor(
                        D,
                        covariance,
                        priors,
                        dataset_details,
                        compressor_name=compressor_name,
                    )
                    inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(
                        sbi_method,
                        dataset_details,
                        theta0,
                        compression_data,
                        priors,
                        compressor_name=compressor_name,
                    )
                else:
                    pass

                #     inversion_data = InversionData(theta0_scaled, sample_results, deepcopy(self.ground_truth_scaler), compression_data)
                #     job_result = JobResult(compressed_dataset, x_0, deepcopy(self.ground_truth_scaler))

                #     compression_data = ScoreCompressionData(x_0, D, compression_data.data_parameter_gradients, None)
                    
                # print(f"Time taken for {sim_name} with {compressor_name}: {time.time() - start_time}s", flush=True)

                # inversion_result = InversionResult(sim_name, inversion_data, inversion_config)
                self.prepare_single_compressor(
                    compressor_name,
                    priors=priors,
                    covariance_data=covariance,
                    dataset_details=dataset_details,
                )
                job_result = None
                if likelihood_config["run"]:
                    print('Starting likelihood inversions.')
                    start_time = time.time()
                    for result in self.run_single_gaussian_likelihood_inversion(
                        single_job, likelihood_config, compressor_name,
                        deepcopy(self.parameters), priors
                    ):
                        yield job_result, result[1]
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

                    sim_name, test_noise, D, theta0_dict, covariance, priors = single_job
                    if covariance is not None:
                        self.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

                    SBIPipelinePlotter(self.job_outputs_path / f"{test_noise}", self.parameters)

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
                            # use per-compressor MLE/compressor API
                            compression_data = self.find_mle_and_set_compressor(
                                D,
                                covariance,
                                priors,
                                dataset_details,
                                compressor_name=compressor_name,
                            )
                            inversion_data, job_result, sbi_model = self.run_single_sbi_inversion(
                                sbi_method,
                                dataset_details,
                                theta0,
                                compression_data,
                                priors,
                                compressor_name=compressor_name,
                            )
                            compressed_dataset = job_result.compressed_dataset
                        else:
                            # reuse existing compressor and SBI model
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
                        inversion_result = InversionResult(sim_name+f'_{num_sims}_{repeat}', inversion_data, inversion_config)
                        yield job_result, inversion_result

                        if likelihood_config["run"] and num_sims == 10000 and repeat == 0:
                            print('Starting likelihood inversions.')
                            start_time = time.time()
                            for result in self.run_single_gaussian_likelihood_inversion(
                                single_job, likelihood_config, compressor_name,
                                deepcopy(self.parameters), priors
                            ):
                                yield result
                            print(f"Time taken for likelihood inversions: {time.time() - start_time}s")


class MLEEstimatePipeline(SingleEventPipeline):

    def run_compressions_and_inversions(self, job_data : List[JobData], sbi_method, likelihood_config, dataset_details, plot = True):

        from ..plotting.results_plotting import SBIPipelinePlotter
        param_names = self.parameters.names
        original_dataset_details = deepcopy(dataset_details)

        for single_job in job_data:
            sim_name, test_noise, D, theta0_dict, covariance, priors = single_job
            if covariance is not None:
                self.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

            plotter = SBIPipelinePlotter(self.job_outputs_path / f"{test_noise}", self.parameters)

            theta0, dataset_details  = self.compute_theta0_and_update_dataset(param_names, original_dataset_details, theta0_dict)

            for compressor_name in self.compressor_keys:

                
                inversion_config = InversionConfig("", test_noise, compressor_name)

                # use per-compressor API for MLE and compressor setup
                compression_data = self.find_mle_and_set_compressor(
                    D,
                    covariance,
                    priors,
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
                inversion_result = InversionResult(sim_name, inversion_data, inversion_config)

                yield None, inversion_result
