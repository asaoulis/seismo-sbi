# %%
import sys
import os
import argparse
import shutil
import pickle
from pathlib import Path
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import SingleEventPipeline, SBIPipelinePlotter


# %%
# increase fontsize, labelsize, title size etc rcparams
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.labelsize': 14})
plt.rcParams.update({'axes.titlesize': 14})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'ytick.labelsize': 14})
plt.rcParams.update({'legend.fontsize': 14})


# %%


from seismo_sbi.sbi.pipeline import MLEEstimatePipeline
import numpy as np
base_dir = Path('/data/alex/cps/cps_long_valley/LV2_perturbations')
config_path = Path('./configs/long_valley/LV2/LV2_synthetic_inversion.yaml')
results = {}
compressed_data_all = {}


### Start SBI Pipeline
kappa = 5
config = SBI_Configuration()
config.parse_config_file(config_path)
cps_output_base_dir = "kappa_" + str(kappa)
config.sim_parameters = config.sim_parameters._replace(cps_GFs_path = str(base_dir / cps_output_base_dir),
                                                        cps_GFs_fiducial_path = str(base_dir / (cps_output_base_dir + "_fiducial")))
sbi_pipeline = MLEEstimatePipeline(config.pipeline_parameters,  config_path)
sbi_pipeline.compression_methods = config.compression_methods
sbi_pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters, config.dataset_parameters)

test_jobs_paths = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, config.test_job_simulations)
sbi_pipeline.compute_data_vector_properties(test_jobs_paths, config.real_event_jobs)
score_compression_data, extra_gradients = sbi_pipeline.compute_required_compression_data(config.compression_methods,
                                                                        config.model_parameters,  
                                                                        rerun_if_stencil_exists = config.pipeline_parameters.generate_dataset)

sbi_pipeline.load_compressors(config.compression_methods, score_compression_data, extra_gradients=extra_gradients)

sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

# Preparations for performing sbi
job_data = sbi_pipeline.create_job_data(test_jobs_paths, config.real_event_jobs)

param_names = sbi_pipeline.parameters.names
original_dataset_details = deepcopy(dataset_details)
iterative_lsq_results = {}
for damping in [0.0, 0.1]:

    for single_job in job_data:
        sim_name, test_noise, D, theta0_dict, covariance, priors = single_job
        if covariance is not None:
            sbi_pipeline.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

        plotter = SBIPipelinePlotter(sbi_pipeline.job_outputs_path / f"{test_noise}", sbi_pipeline.parameters)

        theta0, dataset_details  = sbi_pipeline.compute_theta0_and_update_dataset(param_names, original_dataset_details, theta0_dict)

        for compressor_name, compressor in sbi_pipeline.compressors.items():

            only_moment_tensor_variable = all([sampler =='constant' for param, sampler in dataset_details.sampling_method.items() if param != 'moment_tensor'])
            single_least_squares_step =  only_moment_tensor_variable
            compression_data, extra_gradients = sbi_pipeline.compute_required_compression_data(sbi_pipeline.compression_methods, sbi_pipeline.parameters,)
            sbi_pipeline.load_compressors(sbi_pipeline.compression_methods, score_compression_data=compression_data, priors=priors, covariance_data=covariance_data, extra_gradients=extra_gradients)
            compressor = list(sbi_pipeline.compressors.values())[0]
            _, _, history = sbi_pipeline.least_squares_solver.solve_least_squares(single_job.data_vector, compressor, single_step=single_least_squares_step, return_history=True)

    iterative_lsq_results[damping]  = history

