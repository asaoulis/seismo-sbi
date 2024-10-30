
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

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname('.'), '..')))

from src.sbi.configuration import SBI_Configuration
from src.sbi.pipeline import SingleEventPipeline, SBIPipelinePlotter


# %%
# increase fontsize, labelsize, title size etc rcparams
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'axes.labelsize': 22})
plt.rcParams.update({'axes.titlesize': 22})
plt.rcParams.update({'xtick.labelsize': 22})
plt.rcParams.update({'ytick.labelsize': 22})
plt.rcParams.update({'legend.fontsize': 22})


# %%
from src.sbi.results import InversionResult, InversionData, JobResult, InversionConfig, JobData
import numpy as np
config_path = Path("../configs/single_event/noise_test.yaml")
name = 'gaussian_noise_synth'

print("Parsing config file...")
config = SBI_Configuration()
config.parse_config_file(config_path)
print("Successfully parsed config file.")

### Start SBI Pipeline

# Generate synthetics dataset

sbi_pipeline = SingleEventPipeline(config.pipeline_parameters, config.compression_methods, config_path)
sbi_pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters, config.dataset_parameters)

test_jobs_paths = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, config.test_job_simulations)
if len(test_jobs_paths) > 0:
    data_vector_length  = sbi_pipeline.data_loader.load_flattened_simulation_vector(test_jobs_paths[0]).shape[0]
else:
    real_event_path = list(config.real_event_jobs.values())[0]
    data_vector_length = sbi_pipeline.data_loader.load_flattened_simulation_vector(real_event_path).shape[0]

sbi_pipeline.data_vector_length = data_vector_length

score_compression_data, hessian_gradients = sbi_pipeline.compute_required_compression_data(config.compression_methods,
                                                                        config.model_parameters,  
                                                                        rerun_if_stencil_exists = config.pipeline_parameters.generate_dataset)
sbi_pipeline.load_compressors(config.compression_methods, score_compression_data, hessian_gradients)

sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)


param_names = sbi_pipeline.parameters.names
from copy import deepcopy
original_parameters = deepcopy(sbi_pipeline.parameters)
job_data = sbi_pipeline.create_job_data(test_jobs_paths, config.real_event_jobs)
single_job = job_data[0]
sim_name, test_noise, D, theta0_dict, covariance  = single_job
if covariance is not None:
    sbi_pipeline.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance)

plotter = SBIPipelinePlotter(sbi_pipeline.job_outputs_path / f"{test_noise}", sbi_pipeline.parameters)
plotter.plot_all_stacked_waveforms(single_job, sbi_pipeline.simulation_parameters.receivers)

if theta0_dict is not None:
    theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
else:
    theta0 = None

for compressor_name, compressor in sbi_pipeline.compressors.items():
    
    inversion_config = InversionConfig("", test_noise, compressor_name)

    inversion_data, job_result, sbi_model = sbi_pipeline.run_single_sbi_inversion('posterior', config.dataset_parameters, D, theta0, compressor_name, compressor)
    
    inversion_result = InversionResult(sim_name, inversion_data, inversion_config)


# %%
original_bounds = deepcopy(sbi_pipeline.parameters.bounds)
sbi_pipeline.parameters.bounds['moment_tensor']= np.array(sbi_pipeline.parameters.bounds['moment_tensor']) * 2/3

# %%
new_test_job_config = config.test_job_simulations._replace(random_events = 1000)._replace(fixed_events = [])

# %%
test_jobs_paths = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, new_test_job_config)
job_data = sbi_pipeline.create_job_data(test_jobs_paths, config.real_event_jobs)

# %%
solutions = []
sbi_samples = []
for job in job_data:
    sim_name, test_noise, D, theta0_dict, covariance  = job
    if theta0_dict is not None:
        theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
    if theta0 is not None:
        theta0_scaled = sbi_pipeline.ground_truth_scaler.transform(theta0.reshape(1,-1)).reshape(-1)
    else:
        theta0_scaled = None
    compression_data = sbi_pipeline.least_squares_solver.solve_least_squares(D, compressor, iterations=1)
    x_0 = compression_data.theta_fiducial
    statistic_scaler = sbi_pipeline.ground_truth_scaler
    x_0_scaled = statistic_scaler.transform(x_0.reshape(1,-1)).reshape(-1)
    # check if x_0_scaled is within bounds
    if not np.all(original_bounds['moment_tensor'][0] < x_0) or not np.all(x_0 < original_bounds['moment_tensor'][1]):
        print("x_0_scaled is not within bounds")
        continue
    sample_results, _ = sbi_model.sample_posterior(x_0_scaled, num_samples=10000)
    # try:
    #     sample_results = run_sbi_model(x_0_scaled)
    sbi_samples.append(sample_results)
    solutions.append(theta0_scaled)
    # except TimeoutException as e:
    #     print(e)
    # except Exception as e:
    #     print(f"An error occurred: {e}")

data = (np.stack(sbi_samples), np.stack(solutions))
with open("sbi_samples_solutions.pkl", "wb") as f:
    pickle.dump(data, f)

# %%
gl_samples = []
solutions = []
import src.sbi.likelihood as likelihood

likelihood_config = config.likelihood_config
for index, job in enumerate(job_data):
    sim_name, test_noise, D, theta0_dict, covariance  = job
    if theta0_dict is not None:
        theta0 = np.concatenate([[theta0_dict[param_type][param_name] for param_name in param_names] for param_type, param_names in param_names.items()])
    if theta0 is not None:
        theta0_scaled = sbi_pipeline.ground_truth_scaler.transform(theta0.reshape(1,-1)).reshape(-1)
    else:
        theta0_scaled = None
    compression_data = sbi_pipeline.least_squares_solver.solve_least_squares(D, compressor, iterations=1)
    x_0 = compression_data.theta_fiducial
    statistic_scaler = sbi_pipeline.ground_truth_scaler
    x_0_scaled = statistic_scaler.transform(x_0.reshape(1,-1)).reshape(-1)

    param_names = sbi_pipeline.parameters.names
    ensemble = False
    covariance = likelihood_config['covariance']
    if covariance == 'empirical':
        if 'optimal_score' in sbi_pipeline.compressors.keys():
            covariance = deepcopy(sbi_pipeline.compressors['optimal_score'].C)
    elif isinstance(covariance, float):
        covariance = covariance **2
    walker_burn_in = 10000
    num_samples = likelihood_config['num_samples']
    nsamples_per_walker = num_samples//10
    covariance = deepcopy(sbi_pipeline.compressors['optimal_score'].C)
    covariance_loss_callable = covariance.create_loss_callable()
    simulator_likelihood = likelihood.GaussianLikelihoodEvaluator(D, sbi_pipeline.simulation_callable, sbi_pipeline.ground_truth_scaler, loss_callable=covariance_loss_callable)

    samples = likelihood.generate_samples(simulator_likelihood.log_probability, ensemble,
                                                    sbi_pipeline.num_dim,
                                                    nsamples_per_walker=nsamples_per_walker, nwalkers=10, 
                                                    burn_in=walker_burn_in, num_processes=10, theta0=theta0_scaled)
    gl_samples.append(samples)
    solutions.append(theta0_scaled)

    if index % 5 == 0 and index > 0:
        data = (np.stack(gl_samples), np.stack(solutions))
        with open("gl_samples_solutions.pkl", "wb") as f:
            pickle.dump(data, f)


