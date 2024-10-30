import sys
import os
import argparse
import shutil
import pickle
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname('.'), '..')))

from src.sbi.configuration import SBI_Configuration
from src.sbi.pipeline import SBIPipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for running a complete SBI pipeline. Requires a pre-specified configuration file. ')
    parser.add_argument('--config', '-c', type=str, help='Filepath of sbi_pipeline configuration file.', required = True)

    args = parser.parse_args()
    return args


def main():

    # Parse arguments and prepare configuration data

    args = parse_arguments()
    config_path = args.config

    print("Parsing config file...")
    config = SBI_Configuration()
    config.parse_config_file(config_path)
    print("Successfully parsed config file.")

    ### Start SBI Pipeline

    # Generate synthetics dataset

    sbi_pipeline = SBIPipeline(config.pipeline_parameters, config_path)
    sbi_pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters, config.dataset_parameters)

    score_compression_data, hessian_gradients = sbi_pipeline.compute_required_compression_data(config.compression_methods,
                                                                            config.model_parameters,  
                                                                            rerun_if_stencil_exists = config.pipeline_parameters.generate_dataset)

    # if only the moment tensor is variable, can compute sensitivity kernels once and use them for all simulations
    sbi_pipeline.use_kernel_simulator_if_possible(score_compression_data, 
                                                    config.dataset_parameters.sampling_method)
    
    if config.pipeline_parameters.generate_dataset:
        print("Generating dataset...")
        sbi_pipeline.generate_simulation_data(config.dataset_parameters)
        print("Successfully generated dataset.")

    # Preparations for performing sbi

    sbi_pipeline.load_compressors(config.compression_methods, score_compression_data, hessian_gradients)
    sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

    # Perform 
    test_jobs_paths = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, config.test_job_simulations)

    job_data, job_results, inversion_results = sbi_pipeline.run_compressions_and_inversions(test_jobs_paths, config.sbi_method, config.likelihood_config)

    output_path = Path(config.pipeline_parameters.output_directory) / 'jobs' / config.pipeline_parameters.run_name / config.pipeline_parameters.job_name
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "inversion_results.pkl", 'wb') as f:
        pickle.dump((job_data, job_results, inversion_results), f)

    sbi_pipeline.plot_results(config.plotting_options, job_data, job_results, inversion_results)


if __name__ == '__main__':
    main()
