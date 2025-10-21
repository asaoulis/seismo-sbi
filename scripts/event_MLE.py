import os
import argparse
import pickle
# prevent processes from using multiple threads
# this is necessary because otherwise the multiprocessing
# in emcee may use more threads than requested
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path
from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import MLEEstimatePipeline
from seismo_sbi.sbi import utils as utils

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

    Pipeline = MLEEstimatePipeline
    sbi_pipeline = Pipeline(config.pipeline_parameters, config_path)
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

    results_generator = sbi_pipeline.run_compressions_and_inversions(
        job_data, config.sbi_method, config.likelihood_config, config.dataset_parameters)
    
    for job_result, inversion_result in results_generator:
        pass
    
if __name__ == '__main__':
    main()
