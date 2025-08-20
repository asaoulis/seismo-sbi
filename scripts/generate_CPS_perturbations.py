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
from seismo_sbi.sbi.pipeline import SingleEventPipeline, MultiEventPipeline, VaryDatasetSizeEventPipeline
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

    Pipeline = SingleEventPipeline if config.pipeline_type == 'single_event' else MultiEventPipeline
    Pipeline = VaryDatasetSizeEventPipeline if config.pipeline_type == 'vary_dataset_size' else Pipeline
    sbi_pipeline = Pipeline(config.pipeline_parameters, config_path)
    sbi_pipeline.compression_methods = config.compression_methods
    sbi_pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters, config.dataset_parameters)

    _ = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, config.test_job_simulations)

if __name__ == '__main__':
    main()
