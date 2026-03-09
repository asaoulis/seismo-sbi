import os
import argparse
import pickle
import shutil

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
from seismo_sbi.cps_simulator.compatibility import load_velocity_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for running a complete SBI pipeline. Requires a pre-specified configuration file. ')
    parser.add_argument('--config', '-c', type=str, help='Filepath of sbi_pipeline configuration file.', required = True)

    args = parser.parse_args()
    return args


def find_copy_and_remove_CPS_fiducial(cps_output_path, cps_fiducial_path):
    cps_output_path = Path(cps_output_path)
    cps_fiducial_path = Path(cps_fiducial_path)

    # Find directories inside the output path
    dirs = [d for d in cps_output_path.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No directories found in CPS GFs path {cps_output_path}")

    # Assume the first directory is the fiducial directory
    fiducial_dir = dirs[0]

    # Get all files inside it
    fiducial_files = [f for f in fiducial_dir.iterdir() if f.is_file()]
    if not fiducial_files:
        raise FileNotFoundError(f"No files found in CPS GFs fiducial directory {fiducial_dir}")

    # Ensure destination directory exists
    cps_fiducial_path.mkdir(parents=True, exist_ok=True)

    for file in fiducial_files:
        dest_file = cps_fiducial_path / file.name

        # Convenient and preserves metadata
        shutil.copy2(file, dest_file)

        file.unlink()

    # Try removing directory if now empty
    try:
        fiducial_dir.rmdir()
    except OSError:
        print(f"Directory {fiducial_dir} not empty, not removed.")


def main():

    # Parse arguments and prepare configuration data

    args = parse_arguments()
    config_path = args.config

    print("Parsing config file...")
    config = SBI_Configuration()
    config.parse_config_file(config_path)
    print("Successfully parsed config file.")

    ### Start SBI Pipeline

    generate_CPS_perturbations(config)

def generate_CPS_perturbations(config):
    config_path = None
    Pipeline = SingleEventPipeline if config.pipeline_type == 'single_event' else MultiEventPipeline
    Pipeline = VaryDatasetSizeEventPipeline if config.pipeline_type == 'vary_dataset_size' else Pipeline
    sbi_pipeline = Pipeline(config.pipeline_parameters, config_path)
    sbi_pipeline.compression_methods = config.compression_methods
    sbi_pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters, config.dataset_parameters)
    fiducial_velocity_model = config.model_parameters.nuisance['velocity_model']
    custom_fiducial_job = {'fiducial_job': {"velocity_model": fiducial_velocity_model, **list(config.test_job_simulations.custom_events.values())[0]}}
    only_custom_jobs = config.test_job_simulations._replace(random_events=0, fixed_events=[], custom_events=custom_fiducial_job)  
    _ = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, only_custom_jobs)
    find_copy_and_remove_CPS_fiducial(config.sim_parameters.cps_GFs_path, config.sim_parameters.cps_GFs_fiducial_path)
    only_perturbation_jobs = config.test_job_simulations._replace(custom_events={})
    _ = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, only_perturbation_jobs)

if __name__ == '__main__':
    main()
