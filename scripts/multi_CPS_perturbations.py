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
from seismo_sbi.cps_simulator.compatibility import load_velocity_model

import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for running a complete SBI pipeline. Requires a pre-specified configuration file. ')
    parser.add_argument('--config', '-c', type=str, help='Filepath of sbi_pipeline configuration file.', required = True)

    args = parser.parse_args()
    return args

from generate_CPS_perturbations import generate_CPS_perturbations

def main():

    # Parse arguments and prepare configuration data

    args = parse_arguments()
    config_path = args.config

    print("Parsing config file...")

    print("Successfully parsed config file.")
    # base_dir = Path('/data/alex/cps/cps_long_valley/short_period_perturbations')
    # base_dir = Path('/data/alex/cps/cps_japan/perturbations')
    # base_dir = Path('/data/alex/cps/cps_long_valley/LV2_perturbations')
    # base_dir = Path('/data/alex/cps/cps_long_valley/tham_3pc_200m_explosion')
    # base_dir = Path('/data/alex/cps/JAN/5pc_1km_explosion')
    base_dir = Path('/data/alex/cps/cps_croatia/model_2_test')

    # base_dir = Path('/data/alex/cps/cps_long_valley/synthetic_arrangement')
    
    # rm -rf and recreate base directory
    if base_dir.exists():
        shutil.rmtree(base_dir)
    ### Start SBI Pipeline
    for kappa in [0.1]:
        config = SBI_Configuration()
        config.parse_config_file(config_path)
        cps_output_base_dir = "kappa_" + str(kappa)
        config.sim_parameters = config.sim_parameters._replace(cps_GFs_path = str(base_dir / cps_output_base_dir),
                                                                cps_GFs_fiducial_path = str(base_dir / (cps_output_base_dir + "_fiducial")))
    
        velocity_mod_parameters = config.model_parameters.bounds['velocity_model']
        velocity_mod_parameters[1] = kappa
        config.model_parameters.bounds['velocity_model'] = velocity_mod_parameters
        generate_CPS_perturbations(config)

if __name__ == '__main__':
    main()
