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
    # output dir
    parser.add_argument('--output_dir', '-o', type=str, help='Directory to save generated CPS perturbations.', required = False, default = None)
    # kappa levels to run
    parser.add_argument('--kappa_levels', '-k', type=str, help='Comma-separated list of kappa levels to run (e.g., "0.1,0.5,1.0").', required = False, default = "0.1,0.5,1.0")
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
    
    # base_dir = Path('/data/alex/cps/cps_croatia/model_2_test')
    base_dir = Path(args.output_dir) if args.output_dir is not None else Path('/data/alex/cps/cps_croatia/model_2_test')
    kappas = [float(k) for k in args.kappa_levels.split(',')]
    # base_dir = Path('/data/alex/cps/cps_long_valley/synthetic_arrangement')
    
    # rm -rf and recreate base directory
    if base_dir.exists():
        shutil.rmtree(base_dir)
    ### Start SBI Pipeline
    for kappa in kappas:
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
