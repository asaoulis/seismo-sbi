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

    base_dir = Path('/data/alex/cps/cps_croatia/multi_models_red')
    model_to_receiver_mapping = {
        "model1.txt": ["KALN", "MOSL", "BLY"],
        "model2.txt": [ "PLIT"],
        "model4_simple.txt": ["CEY", "MOZS", "PERS"]
    }

    # base_dir = Path('/data/alex/cps/cps_long_valley/synthetic_arrangement')
    
    # rm -rf and recreate base directory
    if base_dir.exists():
        shutil.rmtree(base_dir)
    ### Start SBI Pipeline
    # croatia multi models
    base_path = Path('/home/alex/work/seismo-sbi/scripts/configs/croatia/multi_models_txt')
    models = ["model1.txt", "model2.txt", "model4_simple.txt"]
    for kappa in [3]:

        for model in models:
            config = SBI_Configuration()
            config.parse_config_file(config_path)
            
            cps_output_base_dir = "kappa_" + str(kappa)
            cps_GFs_path = base_dir / model[:-4] / cps_output_base_dir
            cps_GFs_fiducial_path = base_dir / model[:-4] / (cps_output_base_dir + "_fiducial")
            # make directories if they don't exist
            cps_GFs_path.mkdir(parents=True, exist_ok=True)
            cps_GFs_fiducial_path.mkdir(parents=True, exist_ok=True)
            config.sim_parameters = config.sim_parameters._replace(cps_GFs_path = cps_GFs_path,
                                                                cps_GFs_fiducial_path = cps_GFs_fiducial_path)
            velocity_mod_parameters = config.model_parameters.bounds['velocity_model']
            velocity_mod_parameters[0] = str(base_path / model)
            velocity_mod_parameters[1] = kappa

            config.model_parameters.nuisance["velocity_model"] = load_velocity_model(velocity_mod_parameters[0])
            print(f"Using velocity model {velocity_mod_parameters[0]} with kappa {velocity_mod_parameters[1]} for CPS perturbations.")
            print(f"Generating CPS perturbations for model {model} with kappa {kappa}...")
            config.model_parameters.bounds['velocity_model'] = velocity_mod_parameters
            receivers = config.sim_parameters.receivers
            # rebuild receivers object to only include stations relevant for this model
            relevant_stations = model_to_receiver_mapping[model]
            relevant_receivers = [rec for rec in receivers.iterate() if rec.station_name in relevant_stations]
            receivers.receivers = relevant_receivers
            config.sim_parameters = config.sim_parameters._replace(receivers = receivers)
            print(f"Using receivers {[rec.station_name for rec in config.sim_parameters.receivers.iterate()]} for model {model}.")
            generate_CPS_perturbations(config)

if __name__ == '__main__':
    main()
