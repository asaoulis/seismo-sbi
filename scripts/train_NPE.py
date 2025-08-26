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
from seismo_sbi.sbi.compression.ML.train import CompressionTrainer
from seismo_sbi.sbi.scalers import ZeroOneScaler, FlexibleScaler

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for running a complete SBI pipeline. Requires a pre-specified configuration file. ')
    parser.add_argument('--config', '-c', type=str, help='Filepath of sbi_pipeline configuration file.', required = True)
    # run name argument
    parser.add_argument('--run_name', '-n', type=str, help='Name of the run. Used to create a subfolder in the output directory.', default = 'default_run')
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

    # test_jobs_paths = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, config.test_job_simulations)
    test_jobs_paths = [Path('/data/alex/cps/alex/sims/filtered_data_covariance/simple_synthetic_inversions/random_event_0.h5'),]

    sbi_pipeline.compute_data_vector_properties(test_jobs_paths, config.real_event_jobs)
    score_compression_data, extra_gradients = sbi_pipeline.compute_required_compression_data(config.compression_methods,
                                                                            config.model_parameters,  
                                                                            rerun_if_stencil_exists = config.pipeline_parameters.generate_dataset)

    sbi_pipeline.load_compressors(config.compression_methods, score_compression_data, extra_gradients=extra_gradients)
    
    sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

    components = sbi_pipeline.data_manager.data_loader.components
    station_locations = sbi_pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(sbi_pipeline.parameters)
    trainer = CompressionTrainer(components, station_locations)
    dataloader_args = {
        'data_loader': sbi_pipeline.data_manager.data_loader,
        'data_folder': sbi_pipeline.simulations_output_path,
        'parameter_name_map': sbi_pipeline.parameters.names,
        'synthetic_noise_model_sampler': sbi_pipeline.training_noise_sampler,
        'data_scaler': data_scaler,
        'train_max_index': 19000,
        'train_batch_size': 256,
        'val_batch_size': 256,
        'train_shuffle': True,
        'val_shuffle': False,
        'num_workers': 20,
    }
    run_name = args.run_name
    data_path = Path(config.pipeline_parameters.output_directory)/ config.pipeline_parameters.run_name / config.pipeline_parameters.job_name
    trainer.train(run_name, epochs=250, output_path=data_path, dataloader_args=dataloader_args)

if __name__ == '__main__':
    main()
