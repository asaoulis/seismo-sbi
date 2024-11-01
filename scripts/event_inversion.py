import sys
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
import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname('.'), '..')))

from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import SingleEventPipeline, MultiEventPipeline, VaryDatasetSizeEventPipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for running a complete SBI pipeline. Requires a pre-specified configuration file. ')
    parser.add_argument('--config', '-c', type=str, help='Filepath of sbi_pipeline configuration file.', required = True)

    args = parser.parse_args()
    return args

def results_queue_processing(results_generator, queue):
    job_results = []
    inversion_results = []
    try:
        for job_result, inversion_result in results_generator():
            queue.put((job_result,inversion_result))
            job_results.append(job_result)
            inversion_results.append(inversion_result)
    except Exception as e:
        if job_result is not None and inversion_result is not None:
            print("Exception in results queue processing: ", e)
            raise e
    finally:
        queue.put((None, None))
    return job_results, inversion_results

def asynchronous_plotting(plotting_callable, results_queue, plotting_complete_event):
    try:
        while True:
            job_result, inversion_result = results_queue.get()
            if inversion_result is None:
                break
            plotting_callable(job_result, inversion_result, bounds = None)
        plotting_complete_event.set()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Exception in results plotting: ", e)
        raise e

    
def asynchronous_saving(job_data,output_path, results_queue, saving_complete_event):
    try:
        while True:
            job_result, inversion_result = results_queue.get()
            if inversion_result is None:
                break
            # first load existing data
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    data = pickle.load(f)
                    _, job_results, inversion_results = data
            else:
                job_results = []
                inversion_results = []
            job_results.append(job_result)
            inversion_results.append(inversion_result)
            with open(output_path, 'wb') as f:
                pickle.dump((job_data, job_results, inversion_results), f)
        saving_complete_event.set()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Exception in results saving: ", e)
        raise e

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
    Pipeline = SingleEventPipeline if config.pipeline_type == 'single_event' else MultiEventPipeline
    Pipeline = VaryDatasetSizeEventPipeline if config.pipeline_type == 'vary_dataset_size' else Pipeline
    sbi_pipeline = Pipeline(config.pipeline_parameters, config_path)
    sbi_pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters, config.dataset_parameters)

    test_jobs_paths = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, config.test_job_simulations)
    if len(test_jobs_paths) > 0:
        data_vector_length  = sbi_pipeline.data_loader.load_flattened_simulation_vector(test_jobs_paths[0]).shape[0]
    else:
        real_event_data = list(config.real_event_jobs.values())[0]
        if isinstance(real_event_data, str):
            real_event_path = real_event_data
        else:
            real_event_path = real_event_data['path']
        data_vector_length = sbi_pipeline.data_loader.load_flattened_simulation_vector(real_event_path).shape[0]

    sbi_pipeline.data_vector_length = data_vector_length

    score_compression_data, hessian_gradients = sbi_pipeline.compute_required_compression_data(config.compression_methods,
                                                                            config.model_parameters,  
                                                                            rerun_if_stencil_exists = config.pipeline_parameters.generate_dataset)
    sbi_pipeline.load_compressors(config.compression_methods, score_compression_data, (None,None), hessian_gradients)
    
    sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

    # sbi_pipeline.load_compressors(config.compression_methods, score_compression_data, hessian_gradients)
    

    # Preparations for performing sbi
    job_data = sbi_pipeline.create_job_data(test_jobs_paths, config.real_event_jobs)

    results_generator = partial(sbi_pipeline.run_compressions_and_inversions, 
                                    job_data, config.sbi_method, config.likelihood_config, config.dataset_parameters)
    output_path = Path(config.pipeline_parameters.output_directory) / 'jobs' / config.pipeline_parameters.run_name / config.pipeline_parameters.job_name
    output_path.mkdir(parents=True, exist_ok=True)
    if config.plotting_options['disable_plotting']:
        # job_results, inversion_results = list(results_generator())
        try:
            results_queue = mp.Queue()
            saving_complete_event = mp.Event()
            results_processing = mp.Process(target=asynchronous_saving, args=(job_data, output_path / "results.pkl", results_queue, saving_complete_event))
            results_processing.start()
            job_results, inversion_results = results_queue_processing(results_generator, results_queue)
            saving_complete_event.wait()
        except Exception as e:
            print("Exception in the main thread: ", e)
            import traceback
            traceback.print_exc()
        finally:
            # Terminate all child processes if an exception occurs
            for process in mp.active_children():
                process.terminate()
    else:
        if config.plotting_options['async_plotting']:
            try:
                results_queue = mp.Queue()
                plotting_complete_event = mp.Event()
                results_processing = mp.Process(target=asynchronous_plotting, args=(sbi_pipeline.plot_result, results_queue, plotting_complete_event))
                results_processing.start()
                job_results, inversion_results = results_queue_processing(results_generator, results_queue)
                plotting_complete_event.wait()
            except Exception as e:
                print("Exception in the main thread: ", e)
                import traceback
                traceback.print_exc()
            finally:
                # Terminate all child processes if an exception occurs
                for process in mp.active_children():
                    process.terminate()


        else:
            # job_results, inversion_results = list(results_generator())
            job_results = []
            inversion_results = []
            try:
                for job_result, inversion_result in results_generator():
                    job_results.append(job_result)
                    inversion_results.append(inversion_result)
            except Exception as e:
                print("Exception in results queue processing: ", e)
                    
            # sbi_pipeline.plot_results(job_results, inversion_results)

        with open(output_path / "inversion_results.pkl", 'wb') as f:
            pickle.dump((job_data, job_results, inversion_results), f)
            
        sbi_pipeline.plot_comparisons(inversion_results, config.plotting_options['test_posteriors']['chain_consumer'])




if __name__ == '__main__':
    main()
