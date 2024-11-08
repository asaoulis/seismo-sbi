
import os
import pickle
import multiprocessing as mp
import numpy as np


def results_queue_processing(results_generator, queue):
    job_results = []
    inversion_results = []
    try:
        for job_result, inversion_result in results_generator:
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

def run_asynchronous_plotting(sbi_pipeline, results_generator):
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
    return job_results, inversion_results

    
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

def run_asynchronous_results_saving(job_data, results_generator, output_path):
    try:
        results_queue = mp.Queue()
        saving_complete_event = mp.Event()
        results_processing = mp.Process(target=asynchronous_saving, args=(job_data, output_path / "inversion_results.pkl", results_queue, saving_complete_event))
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
    return job_results, inversion_results

def run_all_inversions_before_plotting(sbi_pipeline, results_generator):
    job_results = []
    inversion_results = []
    try:
        for job_result, inversion_result in results_generator:
            job_results.append(job_result)
            inversion_results.append(inversion_result)
    except Exception as e:
        print("Exception in results queue processing: ", e)
    sbi_pipeline.plot_results(job_results, inversion_results)
    return job_results,inversion_results

def convert_lists_to_arrays(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_lists_to_arrays(value)
        elif isinstance(value, list):
            if type(value[0]) is str:
                value = list(map(float, value))
            d[key] = np.array(value)
    return d