from pathlib import Path

from tqdm import tqdm
import joblib
import traceback


from src.instaseis_simulator.simulation_saver import SimulationSaver
from src.instaseis_simulator.dataset_generator import tqdm_joblib


class NoiseDatabaseGenerator:

    def __init__(self, collect_noise_callable, data_vector_length = None, mseed_output=False, num_jobs = 1) -> None:

        self.collect_noise_generator = collect_noise_callable
        self.data_vector_length = data_vector_length
        self.num_jobs = num_jobs
        self.mseed_output = mseed_output
    
    def create_database(self, base_output_path: Path, padded_noise_windows):

        with tqdm_joblib(tqdm(desc="Collecting noise windows: ", total=len(padded_noise_windows))) as progress_bar:
            with joblib.parallel_backend('loky', n_jobs=self.num_jobs):
                results = joblib.Parallel()(
                    joblib.delayed(self._collect_and_save_noise)(base_output_path, noise_window) for
                        noise_window in padded_noise_windows
                )
        # print(f"Collected {sum(results)} noise windows out of {len(padded_noise_windows)}.")
    #    for noise_window in padded_noise_windows:
    #         self._collect_and_save_noise(base_output_path, noise_window) 


    def _collect_and_save_noise(self, base_output_path, noise_window, name = None):
        if name is None:
            noise_window_string_id = noise_window[0].strftime("%Y.%m.%d.%H.%M")
        else:
            noise_window_string_id = name
        output_path = base_output_path / f'{noise_window_string_id}.h5'
        try:
            noise_data = self.collect_noise_generator(noise_window)
        except Exception as e:
            traceback.print_exc()
            return
        if noise_data is not None:
            if isinstance(noise_data, tuple):
                return self.save_data_stream(noise_window_string_id, output_path, *noise_data)
            else:
                return self.save_data_stream(noise_window_string_id, output_path, noise_data)
        else:
            return 0

    def save_data_stream(self, noise_window_string_id, output_path, noise_stream, misc_data = None):
        if self._check_data_quality(noise_stream):
            if self.mseed_output:
                from obspy import Stream
                all_noise_streams = Stream()
                for key in noise_stream:
                    for comp, trace in noise_stream[key].items():
                        all_noise_streams.append(trace)

                all_noise_streams.write(str(output_path), format='mseed')
            else:
                trace_saver = SimulationSaver(output_data=noise_stream, misc_data=misc_data)
                trace_saver.dump_data_as_hdf5(output_path)
            return 1
        else:
            return 0

    
    def _check_data_quality(self, noise_stream):
        # check all arrays in dict recursively
        if self.data_vector_length is None:
            return True
        if isinstance(noise_stream, dict):
            return all(self._check_data_quality(noise_stream[key]) for key in noise_stream)
        else:
            if noise_stream.shape[0] != self.data_vector_length:
                print(noise_stream.shape[0], self.data_vector_length)
            return noise_stream.shape[0] == self.data_vector_length
