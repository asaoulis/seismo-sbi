from constants import ( UPFLOW_config, STATION_CODES_PATHS, 
                       STATION_LOCATIONS, STATION_CODES_PATHS )

import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import math as m

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname('.'), '..')))

from src.data_handling.noise_collection import NoiseCollector, EventNoiseAggregator, ProcessedDataSlicer
from src.data_handling.event_window_selection import EventWindowSelector
from src.data_handling.noise_database import NoiseDatabaseGenerator

from src.instaseis_simulator.receivers import Receivers
from src.sbi.noises.covariance_estimation import EmpiricalCovarianceEstimator


NUM_JOBS = 14

noise_name = 'november_cluster'

DAILY_OUTPUT_DIRECTORY = Path(f'data/noise/{noise_name}_daily')
DAILY_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

NOISE_OUTPUT_DIRECTORY = Path(f'data/noise/{noise_name}_samples')
NOISE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

date_format = '%Y-%m-%d'
noise_start_time = '2021-10-23'
noise_end_time = '2021-11-15'


# event_window = [datetime(2022, 3, 29, 21, 55, 16), datetime(2022, 3, 29, 22, 10, 16)] # sao jorge M4
# event_location = (38.67, -28.1)

# # event_location = (sao_jorge_lat, sao_jorge_lon)
# event_window = [datetime(2022, 2, 16, 4, 31, 40), datetime(2022, 2, 16, 4, 46, 40)] # madeira M5
# event_location = (32.3700, -16.7000) # madeira

# # 021/12/07 15:06:52.81 Event 621618746 Azores-Cape St. Vincent Ridge
# event_window = [datetime(2021, 12, 7, 15, 5, 52), datetime(2021, 12, 7, 15, 20, 52)]
# event_location = (37.6302,  -19.3792)

# # north islands 2022/01/13 06:46:12.23 Event 621827081
# event_window = [datetime(2022, 1, 13, 6, 45, 12), datetime(2022, 1, 13, 7, 0, 12)]
# event_location = (39.9267,  -29.9392)

# 2022/03/06 02:26:13.60 Event 622117678 Azores Islands region # north_2
# event_window = [datetime(2022, 3, 6, 2, 25, 13), datetime(2022, 3, 6, 2, 40, 13)]
# event_location = (40.7900, -29.1900)

# 2021/11/10 11:02:12.20 Event 621385927 Azores Islands azores_2
event_window = [datetime(2021, 11, 10, 11, 1, 12), datetime(2021, 11, 10, 11, 16, 12)]
event_location = (36.7800,  -33.3700)

# 2021/12/31 00:58:33.50 Event 621657646 Azores Islands region azores_3
# event_window = [datetime(2021, 12, 31, 0, 57, 33), datetime(2021, 12, 31, 1, 12, 33)]
# event_location = (37.5700,  -31.7500)

# # 2022/01/13 06:46:15.90 Event 621827081 Azores Islands azores_4
# event_window = [datetime(2022, 1, 13, 6, 45, 15), datetime(2022, 1, 13, 7, 0, 15)]
# event_location = (40.1700, -29.7000 ) 

# 2021/11/14 16:32:44.03 Event 621468712 Azores Islands region gloria
# event_window = [datetime(2021, 11, 14, 16, 31, 44), datetime(2021, 11, 14, 16, 46, 44)]
# event_location = (37.1036 , -23.9688)

max_frequency =  1
noise_collector = NoiseCollector(UPFLOW_config, STATION_CODES_PATHS, STATION_LOCATIONS)
event_noise_aggregator = EventNoiseAggregator(noise_collector, STATION_CODES_PATHS, sampling_rate = max_frequency)
event_noise_aggregator.select_event_and_check_available_stations(event_window, event_location, buffer=timedelta(minutes=3), convert_to_numpy=True)


receivers = Receivers(station_config=UPFLOW_config, 
                      stations=event_noise_aggregator.available_stations_during_event, 
                      station_codes_paths=STATION_CODES_PATHS)


pre_event_window = (event_window[0] - timedelta(minutes=15), event_window[0])


event_window_selector = EventWindowSelector(num_jobs=NUM_JOBS)
distant_events, near_events = event_window_selector.select_events(
    event_location[0], event_location[1],
    noise_start_time, noise_end_time,
    distant_min_radius = 20, distant_min_magnitude = 6.0,
    close_max_radius = 20, close_min_magnitude=4
)
           
event_start_end_times = event_window_selector.get_unavailability_time_pairs(
    receivers, near_events, distant_events
)

continuous_noise_regions, gaps = event_window_selector.get_continuous_regions(event_start_end_times,
    datetime.strptime(noise_start_time, date_format), 
    datetime.strptime(noise_end_time, date_format))

noise_time_windows= event_window_selector.create_daily_overlapping_windows_from_regions(
    continuous_noise_regions, window_length=timedelta(minutes= 20), buffer = timedelta(minutes=20), overlap_offset=timedelta(minutes=3)
)
print("Number of day windows: ", len(noise_time_windows))
print("Num noise windows: ", sum([len(windows) for windows in noise_time_windows.values()]) )

# create a datetime window for each date in noise_time_windows
times = [datetime.time(datetime.strptime('00:00:00', '%H:%M:%S')), datetime.time(datetime.strptime('23:59:59', '%H:%M:%S'))]
windows = [[datetime.combine(date, time) for time in times] for date in noise_time_windows.keys()]

from functools import partial

window_length = timedelta(minutes=15)
noise_collection_callable = partial(event_noise_aggregator.collect_noise_data, noise_window_length=timedelta(hours=24),  convert_to_numpy = False)

noise_database_generator = NoiseDatabaseGenerator(noise_collection_callable, num_jobs=NUM_JOBS, mseed_output=True)

noise_database_generator.create_database(DAILY_OUTPUT_DIRECTORY, windows)

data_slicer = ProcessedDataSlicer(data_folder=DAILY_OUTPUT_DIRECTORY, 
                                  sampling_rate=max_frequency, 
                                  covariance_estimation_window=timedelta(minutes=15),
                                  full_auto_correlation=True)
data_saver = NoiseDatabaseGenerator(data_slicer.load_noise_window_data, data_vector_length=901, num_jobs=NUM_JOBS)
data_saver._collect_and_save_noise(Path('./data/events/'), noise_window=event_window, name=f'{noise_name}_event_filtered_1hz')

h5_files = list(Path(DAILY_OUTPUT_DIRECTORY).glob("*.h5"))[:-6]
successful_files = []
for h5_file in h5_files:
    # Extract the file's stem (filename without extension)
    file_stem = h5_file.stem
    
    # Parse the stem into a datetime object
    try:
        file_datetime = datetime.strptime(file_stem, "%Y.%m.%d.%H.%M").date()
        successful_files.append(file_datetime)
    except ValueError:
        continue
print(len(successful_files))

noise_time_windows= event_window_selector.create_daily_overlapping_windows_from_regions(
    continuous_noise_regions, window_length=timedelta(minutes= 15), buffer = timedelta(minutes=20), overlap_offset=timedelta(minutes=3)
)

final_noise_windows = {date: noise_time_windows[date] for date in successful_files}

print("Number of day windows: ", len(final_noise_windows))
print("Num noise windows: ", sum([len(windows) for windows in final_noise_windows.values()]) )

data_slicer = ProcessedDataSlicer(data_folder=DAILY_OUTPUT_DIRECTORY, 
                                  sampling_rate=max_frequency, 
                                  covariance_estimation_window=timedelta(minutes=15),
                                  full_auto_correlation=True)
flattened_time_windows = [time for date in final_noise_windows for time in final_noise_windows[date]]

noise_database_generator = NoiseDatabaseGenerator(data_slicer.load_noise_window_data, data_vector_length=901, num_jobs=NUM_JOBS)

noise_database_generator.create_database(NOISE_OUTPUT_DIRECTORY, flattened_time_windows)