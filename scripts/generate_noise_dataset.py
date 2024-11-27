from pathlib import Path
from datetime import datetime, timedelta

from seismo_sbi.data_handling.noise_collection import NoiseCollector, EventNoiseAggregator, ProcessedDataSlicer
from seismo_sbi.data_handling.event_window_selection import EventWindowSelector
from seismo_sbi.data_handling.noise_database import NoiseDatabaseGenerator
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length

from seismo_sbi.instaseis_simulator.receivers import Receivers

from constants import STATION_CODES_PATHS, create_ipma_data_format, STATION_CODES_PATHS_INDO, INDO_DATA_FORMAT
STATION_CODES_PATHS = STATION_CODES_PATHS_INDO
output_dir = Path('/data/alex')
station_config = create_ipma_data_format(output_dir)
station_config = INDO_DATA_FORMAT
PREFILTER_KWARGS = dict(pre_filt=[0.004, 0.008, 0.04, 0.08])
FILTER_KWARGS = dict(freqmin=1/100, freqmax=1/50, corners=4, zerophase=True)
DURATION = 29
NUM_JOBS = 20

noise_name = 'long_period'

DAILY_OUTPUT_DIRECTORY = Path(f'/data/alex/noise/indo_pacific/{noise_name}_daily')
DAILY_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

NOISE_OUTPUT_DIRECTORY = Path(f'/data/alex/noise/indo_pacific/{noise_name}_samples')
NOISE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

date_format = '%Y-%m-%d'
noise_start_time = '2024-09-01'
noise_end_time = '2024-09-20'

# # north islands 2022/01/13 06:46:12.23 Azores Islands Event 621827081
event_window = [datetime(2024, 9, 13, 6, 45, 12), datetime(2024, 9, 13, 7, 0, 12)]
event_location = (5,  117)

max_frequency =  0.5
noise_collector = NoiseCollector(station_config, STATION_CODES_PATHS, {}, prefilter_kwargs=PREFILTER_KWARGS, filter_kwargs=FILTER_KWARGS)
event_noise_aggregator = EventNoiseAggregator(noise_collector, STATION_CODES_PATHS, sampling_rate = max_frequency)
event_noise_aggregator.select_event_and_check_available_stations(event_window, event_location, buffer=timedelta(minutes=3), convert_to_numpy=True)
print("Num stations ", len(event_noise_aggregator.available_stations_during_event))
receivers = Receivers(station_config=station_config, 
                      stations=event_noise_aggregator.available_stations_during_event, 
                      station_codes_paths=STATION_CODES_PATHS)


pre_event_window = (event_window[0] - timedelta(minutes=DURATION), event_window[0])


event_window_selector = EventWindowSelector(num_jobs=NUM_JOBS)
distant_events, near_events = event_window_selector.select_events(
    event_location[0], event_location[1],
    noise_start_time, noise_end_time,
    distant_min_radius = 40, distant_min_magnitude = 6.0,
    close_max_radius = 20, close_min_magnitude=4.5
)
try:
    event_start_end_times = event_window_selector.get_unavailability_time_pairs(
        receivers, near_events, distant_events
    )

    continuous_noise_regions, gaps = event_window_selector.get_continuous_regions(event_start_end_times,
        datetime.strptime(noise_start_time, date_format), 
        datetime.strptime(noise_end_time, date_format))
except Exception as e:
    print('Error with event window selector')
    print(e)
    continuous_noise_regions = [(datetime.strptime(noise_start_time, date_format), datetime.strptime(noise_end_time, date_format))]
noise_time_windows= event_window_selector.create_daily_overlapping_windows_from_regions(
    continuous_noise_regions, window_length=timedelta(minutes= 20), buffer = timedelta(minutes=20), overlap_offset=timedelta(minutes=3)
)
print("Number of day windows: ", len(noise_time_windows))
print("Num noise windows: ", sum([len(windows) for windows in noise_time_windows.values()]) )

# create a datetime window for each date in noise_time_windows
times = [datetime.time(datetime.strptime('00:00:00', '%H:%M:%S')), datetime.time(datetime.strptime('23:59:59', '%H:%M:%S'))]
windows = [[datetime.combine(date, time) for time in times] for date in noise_time_windows.keys()]

from functools import partial

window_length = timedelta(minutes=DURATION)
noise_collection_callable = partial(event_noise_aggregator.collect_noise_data, noise_window_length=timedelta(hours=24),  convert_to_numpy = False)

noise_database_generator = NoiseDatabaseGenerator(noise_collection_callable, num_jobs=NUM_JOBS, mseed_output=True)

# noise_database_generator.create_database(DAILY_OUTPUT_DIRECTORY, windows)

data_slicer = ProcessedDataSlicer(data_folder=DAILY_OUTPUT_DIRECTORY, 
                                  sampling_rate=max_frequency, 
                                  covariance_estimation_window=timedelta(minutes=DURATION),
                                  full_auto_correlation=True,
                                  receivers=event_noise_aggregator.available_stations_during_event)

data_vector_length = compute_data_vector_length(DURATION*60, max_frequency) + 1
print(data_vector_length)
data_saver = NoiseDatabaseGenerator(data_slicer.load_noise_window_data, data_vector_length=data_vector_length, num_jobs=NUM_JOBS)
# data_saver._collect_and_save_noise(Path('/data/alex/indo_pacific/events/'), noise_window=event_window, name=f'{noise_name}_event_filtered_1hz')

h5_files = list(Path(DAILY_OUTPUT_DIRECTORY).glob("*.h5"))
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
    continuous_noise_regions, window_length=timedelta(minutes= DURATION), buffer = timedelta(minutes=20), overlap_offset=timedelta(minutes=3)
)

final_noise_windows = {date: noise_time_windows[date] for date in successful_files}

print("Number of day windows: ", len(final_noise_windows))
print("Num noise windows: ", sum([len(windows) for windows in final_noise_windows.values()]) )

data_slicer = ProcessedDataSlicer(data_folder=DAILY_OUTPUT_DIRECTORY, 
                                  sampling_rate=max_frequency, 
                                  covariance_estimation_window=timedelta(minutes=DURATION),
                                  full_auto_correlation=True,
                                  receivers=event_noise_aggregator.available_stations_during_event)
flattened_time_windows = [time for date in final_noise_windows for time in final_noise_windows[date]]

noise_database_generator = NoiseDatabaseGenerator(data_slicer.load_noise_window_data, data_vector_length=data_vector_length, num_jobs=NUM_JOBS)

noise_database_generator.create_database(NOISE_OUTPUT_DIRECTORY, flattened_time_windows)