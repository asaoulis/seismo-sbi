import argparse
from pathlib import Path
from datetime import datetime, timedelta, time
from functools import partial

from seismo_sbi.data_handling.noise_collection import NoiseCollector, EventNoiseAggregator, ProcessedDataSlicer
from seismo_sbi.data_handling.noise_database import NoiseDatabaseGenerator
from seismo_sbi.instaseis_simulator.receivers import Receivers
def get_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a noise database from seismic data.")
    
    parser.add_argument(
        '--stations_file', type=Path,
        default=Path(__file__).resolve().parent / "configs/long_valley/stations.txt",
        help="Path to stations file with columns: STATION NETWORK LAT LON"
    )
    parser.add_argument(
        '--data_dir', type=Path, default=Path('/data/alex/long_valley'),
        help="Directory where seismic data (mseed, stationxml) is stored."
    )
    parser.add_argument(
        '--output_dir', type=Path, default=Path('/data/alex/noise/long_valley'),
        help="Base directory to save the generated noise database."
    )
    parser.add_argument(
        '--event_name', type=str, default='LV2',
        help="Name for the noise database."
    )
    parser.add_argument(
        '--event_starttime', type=str, default='1997-11-22T17:20:35',
        help="Event start time in ISO format (e.g., '1997-11-22T17:20:35')."
    )
    parser.add_argument(
        '--event_endtime', type=str, default='1997-11-22T17:23:54',
        help="Event end time in ISO format (e.g., '1997-11-22T17:23:34')."
    )
    parser.add_argument(
        '--num_jobs', type=int, default=20, help="Number of parallel jobs for noise collection."
    )
    parser.add_argument(
        '--max_frequency', type=float, default=1.0, help="Maximum frequency for processing."
    )
    parser.add_argument(
        '--duration', type=int, default=5, help="Duration in minutes for covariance estimation window."
    )
    parser.add_argument(
        '--data_vector_length', type=int, default=200, help="Length of the data vector."
    )
    return parser.parse_args()

def create_lv_data_format(data_path):
    """Creates a data format configuration for Long Valley."""
    def create_format(network):
        return {
            'call_key': 'OBS',
            'master_path': '.',
            'path_structure': f'{data_path}/{{sta}}/{{year}}.{{jday}}/{{net}}.{{sta}}..{{cha}}.{{year}}.{{jday}}.mseed',
            'sta_cha': ['BHZ', 'BHE', 'BHN'],
            'network': network,
            'location': '',
            'years': [1997], # Expanded to include 1997
            'instrument_correction': True,
            'response_seismometer': f'{data_path}/RESP/RESP.{{net}}.{{sta}}.{{loc}}.{{cha}}',
        }
    return {'BK': create_format('BK'), 'US': create_format('US')}

def load_stations(stations_path):
    """Load station codes from a file."""
    stations = []
    with stations_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                sta, net = parts[0], parts[1]
                stations.append((net, sta))
    return {f'{sta}': f'{net}' for net, sta in stations}

def main():
    args = get_arguments()

    # Load station information
    station_codes_paths = load_stations(args.stations_file)

    # Configure paths and data formats
    station_config = create_lv_data_format(args.data_dir)
    daily_output_directory = args.output_dir / f"{args.event_name}_daily"
    daily_output_directory.mkdir(parents=True, exist_ok=True)

    # Define processing parameters
    prefilter_kwargs = dict(pre_filt=[0.005, 0.01, 0.2, 0.4])
    filter_kwargs = dict(freqmin=1/50, freqmax=1/20, corners=4, zerophase=False)

    # Define event window and location
    event_window = [datetime.fromisoformat(args.event_starttime), datetime.fromisoformat(args.event_endtime)]
    event_location = (37.6, -117.0)

    # Initialize noise collection and aggregation
    noise_collector = NoiseCollector(station_config, station_codes_paths, {}, prefilter_kwargs=prefilter_kwargs, filter_kwargs=filter_kwargs)
    event_noise_aggregator = EventNoiseAggregator(noise_collector, station_codes_paths, sampling_rate=args.max_frequency)
    
    # Find available stations for the event
    event_noise_aggregator.select_event_and_check_available_stations(event_window, event_location, buffer=timedelta(minutes=1), convert_to_numpy=True)
    print(f"Found {len(event_noise_aggregator.available_stations_during_event)} available stations for the event.")

    # Define time windows for noise extraction
    times = [time(17, 15, 0), time(17, 30, 0)]
    windows = [[datetime.combine(event_window[0].date(), t) for t in times]]

    # Set up the noise database generator
    noise_collection_callable = partial(event_noise_aggregator.collect_noise_data, noise_window_length=timedelta(hours=1), convert_to_numpy=False)
    noise_database_generator = NoiseDatabaseGenerator(noise_collection_callable, num_jobs=args.num_jobs, mseed_output=True)

    # Create the noise database
    print(f"Generating database in {daily_output_directory}...")
    noise_database_generator.create_database(daily_output_directory, windows)
    print("Database generation complete.")
    print("Slicing data and generating event file...")
    data_slicer = ProcessedDataSlicer(
        data_folder=daily_output_directory,
        sampling_rate=args.max_frequency,
        covariance_estimation_window=timedelta(minutes=args.duration),
        full_auto_correlation=True,
        receivers=event_noise_aggregator.available_stations_during_event
    )

    output_path = args.output_dir / 'events'
    output_path.mkdir(parents=True, exist_ok=True)
    
    noise_window = [datetime.fromisoformat(args.event_starttime), datetime.fromisoformat(args.event_endtime)]

    print(f"Data vector length: {args.data_vector_length}")
    data_saver = NoiseDatabaseGenerator(data_slicer.load_noise_window_data, data_vector_length=args.data_vector_length, num_jobs=args.num_jobs)
    data_saver._collect_and_save_noise(
        output_path,
        noise_window=noise_window,
        name=f'{args.event_name}_noise_filt_20_50_1hz'
    )
    print("Final noise database saved.")

if __name__ == "__main__":
    main()