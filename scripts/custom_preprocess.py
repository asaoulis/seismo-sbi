"""Custom preprocessing script using the new obspy-centred pipeline.

Replaces the legacy NoiseCollector / EventNoiseAggregator / ProcessedDataSlicer
pipeline.  The new flow:

  1. find_mseed_files  — locate raw data on disk
  2. load_waveforms    — read into a single obspy.Stream
  3. load_inventory    — read StationXML from a directory
  4. deconvolve_and_filter — remove response, bandpass, resample
  5. write_window      — write preprocessed daily mseed (optional, for audit)
  6. export_to_sbi_h5  — the ONLY h5 write; RealNoiseSampler reads this

Usage:
    python custom_preprocess.py \
        --stations_file configs/long_valley/stations.txt \
        --data_dir /data/alex/long_valley \
        --output_dir /data/alex/noise/long_valley \
        --event_name LV2 \
        --event_starttime 1997-11-22T17:20:35 \
        --event_endtime 1997-11-22T17:23:54
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import obspy
from obspy import UTCDateTime

from seismo_sbi.data_handling.preprocessing import (
    find_mseed_files,
    load_waveforms,
    load_inventory,
    write_window,
    deconvolve_and_filter,
    export_to_sbi_h5,
)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess seismic data for seismo-sbi using the new obspy API."
    )
    parser.add_argument(
        "--stations_file", type=Path,
        default=Path(__file__).resolve().parent / "configs/long_valley/stations.txt",
        help="Path to stations file with columns: STATION NETWORK LAT LON",
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path("/data/alex/long_valley"),
        help="Root directory containing {station}/{year}.{jday}/ mseed data.",
    )
    parser.add_argument(
        "--stationxml_dir", type=Path, default=None,
        help="Directory containing StationXML files.  Defaults to <data_dir>/stationxml.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("/data/alex/noise/long_valley"),
        help="Base directory for preprocessed output.",
    )
    parser.add_argument("--event_name", type=str, default="LV2")
    parser.add_argument(
        "--event_starttime", type=str, default="1997-11-22T17:20:35",
        help="Event start time (ISO 8601).",
    )
    parser.add_argument(
        "--event_endtime", type=str, default="1997-11-22T17:23:54",
        help="Event end time (ISO 8601).",
    )
    parser.add_argument(
        "--max_frequency", type=float, default=1.0,
        help="Target sampling rate (Hz) after resampling.",
    )
    parser.add_argument(
        "--duration", type=int, default=5,
        help="Covariance estimation window (minutes) before event start.",
    )
    parser.add_argument(
        "--channel_glob", type=str, default="BH?",
        help="Glob pattern for channel selection (e.g. 'BH?' or 'HH?').",
    )
    parser.add_argument(
        "--no_instrument_correction", action="store_true",
        help="Skip instrument response removal (useful for synthetic data).",
    )
    return parser.parse_args()


def load_stations(stations_path: Path) -> list:
    """Load (station, network) pairs from a whitespace-delimited file.

    Expects columns: STATION NETWORK [LAT LON ...]
    """
    stations = []
    with stations_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                stations.append((parts[0], parts[1]))  # (station, network)
    return stations


def main():
    args = get_arguments()

    event_start = datetime.fromisoformat(args.event_starttime)
    event_end = datetime.fromisoformat(args.event_endtime)
    cov_window = timedelta(minutes=args.duration)
    remove_response = not args.no_instrument_correction

    stationxml_dir = args.stationxml_dir or (args.data_dir / "stationxml")

    # Preprocessing parameters matching Long Valley defaults
    prefilter_kwargs = dict(pre_filt=[0.005, 0.01, 0.2, 0.4])
    filter_kwargs = dict(freqmin=1 / 50, freqmax=1 / 20, corners=4, zerophase=False)

    # Load stations list
    station_pairs = load_stations(args.stations_file)
    print(f"Loaded {len(station_pairs)} stations from {args.stations_file}")

    # Time window to load: event window + covariance pre-window + 2-min padding
    t0_load = UTCDateTime(event_start) - cov_window.total_seconds() - 120
    t1_load = UTCDateTime(event_end) + 120

    # Find and load mseed files for all stations
    combined_stream = obspy.Stream()
    available_stations = []

    for station, network in station_pairs:
        paths = find_mseed_files(
            args.data_dir, station, t0_load, t1_load,
            network=network, channel_glob=args.channel_glob,
        )
        if not paths:
            print(f"  No data for {network}.{station} — skipping")
            continue
        try:
            st = load_waveforms(paths, starttime=t0_load, endtime=t1_load)
            if len(st) == 0:
                print(f"  Empty stream for {network}.{station} — skipping")
                continue
            combined_stream += st
            available_stations.append(station)
        except Exception as exc:
            print(f"  Error loading {network}.{station}: {exc}")
            continue

    print(f"Available stations: {available_stations} ({len(available_stations)} total)")

    if len(available_stations) == 0:
        print("No stations available — aborting.")
        sys.exit(1)

    # Load inventory and process
    inventory = None
    if remove_response:
        if not stationxml_dir.is_dir():
            print(
                f"WARNING: stationxml_dir {stationxml_dir} not found — "
                "skipping response removal."
            )
            remove_response = False
        else:
            try:
                inventory = load_inventory(stationxml_dir)
                print(f"Loaded inventory from {stationxml_dir}")
            except FileNotFoundError as exc:
                print(f"WARNING: {exc} — skipping response removal.")
                remove_response = False

    print("Processing waveforms...")
    processed_stream = deconvolve_and_filter(
        combined_stream,
        inventory=inventory,
        remove_response=remove_response,
        prefilter_kwargs=prefilter_kwargs,
        filter_kwargs=filter_kwargs,
        target_sr=args.max_frequency,
    )

    # Write preprocessed daily mseed (audit trail)
    daily_output_dir = args.output_dir / f"{args.event_name}_daily"
    daily_output_dir.mkdir(parents=True, exist_ok=True)
    daily_mseed = daily_output_dir / "preprocessed.mseed"
    write_window(processed_stream, daily_mseed)
    print(f"Daily preprocessed mseed written to {daily_mseed}")

    # Export event h5
    event_output_dir = args.output_dir / "events"
    event_output_dir.mkdir(parents=True, exist_ok=True)
    h5_name = f"{args.event_name}_noise_filt_20_50_1hz"
    h5_path = event_output_dir / f"{h5_name}.h5"

    export_to_sbi_h5(
        stream=processed_stream,
        receivers=available_stations,
        event_window=(event_start, event_end),
        out_path=h5_path,
        sampling_rate=args.max_frequency,
        covariance_window=cov_window,
        full_auto_correlation=True,
    )
    print(f"Event h5 written to {h5_path}")


if __name__ == "__main__":
    main()
