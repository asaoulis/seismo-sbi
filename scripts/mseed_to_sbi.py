"""mseed_to_sbi.py — Convert preprocessed mseed + StationXML to seismo-sbi h5.

This is the explicit SBI-boundary tool described in Phase 2 of the refactor plan.
It consumes data that has already been downloaded and optionally processed
(e.g. by custom_download.py), deconvolves the instrument response, applies
the bandpass filter, and writes the final h5 file that RealNoiseSampler reads.

Typical workflow:
  1. custom_download.py  → raw mseed + stationxml
  2. mseed_to_sbi.py     → event h5 (for SBI)

Usage:
    python mseed_to_sbi.py \
        --data_dir /data/alex/long_valley \
        --stations_file configs/long_valley/stations.txt \
        --event_starttime 1997-11-22T17:20:35 \
        --event_endtime   1997-11-22T17:23:54 \
        --output_path     /data/alex/noise/long_valley/events/LV2.h5
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import obspy
from obspy import UTCDateTime

from seismo_sbi.data_handling.preprocessing import (
    find_mseed_files,
    load_waveforms,
    load_inventory,
    deconvolve_and_filter,
    export_to_sbi_h5,
)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert downloaded mseed + StationXML to seismo-sbi HDF5."
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Root directory containing {station}/{year}.{jday}/ mseed layout.",
    )
    parser.add_argument(
        "--stationxml_dir", type=Path, default=None,
        help="Directory with StationXML files. Defaults to <data_dir>/stationxml.",
    )
    parser.add_argument(
        "--stations_file", type=Path, required=True,
        help="Stations file with columns: STATION NETWORK [LAT LON ...]",
    )
    parser.add_argument(
        "--event_starttime", type=str, required=True,
        help="Event window start (ISO 8601, e.g. '1997-11-22T17:20:35').",
    )
    parser.add_argument(
        "--event_endtime", type=str, required=True,
        help="Event window end (ISO 8601).",
    )
    parser.add_argument(
        "--output_path", type=Path, required=True,
        help="Destination .h5 file path.",
    )
    parser.add_argument(
        "--sampling_rate", type=float, default=1.0,
        help="Target sampling rate in Hz (default: 1.0).",
    )
    parser.add_argument(
        "--covariance_minutes", type=float, default=5.0,
        help="Pre-event covariance estimation window in minutes (default: 5).",
    )
    parser.add_argument(
        "--freqmin", type=float, default=0.02,
        help="Bandpass low corner frequency (Hz).",
    )
    parser.add_argument(
        "--freqmax", type=float, default=0.05,
        help="Bandpass high corner frequency (Hz).",
    )
    parser.add_argument(
        "--channel_glob", type=str, default="BH?",
        help="Channel glob pattern (e.g. 'BH?' or 'HH?').",
    )
    parser.add_argument(
        "--no_instrument_correction", action="store_true",
        help="Skip instrument response removal.",
    )
    parser.add_argument(
        "--scalar_variance", action="store_true",
        help="Write scalar variance instead of full autocorrelation to /misc.",
    )
    return parser.parse_args()


def load_stations(stations_path: Path) -> list:
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
    cov_window = timedelta(minutes=args.covariance_minutes)
    remove_response = not args.no_instrument_correction
    stationxml_dir = args.stationxml_dir or (args.data_dir / "stationxml")

    filter_kwargs = dict(
        freqmin=args.freqmin, freqmax=args.freqmax, corners=4, zerophase=False
    )
    prefilter_kwargs = dict(pre_filt=[0.005, 0.01, args.freqmax * 4, args.freqmax * 8])
    prefilter_kwargs["pre_filt"] = [
        0.005, 0.01,
        min(args.freqmax * 4, args.sampling_rate * 0.4),
        min(args.freqmax * 8, args.sampling_rate * 0.45),
    ]

    station_pairs = load_stations(args.stations_file)
    print(f"Processing {len(station_pairs)} stations from {args.stations_file}")

    t0_load = UTCDateTime(event_start) - cov_window.total_seconds() - 120
    t1_load = UTCDateTime(event_end) + 120

    # Load waveforms
    combined = obspy.Stream()
    available_stations = []
    for station, network in station_pairs:
        paths = find_mseed_files(
            args.data_dir, station, t0_load, t1_load,
            network=network, channel_glob=args.channel_glob,
        )
        if not paths:
            continue
        try:
            st = load_waveforms(paths, starttime=t0_load, endtime=t1_load)
            if len(st) == 0:
                continue
            combined += st
            available_stations.append(station)
        except Exception as exc:
            print(f"  Skipping {network}.{station}: {exc}")

    if not available_stations:
        print("No stations available — aborting.")
        sys.exit(1)

    print(f"Loaded {len(available_stations)} stations: {available_stations}")

    # Load inventory
    inventory = None
    if remove_response:
        if stationxml_dir.is_dir():
            try:
                inventory = load_inventory(stationxml_dir)
                print(f"Inventory loaded from {stationxml_dir}")
            except FileNotFoundError as exc:
                print(f"WARNING: {exc} — skipping response removal.")
                remove_response = False
        else:
            print(f"WARNING: {stationxml_dir} not found — skipping response removal.")
            remove_response = False

    # Process
    print("Deconvolving and filtering...")
    processed = deconvolve_and_filter(
        combined,
        inventory=inventory,
        remove_response=remove_response,
        prefilter_kwargs=prefilter_kwargs,
        filter_kwargs=filter_kwargs,
        target_sr=args.sampling_rate,
    )

    # Export
    print(f"Exporting to {args.output_path}")
    export_to_sbi_h5(
        stream=processed,
        receivers=available_stations,
        event_window=(event_start, event_end),
        out_path=args.output_path,
        sampling_rate=args.sampling_rate,
        covariance_window=cov_window,
        full_auto_correlation=not args.scalar_variance,
    )
    print("Done.")


if __name__ == "__main__":
    main()
