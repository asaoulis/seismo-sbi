"""Build an SBI event + noise catalogue from a QuakeML catalogue file.

This is the top-level entry-point for large-scale data preparation.  Given a
downloaded QuakeML file and a directory of mseed data it produces:

  <output_dir>/events/<event_id>.h5  — one file per event in the catalogue
  <output_dir>/noise/<label>.h5      — one file per clean noise window
  <output_dir>/errors.csv            — per-window failures for resumability

Usage
-----
    python build_catalogue.py \
        --catalogue events.xml \
        --data_dir /data/raw \
        --stationxml_dir /data/stationxml \
        --stations_file configs/stations.txt \
        --output_dir /data/catalogue \
        --duration 200 \
        --sampling_rate 1.0 \
        --noise_start 2024-01-01 \
        --noise_end   2024-02-01 \
        --n_jobs 8

    # Alternatively, query FDSN for events instead of loading a file:
    python build_catalogue.py \
        --fdsn_query_center 35.7,-117.5 \
        --fdsn_min_magnitude 4.0 \
        --fdsn_client IRIS \
        ...

All parallelism is via joblib; set --n_jobs 1 for serial execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import obspy

from seismo_sbi.data_handling.preprocessing.catalogue import (
    build_event_catalogue,
    build_noise_catalogue,
    read_stations_file,
)


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--catalogue", type=Path,
                     help="Path to a QuakeML / ObsPy-readable catalogue file.")
    src.add_argument("--fdsn_query_center", type=str,
                     help="'lat,lon' for FDSN event query (requires --fdsn_* opts).")

    p.add_argument("--fdsn_client", default="IRIS",
                   help="FDSN client name when using --fdsn_query_center.")
    p.add_argument("--fdsn_min_magnitude", type=float, default=4.0)
    p.add_argument("--fdsn_max_radius_deg", type=float, default=90.0)
    p.add_argument("--fdsn_min_radius_deg", type=float, default=0.0)

    p.add_argument("--data_dir", type=Path, required=True,
                   help="Root mseed data directory.")
    p.add_argument("--stationxml_dir", type=Path,
                   help="Directory with StationXML files (optional).")
    p.add_argument("--stations_file", type=Path, required=True,
                   help="stations.txt: 'network station' per line.")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output directory for h5 files.")

    p.add_argument("--duration", type=float, default=200.0,
                   help="Event/noise window length in seconds (default 200).")
    p.add_argument("--sampling_rate", type=float, default=1.0,
                   help="Target sampling rate in Hz (default 1.0).")
    p.add_argument("--covariance_window", type=float, default=200.0,
                   help="Pre-event covariance window in seconds (default 200).")

    p.add_argument("--noise_start", type=str,
                   help="Noise period start ISO date, e.g. 2024-01-01.")
    p.add_argument("--noise_end", type=str,
                   help="Noise period end ISO date, e.g. 2024-02-01.")
    p.add_argument("--buffer_minutes", type=float, default=20.0,
                   help="Gap between noise windows and from event edges (minutes).")
    p.add_argument("--no_noise", action="store_true",
                   help="Skip noise catalogue generation.")
    p.add_argument("--no_events", action="store_true",
                   help="Skip event catalogue generation.")

    p.add_argument("--n_jobs", type=int, default=4,
                   help="Parallel workers (default 4).")
    p.add_argument("--processed_dir", type=Path, default=None,
                   help="Shared daily-processed cache directory.  Point both the "
                        "event and noise builds at the SAME path so each "
                        "(station, calendar-day) is response-removed/filtered/"
                        "resampled only once (the daily cache is resumable — "
                        "existing days are skipped).  Defaults to "
                        "<output_dir>/{events,noise}/_daily (i.e. NOT shared, so "
                        "overlapping days get processed twice).")
    p.add_argument("--channel_glob", default="BH?",
                   help="Channel glob pattern (default 'BH?').")
    p.add_argument("--taup_model", default="prem",
                   help="TauPy earth model for arrival windows (default 'prem').")
    p.add_argument("--use_taup", action="store_true", default=False,
                   help="Use TauPy to compute precise arrival windows for event "
                        "avoidance in the noise catalogue (default: off — use "
                        "onset time directly, suitable for local/regional events).")
    p.add_argument("--pre_event_window", type=float, default=0.0,
                   help="Start event windows this many seconds before the origin "
                        "time (default 0). Useful when filtering shifts the onset.")
    p.add_argument("--rolling_window_gap", type=float, default=30.0,
                   help="Step in seconds between consecutive noise windows "
                        "(default 30). Use a larger value for sparser catalogues.")
    p.add_argument("--min_completeness", type=float, default=0.9,
                   help="Minimum per-trace completeness fraction (default 0.9).")

    p.add_argument("--filter", dest="filter_kwargs", type=json.loads, default=None,
                   metavar="JSON",
                   help="JSON dict of bandpass filter overrides passed to "
                        "deconvolve_and_filter (merged with defaults "
                        "freqmin=0.02, freqmax=0.05, corners=4, zerophase=false). "
                        "Example: '{\"freqmin\": 0.06, \"freqmax\": 0.2}'")
    p.add_argument("--prefilter", dest="prefilter_kwargs", type=json.loads, default=None,
                   metavar="JSON",
                   help="JSON dict of instrument-response pre-filter overrides "
                        "(merged with defaults pre_filt=[0.005,0.01,0.1,0.2], "
                        "taper=true, taper_fraction=0.05). "
                        "Example: '{\"pre_filt\": [0.01, 0.03, 1.0, 2.0]}'")

    return p.parse_args()


def main():
    args = _parse_args()

    station_networks = read_stations_file(args.stations_file)
    if not station_networks:
        sys.exit("ERROR: No stations found in stations file.")

    print(f"Stations: {len(station_networks)}")

    # ── Load catalogue ────────────────────────────────────────────────────
    if args.catalogue:
        print(f"Loading catalogue from {args.catalogue}…")
        events = obspy.read_events(str(args.catalogue))
    elif args.fdsn_query_center:
        lat_s, lon_s = args.fdsn_query_center.split(",")
        lat, lon = float(lat_s), float(lon_s)
        print(f"Querying FDSN ({args.fdsn_client}) for events near {lat:.2f},{lon:.2f}…")
        client = obspy.clients.fdsn.Client(args.fdsn_client)
        events = client.get_events(
            latitude=lat, longitude=lon,
            minradius=args.fdsn_min_radius_deg,
            maxradius=args.fdsn_max_radius_deg,
            minmagnitude=args.fdsn_min_magnitude,
        )
    else:
        sys.exit("ERROR: Provide either --catalogue or --fdsn_query_center.")

    print(f"Catalogue: {len(events)} events")

    error_log = args.output_dir / "errors.csv"

    # ── Event catalogue ───────────────────────────────────────────────────
    if not args.no_events:
        events_dir = args.output_dir / "events"
        print(f"Building event catalogue → {events_dir}")
        written = build_event_catalogue(
            events=events,
            data_dir=args.data_dir,
            stationxml_dir=args.stationxml_dir,
            station_networks=station_networks,
            output_dir=events_dir,
            duration_s=args.duration,
            sampling_rate=args.sampling_rate,
            covariance_window_s=args.covariance_window,
            pre_event_window_s=args.pre_event_window,
            prefilter_kwargs=args.prefilter_kwargs,
            filter_kwargs=args.filter_kwargs,
            channel_glob=args.channel_glob,
            min_completeness=args.min_completeness,
            n_jobs=args.n_jobs,
            error_log=error_log,
            processed_dir=args.processed_dir,
        )
        print(f"  {len(written)} event h5 files written.")

    # ── Noise catalogue ───────────────────────────────────────────────────
    if not args.no_noise:
        if not args.noise_start or not args.noise_end:
            sys.exit("ERROR: --noise_start and --noise_end required for noise catalogue.")

        noise_start = datetime.fromisoformat(args.noise_start)
        noise_end = datetime.fromisoformat(args.noise_end)
        noise_dir = args.output_dir / "noise"
        print(f"Building noise catalogue → {noise_dir}")
        written = build_noise_catalogue(
            noise_start=noise_start,
            noise_end=noise_end,
            interfering_events=events,
            data_dir=args.data_dir,
            stationxml_dir=args.stationxml_dir,
            station_networks=station_networks,
            output_dir=noise_dir,
            duration_s=args.duration,
            sampling_rate=args.sampling_rate,
            prefilter_kwargs=args.prefilter_kwargs,
            filter_kwargs=args.filter_kwargs,
            channel_glob=args.channel_glob,
            buffer_minutes=args.buffer_minutes,
            min_completeness=args.min_completeness,
            taup_model=args.taup_model,
            use_taup=args.use_taup,
            rolling_window_gap_s=args.rolling_window_gap,
            n_jobs=args.n_jobs,
            error_log=error_log,
            processed_dir=args.processed_dir,
        )
        print(f"  {len(written)} noise h5 files written.")


if __name__ == "__main__":
    main()
