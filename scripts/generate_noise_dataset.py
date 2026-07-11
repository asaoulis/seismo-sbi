"""Generate a noise database using the new obspy-centred preprocessing API.

Processing flow (daily mode, default):
  1. For each station-day in [NOISE_START, NOISE_END], remove instrument
     response, bandpass filter, and resample → save as processed daily mseed.
  2. Query FDSN for interfering events to compute clean noise windows.
  3. Slice each noise window from the pre-processed daily files → export h5.

The daily-processing step is parallelised over (station, day) pairs, and
the slicing step over individual windows.  Both use NUM_JOBS workers.

Set USE_DAILY_PROCESSING = False to fall back to per-window processing
(re-applies full deconvolution for every window — slower but no intermediate
files on disk).
"""

from pathlib import Path
from datetime import datetime, timedelta
import csv
import traceback

import obspy
import joblib

from seismo_sbi.data_handling.preprocessing import (
    find_mseed_files,
    load_waveforms,
    load_inventory,
    deconvolve_and_filter,
    export_to_sbi_h5,
    check_window_quality,
    process_daily_files,
    build_noise_catalogue,
    build_event_catalogue,
)
from seismo_sbi.data_handling.preprocessing.windowing import (
    get_continuous_regions,
    make_noise_windows,
)
from seismo_sbi.data_handling.event_window_selection import EventWindowSelector
from seismo_sbi.instaseis_simulator.receivers import Receivers
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

from constants import STATION_CODES_PATHS_INDO, INDO_DATA_FORMAT

STATION_CODES_PATHS = STATION_CODES_PATHS_INDO

DATA_DIR = Path("/data/alex")
STATIONXML_DIR = DATA_DIR / "stationxml"
OUTPUT_DIR = Path("/data/alex/noise/indo_pacific")

PREFILTER_KWARGS = dict(pre_filt=[0.005, 0.01, 0.08, 0.1])
FILTER_KWARGS = dict(freqmin=1 / 50, freqmax=1 / 20, corners=4, zerophase=False)

DURATION = timedelta(minutes=3 + 1 / 3)   # noise window length per h5 file
MAX_FREQUENCY = 1.0                         # Hz
NUM_JOBS = 4                                # parallel worker processes
NOISE_NAME = "50_20s_200sec_samples"

USE_DAILY_PROCESSING = True                 # recommended; set False to disable

EVENT_WINDOW = [datetime(2024, 9, 13, 6, 45, 12), datetime(2024, 9, 13, 7, 0, 12)]
EVENT_LOCATION = (5, 117)

NOISE_START = datetime(2024, 9, 1)
NOISE_END = datetime(2024, 9, 20)

CHANNEL_GLOB = "BH?"
MIN_COMPLETENESS = 0.9

# ---------------------------------------------------------------------------
# Build output directories
# ---------------------------------------------------------------------------

noise_samples_dir = OUTPUT_DIR / f"{NOISE_NAME}_samples"
noise_samples_dir.mkdir(parents=True, exist_ok=True)

event_dir = OUTPUT_DIR / "events"
event_dir.mkdir(parents=True, exist_ok=True)

error_log = OUTPUT_DIR / f"{NOISE_NAME}_errors.csv"

# ---------------------------------------------------------------------------
# Build station→network mapping and detect available stations
# ---------------------------------------------------------------------------

print("Loading inventory...")
try:
    inventory_check = load_inventory(STATIONXML_DIR)
    has_inventory = True
except FileNotFoundError:
    print(f"WARNING: No StationXML in {STATIONXML_DIR} — skipping response removal.")
    has_inventory = False


def _station_has_data(station, network, t0, t1):
    return bool(find_mseed_files(DATA_DIR, station, t0, t1,
                                 network=network, channel_glob=CHANNEL_GLOB))


event_t0 = EVENT_WINDOW[0] - timedelta(minutes=5)
event_t1 = EVENT_WINDOW[1] + timedelta(minutes=1)
available_stations = []
station_networks = {}

for station, network_key in STATION_CODES_PATHS.items():
    network = network_key if len(network_key) <= 2 else "*"
    if _station_has_data(station, network, event_t0, event_t1):
        available_stations.append(station)
        station_networks[station] = network

print(f"Available stations: {len(available_stations)}")
if not available_stations:
    raise RuntimeError("No stations have data for the event window.")

# ---------------------------------------------------------------------------
# Optional: process daily files for the full noise period up front
# ---------------------------------------------------------------------------

if USE_DAILY_PROCESSING:
    processed_dir = OUTPUT_DIR / "_daily"
    print(f"Processing daily files → {processed_dir}  (NUM_JOBS={NUM_JOBS})")
    process_daily_files(
        data_dir=DATA_DIR,
        station_networks=station_networks,
        processed_dir=processed_dir,
        t_start=NOISE_START,
        t_end=NOISE_END,
        stationxml_dir=STATIONXML_DIR if has_inventory else None,
        prefilter_kwargs=PREFILTER_KWARGS,
        filter_kwargs=FILTER_KWARGS,
        sampling_rate=MAX_FREQUENCY,
        channel_glob=CHANNEL_GLOB,
        n_jobs=NUM_JOBS,
    )
    effective_data_dir = processed_dir
    remove_response = False
    use_filter_kwargs = None
    use_prefilter_kwargs = None
    inventory = None
else:
    effective_data_dir = DATA_DIR
    inventory = load_inventory(STATIONXML_DIR) if has_inventory else None
    remove_response = has_inventory
    use_filter_kwargs = FILTER_KWARGS
    use_prefilter_kwargs = PREFILTER_KWARGS

# ---------------------------------------------------------------------------
# Export the event window h5 (from pre-processed or raw data)
# ---------------------------------------------------------------------------

def _load_and_process_window(t_start, t_end):
    pad_s = 60
    t0_load = t_start - DURATION.total_seconds() - pad_s
    t1_load = t_end + pad_s

    combined = obspy.Stream()
    good_stations = []
    for sta in available_stations:
        network = station_networks.get(sta, "*")
        paths = find_mseed_files(effective_data_dir, sta, t0_load, t1_load,
                                 network=network, channel_glob=CHANNEL_GLOB)
        if not paths:
            continue
        try:
            st = load_waveforms(paths, starttime=t0_load, endtime=t1_load)
            if len(st) == 0:
                continue
            combined += st
            good_stations.append(sta)
        except Exception as exc:
            print(f"  {sta}: load error — {exc}")

    if USE_DAILY_PROCESSING:
        combined.merge(method=0, fill_value="latest")
        return combined, good_stations
    else:
        if combined:
            combined = deconvolve_and_filter(
                combined, inventory=inventory, remove_response=remove_response,
                prefilter_kwargs=use_prefilter_kwargs, filter_kwargs=use_filter_kwargs,
                target_sr=MAX_FREQUENCY,
            )
        return combined, good_stations


event_stream, event_available = _load_and_process_window(EVENT_WINDOW[0], EVENT_WINDOW[1])
if event_stream:
    data_vector_length = compute_data_vector_length(
        (EVENT_WINDOW[1] - EVENT_WINDOW[0]).total_seconds(), MAX_FREQUENCY
    ) + 1
    export_to_sbi_h5(
        event_stream,
        receivers=event_available,
        event_window=EVENT_WINDOW,
        out_path=event_dir / f"{NOISE_NAME}_event_filtered_1hz.h5",
        sampling_rate=MAX_FREQUENCY,
        covariance_window=DURATION,
        full_auto_correlation=True,
    )
    print(f"Event h5 written: data_vector_length={data_vector_length}")

# ---------------------------------------------------------------------------
# Find event-free noise windows
# ---------------------------------------------------------------------------

print("Querying FDSN for interfering events...")
from seismo_sbi.instaseis_simulator.receivers import Receiver
dummy_receivers = Receivers(receivers=[
    Receiver(0.0, 0.0, station_networks.get(s, "XX"), s, ["Z"])
    for s in available_stations
])

ews = EventWindowSelector(num_jobs=NUM_JOBS)
try:
    distant, near = ews.select_events(
        EVENT_LOCATION[0], EVENT_LOCATION[1],
        NOISE_START.strftime("%Y-%m-%d"), NOISE_END.strftime("%Y-%m-%d"),
        distant_min_radius=40, distant_min_magnitude=6.0,
        close_max_radius=20, close_min_magnitude=4.5,
    )
    unavail = ews.get_unavailability_time_pairs(dummy_receivers, near, distant)
    continuous_regions, _ = get_continuous_regions(unavail, NOISE_START, NOISE_END)
except Exception as exc:
    print(f"Event selection failed ({exc}) — using full date range as one region.")
    continuous_regions = [(NOISE_START, NOISE_END)]

noise_time_windows = list(make_noise_windows(
    continuous_regions,
    window_length=DURATION,
    buffer=timedelta(minutes=20),
))
print(f"Found {len(noise_time_windows)} candidate noise windows.")

# ---------------------------------------------------------------------------
# Parallel noise window export
# ---------------------------------------------------------------------------

def _process_one_noise_window(t_start, t_end):
    label = t_start.strftime("%Y.%m.%d.%H.%M")
    out_path = noise_samples_dir / f"{label}.h5"
    if out_path.exists():
        return label, True, "already_exists"
    try:
        stream, good_stations = _load_and_process_window(t_start, t_end)
        if not stream or not good_stations:
            return label, False, "no_data"

        ok, reason = check_window_quality(
            stream, good_stations, MAX_FREQUENCY, DURATION, MIN_COMPLETENESS
        )
        if not ok:
            return label, False, f"quality: {reason}"

        export_to_sbi_h5(
            stream,
            receivers=good_stations,
            event_window=(t_start, t_end),
            out_path=out_path,
            sampling_rate=MAX_FREQUENCY,
            covariance_window=DURATION,
            full_auto_correlation=False,
        )
        return label, True, ""
    except Exception:
        return label, False, traceback.format_exc().splitlines()[-1]


print(f"Generating noise samples (NUM_JOBS={NUM_JOBS})...")
results = joblib.Parallel(n_jobs=NUM_JOBS, backend="loky")(
    joblib.delayed(_process_one_noise_window)(t_start, t_end)
    for t_start, t_end in noise_time_windows
)

n_written = sum(1 for _, ok, r in results if ok and r != "already_exists")
n_existed = sum(1 for _, ok, r in results if r == "already_exists")
failures = [(label, r) for label, ok, r in results if not ok]

if failures:
    with open(error_log, "a", newline="") as f:
        writer = csv.writer(f)
        for label, reason in failures:
            f.write(f"{label},{reason}\n")
    print(f"  {len(failures)} failures logged to {error_log}")

print(
    f"Noise database complete: {n_written} new + {n_existed} existing = "
    f"{n_written + n_existed} total samples in {noise_samples_dir}"
)
