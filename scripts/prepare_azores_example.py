"""Download and prepare the Azores example data: the event waveform and a screened noise pool.

The ``PM`` network (IPMA/CIVISA) is served only by IPMA's FDSN node (``--provider``). Each window is
response-removed to displacement, band-passed and resampled as the synthetics are, and written to
``<output_dir>/events/`` and ``<output_dir>/noise/``; the noise windows that hold no earthquake are
copied to ``<output_dir>/noise_screened/``. BART (a Lennartz LE-3D/20s) records instrument noise at
25-50 s, so ``azores_components.json`` gives it no components.
"""
import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from seismo_sbi.data_handling.preprocessing.noise_windows import quiet_window_mask
# SimulationSaver guarantees the on-disk schema the pipeline reads.
from seismo_sbi.simulators.simulation_io import SimulationSaver
from seismo_sbi.utils.seismograms import compute_data_vector_length

# Channel renaming on disk: vertical -> Z, east -> 1, north -> 2.
_COMPONENT_OF_CHANNEL = {"Z": "Z", "E": "1", "N": "2", "1": "1", "2": "2"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent.parent / "examples/data/azores",
                   help="Base output dir; writes events/ and noise/ underneath.")
    p.add_argument("--stations_file", type=Path,
                   default=Path(__file__).resolve().parent / "configs/azores/azores_stations.txt",
                   help="STATION NETWORK LAT LON (lines starting with # are skipped).")
    p.add_argument("--provider", type=str, default="http://ceida.ipma.pt",
                   help="FDSN base URL or named provider hosting the PM network.")
    p.add_argument("--channel", type=str, default="HH?", help="Channel glob to request.")
    p.add_argument("--event_time", type=str, default="2022-01-13T06:46:12",
                   help="Event origin time (ISO).")
    p.add_argument("--event_lead", type=float, default=60.0,
                   help="Seconds the event window leads the origin (matches the synthetic time axis; "
                        "the legacy pipeline used a 60 s pre-origin lead).")
    p.add_argument("--duration", type=float, default=900.0, help="Window length in seconds.")
    p.add_argument("--sampling_rate", type=float, default=1.0, help="Target sampling rate (Hz).")
    p.add_argument("--freqmin", type=float, default=0.02, help="Band-pass low corner (Hz).")
    p.add_argument("--freqmax", type=float, default=0.04, help="Band-pass high corner (Hz).")
    p.add_argument("--corners", type=int, default=4, help="Band-pass corners.")
    p.add_argument("--cov_window", type=float, default=900.0,
                   help="Seconds before each window used for the noise (misc) autocovariance.")
    p.add_argument("--noise_start", type=str, default="2022-01-10", help="Noise period start (date).")
    p.add_argument("--noise_end", type=str, default="2022-01-13", help="Noise period end (date, exclusive of event-day tail).")
    p.add_argument("--cadence", type=float, default=900.0, help="Spacing between noise window starts (s).")
    p.add_argument("--num_noise", type=int, default=300, help="Cap on number of noise windows produced.")
    p.add_argument("--event_buffer", type=float, default=3600.0,
                   help="Skip noise windows within this many seconds of the event.")
    p.add_argument("--n_jobs", type=int, default=8, help="Parallel workers (one per day).")
    p.add_argument("--event_only", action="store_true", help="Only write the event file (skip noise dataset).")
    p.add_argument("--force", action="store_true", help="Regenerate even if the outputs already exist.")
    p.add_argument("--timeout", type=float, default=120.0, help="FDSN client timeout (s).")
    return p.parse_args()


def load_stations(path):
    stations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                stations.append((parts[0], parts[1]))  # (station, network)
    return stations


def process_stream(st, inv, args):
    """Response-removal -> taper -> band-pass -> resample."""
    st = st.merge(method=0, fill_value="latest")
    pre_filt = [args.freqmin / 4, args.freqmin / 2, args.freqmax * 2.5, args.freqmax * 5]
    st.remove_response(inventory=inv, output="DISP", pre_filt=pre_filt,
                       taper=True, taper_fraction=0.05)
    st.taper(max_percentage=0.01, type="cosine")
    st.filter("bandpass", freqmin=args.freqmin, freqmax=args.freqmax, corners=args.corners, zerophase=False)
    st.resample(args.sampling_rate)
    return st


def autocovariance(data, n):
    """Length-normalised autocovariance (lags 0..n-1), as in ``export_to_sbi_h5``."""
    ac = np.correlate(data, data, mode="full")
    full = ac[: data.shape[0]][::-1] / np.arange(data.shape[0], 0, -1)
    out = np.zeros(n)
    m = min(n, full.shape[0])
    out[:m] = full[:m]
    return out


def slice_to_length(tr, start, npts, sr):
    """Return exactly npts samples starting at `start` (UTCDateTime), zero-padded if short."""
    seg = tr.slice(start, start + (npts - 1) / sr).copy()
    arr = np.asarray(seg.data, dtype=np.float64)
    if arr.shape[0] >= npts:
        return arr[:npts]
    out = np.zeros(npts)
    out[: arr.shape[0]] = arr
    return out


def build_window_h5(processed, stations_components, w_start, args, out_path):
    """Write one window (outputs + misc) using already-processed day traces."""
    npts = compute_data_vector_length(int(args.duration), args.sampling_rate) + 1
    cov_npts = compute_data_vector_length(int(args.cov_window), args.sampling_rate) + 1
    sr = args.sampling_rate
    outputs, misc = {}, {}
    for sta, comps in stations_components.items():
        if sta not in processed:
            return False, f"{sta}: no processed data"
        outputs[sta], misc[sta] = {}, {}
        for chan_last, disk_comp in comps:  # e.g. ('Z','Z'), ('E','1'), ('N','2')
            tr = processed[sta].get(disk_comp)
            if tr is None:
                return False, f"{sta}.{disk_comp}: missing trace"
            data = slice_to_length(tr, UTCDateTime(w_start), npts, sr)
            pre = slice_to_length(tr, UTCDateTime(w_start) - args.cov_window, cov_npts, sr)
            if np.var(data) <= 1e-30 or np.var(pre) <= 1e-30:
                return False, f"{sta}.{disk_comp}: dead trace"
            outputs[sta][disk_comp] = data
            misc[sta][disk_comp] = autocovariance(pre, npts)
    SimulationSaver(output_data=outputs, misc_data=misc).dump_data_as_hdf5(out_path)
    return True, "ok"


def comps_for_station(sta):
    """(channel_last_letter, disk_component) pairs; vertical+horizontals."""
    return [("Z", "Z"), ("E", "1"), ("N", "2")]


def download_and_process_day(day, stations, inv_map, args):
    """Download + process all stations for one calendar day; return {station: {Z/1/2: Trace}}."""
    client = Client(base_url=args.provider, timeout=args.timeout)
    pad = args.cov_window + 600.0  # cover the pre-window + filter edge effects
    day_start = UTCDateTime(datetime.combine(day, datetime.min.time()))
    t0 = day_start - pad
    t1 = day_start + 86400 + pad
    net = stations[0][1]
    processed = {}
    for sta, network in stations:
        try:
            st = client.get_waveforms(network=network, station=sta, location="*",
                                      channel=args.channel, starttime=t0, endtime=t1)
        except Exception as e:  # noqa: BLE001
            print(f"  [{day}] {sta}: download failed ({repr(e)[:80]})", flush=True)
            continue
        inv = inv_map.get(sta)
        if inv is None or len(st) == 0:
            continue
        try:
            st = process_stream(st, inv, args)
        except Exception as e:  # noqa: BLE001
            print(f"  [{day}] {sta}: processing failed ({repr(e)[:80]})", flush=True)
            continue
        chans = {}
        for tr in st:
            last = tr.stats.channel[-1].upper()
            if last in _COMPONENT_OF_CHANNEL:
                chans[_COMPONENT_OF_CHANNEL[last]] = tr
        if chans:
            processed[sta] = chans
    return processed


def write_screened_noise_pool(noise_dir, screened_dir):
    """Copy the windows of ``noise_dir`` that hold no earthquake into ``screened_dir``."""
    windows = sorted(noise_dir.glob("*.h5"))
    with h5py.File(windows[0], "r") as first_window:
        stations = sorted(first_window["outputs"])
    vertical_rms = np.zeros((len(windows), len(stations)))
    for row, window in enumerate(windows):
        with h5py.File(window, "r") as traces:
            vertical_rms[row] = [np.sqrt(np.mean(traces["outputs"][station]["Z"][()] ** 2)) for station in stations]
    screened_dir.mkdir(parents=True, exist_ok=True)
    for stale in screened_dir.glob("*.h5"):
        stale.unlink()
    keep = quiet_window_mask(vertical_rms)
    for window in (window for window, kept in zip(windows, keep) if kept):
        shutil.copy2(window, screened_dir / window.name)
    print(f"Screened noise pool: kept {keep.sum()} of {len(windows)} windows in {screened_dir}", flush=True)


def main():
    args = parse_args()
    download_event_and_noise(args)
    screened_dir = args.output_dir / "noise_screened"
    if not args.event_only and (args.force or not any(screened_dir.glob("*.h5"))):
        write_screened_noise_pool(args.output_dir / "noise", screened_dir)


def download_event_and_noise(args):
    """Write the event file and the noise windows, unless they are already on disk."""
    stations = load_stations(args.stations_file)
    stations_components = {sta: comps_for_station(sta) for sta, _ in stations}
    event_time = datetime.fromisoformat(args.event_time)
    events_dir = args.output_dir / "events"
    noise_dir = args.output_dir / "noise"
    events_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    ev_path = events_dir / "azores_event_event_filtered_1hz.h5"
    existing_noise = len(list(noise_dir.glob("*.h5")))
    if not args.force and ev_path.exists() and (args.event_only or existing_noise >= args.num_noise):
        print(f"Already prepared: {ev_path.name} + {existing_noise} noise files present. "
              f"Use --force to regenerate.", flush=True)
        return

    print(f"Provider: {args.provider}; {len(stations)} stations; band {args.freqmin}-{args.freqmax} Hz", flush=True)

    # One response inventory per station for the whole period (cached, picklable).
    client = Client(base_url=args.provider, timeout=args.timeout)
    period_start = UTCDateTime(args.noise_start) - args.cov_window - 600
    period_end = UTCDateTime(event_time) + args.duration + 600
    inv_map = {}
    for sta, network in stations:
        try:
            inv_map[sta] = client.get_stations(network=network, station=sta, location="*",
                                               channel=args.channel, level="response",
                                               starttime=period_start, endtime=period_end)
        except Exception as e:  # noqa: BLE001
            print(f"  {sta}: response unavailable ({repr(e)[:80]})", flush=True)
    print(f"Got responses for {len(inv_map)}/{len(stations)} stations", flush=True)

    # ---- Event file ----
    event_start = event_time - timedelta(seconds=args.event_lead)
    event_day = event_start.date()
    print(f"Processing event day {event_day} (window start {event_start}, {args.event_lead}s lead) ...", flush=True)
    processed = download_and_process_day(event_day, stations, inv_map, args)
    ev_path = events_dir / "azores_event_event_filtered_1hz.h5"
    ok, msg = build_window_h5(processed, stations_components, event_start, args, ev_path)
    print(f"Event file: {'WROTE ' + str(ev_path) if ok else 'FAILED (' + msg + ')'}", flush=True)
    if not ok:
        sys.exit(2)

    if args.event_only:
        return
    write_noise_windows(args, stations, stations_components, inv_map, processed, event_day)


def write_noise_windows(args, stations, stations_components, inv_map, processed, event_day):
    """Write one noise window every ``--cadence`` seconds of the noise period, clear of the event."""
    event_time = datetime.fromisoformat(args.event_time)
    noise_dir = args.output_dir / "noise"
    d0 = datetime.fromisoformat(args.noise_start).date()
    d1 = datetime.fromisoformat(args.noise_end).date()
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

    from joblib import Parallel, delayed

    def do_day(day):
        proc = processed if day == event_day else download_and_process_day(day, stations, inv_map, args)
        written = 0
        day_start = datetime.combine(day, datetime.min.time())
        t = day_start
        end = day_start + timedelta(seconds=86400)
        while t + timedelta(seconds=args.duration) <= end:
            if abs((t - event_time).total_seconds()) < args.event_buffer:
                t += timedelta(seconds=args.cadence)
                continue
            out_path = noise_dir / (t.strftime("%Y.%m.%d.%H.%M") + ".h5")
            if not out_path.exists():
                ok, _ = build_window_h5(proc, stations_components, t, args, out_path)
                if ok:
                    written += 1
            t += timedelta(seconds=args.cadence)
        print(f"  [{day}] wrote {written} noise windows", flush=True)
        return written

    results = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(do_day)(day) for day in days)
    total = sum(results)
    # Trim to num_noise if we overshot (keep deterministic subset).
    existing = sorted(noise_dir.glob("*.h5"))
    if len(existing) > args.num_noise:
        step = len(existing) / args.num_noise
        keep = {existing[int(i * step)] for i in range(args.num_noise)}
        for f in existing:
            if f not in keep:
                f.unlink()
    print(f"Noise dataset: {len(sorted(noise_dir.glob('*.h5')))} files in {noise_dir} (produced {total})", flush=True)


if __name__ == "__main__":
    main()
