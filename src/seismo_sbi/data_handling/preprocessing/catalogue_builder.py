"""Build an SBI event catalogue from mseed and QuakeML data.

With ``use_daily_processing`` (the default) all raw data in the date range is response-removed,
filtered and resampled in daily chunks, written under ``<output_dir>/_daily/``, and the event
and noise windows are then sliced from those files with no further processing. Each station-day
is processed once however many windows fall in it, the taper needed before response removal only
touches the ends of each daily file, and existing daily files are skipped on a re-run. The noise
catalogue (:mod:`~seismo_sbi.data_handling.preprocessing.noise_catalogue`) shares these steps.
"""

from __future__ import annotations

import csv
import traceback
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import joblib
import obspy

from seismo_sbi.data_handling.preprocessing.io import (
    find_mseed_files,
    load_waveforms,
    load_inventory,
)
from seismo_sbi.data_handling.preprocessing.processing import deconvolve_and_filter
from seismo_sbi.data_handling.preprocessing.sbi_export import export_to_sbi_h5
from seismo_sbi.data_handling.preprocessing.quality import partition_window_quality
from seismo_sbi.data_handling.preprocessing.daily import process_daily_files
from seismo_sbi.utils.seismograms import compute_data_vector_length


# --- Public API ---

def build_event_catalogue(
    events,
    data_dir: Path,
    stationxml_dir: Optional[Path],
    station_networks: dict,
    output_dir: Path,
    duration_s: float,
    sampling_rate: float,
    covariance_window_s: float = 200.0,
    pre_event_window_s: float = 60.0,
    prefilter_kwargs: Optional[dict] = None,
    filter_kwargs: Optional[dict] = None,
    channel_glob: str = "BH?",
    min_completeness: float = 0.9,
    max_flat_fraction: float = 0.05,
    n_jobs: int = 1,
    error_log: Optional[Path] = None,
    use_daily_processing: bool = True,
    processed_dir: Optional[Path] = None,
) -> List[Path]:
    """Export one HDF5 file per event of ``events``; returns the paths written.

    :param events: ``obspy.Catalog`` or a list of obspy ``Event`` objects.
    :param data_dir: root directory with per-station mseed subdirectories.
    :param stationxml_dir: StationXML response files; None skips response removal.
    :param station_networks: ``{station_code: network_code}``.
    :param output_dir: destination directory, created if absent.
    :param duration_s: event window length in seconds.
    :param sampling_rate: target sampling rate in Hz.
    :param covariance_window_s: pre-event covariance window length in seconds.
    :param pre_event_window_s: the event window starts this many seconds before the origin time;
        60 s matches the pre-origin pad of the Instaseis synthetics (``SYNTHETICS_PRE_EVENT_PAD_S``),
        so observations and synthetics align. Pass 0 only for a forward model without that pad.
    :param prefilter_kwargs: overrides for the :func:`deconvolve_and_filter` pre-filter.
    :param filter_kwargs: overrides for the :func:`deconvolve_and_filter` bandpass.
    :param channel_glob: glob for the mseed channel codes (default ``'BH?'``).
    :param min_completeness: minimum sample completeness fraction, 0-1.
    :param max_flat_fraction: maximum fraction of consecutive identical samples per trace, 0-1
        (default 0.05).
    :param n_jobs: parallel workers.
    :param error_log: path to append failures to as CSV rows; None logs nothing.
    :param use_daily_processing: process the raw data in daily files first, then slice windows
        (default True; avoids taper edge effects on long catalogues).
    :param processed_dir: where the daily files live; defaults to ``output_dir / "_daily"``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = list(events)
    if not events:
        return []

    duration = timedelta(seconds=duration_s)
    cov_window = timedelta(seconds=covariance_window_s)
    pre_event = timedelta(seconds=pre_event_window_s)

    eff_data_dir, inventory, remove_resp, eff_pre, eff_filt = _setup_data_source(
        events=events,
        data_dir=data_dir,
        stationxml_dir=stationxml_dir,
        station_networks=station_networks,
        output_dir=output_dir,
        processed_dir=processed_dir,
        use_daily_processing=use_daily_processing,
        t_start=min(ev.origins[0].time for ev in events) - covariance_window_s - 120 - pre_event_window_s,
        t_end=max(ev.origins[0].time for ev in events) + duration_s + 60,
        prefilter_kwargs=prefilter_kwargs,
        filter_kwargs=filter_kwargs,
        sampling_rate=sampling_rate,
        channel_glob=channel_glob,
        n_jobs=n_jobs,
    )

    def _process_event(event):
        origin = event.origins[0]
        event_id = _event_id(event)
        out_path = output_dir / f"{event_id}.h5"
        if out_path.exists():
            return out_path, True, "already_exists"

        t_start = origin.time.datetime - pre_event
        t_end = t_start + duration

        try:
            stream, good_stations = _load_window(
                eff_data_dir, station_networks, t_start, t_end, duration, channel_glob,
            )
            if not stream or not good_stations:
                return out_path, False, "no_data"

            proc = _prepare_stream(
                stream, use_daily_processing, inventory, remove_resp,
                eff_pre, eff_filt, sampling_rate,
            )
            kept_stations, dropped = partition_window_quality(
                proc, good_stations, sampling_rate, duration,
                min_completeness=min_completeness,
                max_flat_fraction=max_flat_fraction,
                min_npts=compute_data_vector_length(duration_s, sampling_rate) + 1,
            )
            if not kept_stations:
                return out_path, False, (
                    f"quality: all {len(good_stations)} stations dropped "
                    f"({'; '.join(r for _, r in dropped)})"
                )

            export_to_sbi_h5(
                proc,
                receivers=kept_stations,
                event_window=(t_start, t_end),
                out_path=out_path,
                sampling_rate=sampling_rate,
                covariance_window=cov_window,
                full_auto_correlation=True,
            )
            return out_path, True, ""
        except Exception:
            return out_path, False, traceback.format_exc().splitlines()[-1]

    results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_process_event)(ev) for ev in events
    )

    _write_errors(error_log, [(p.name, r) for p, ok, r in results if not ok])
    return [p for p, ok, _ in results if ok]


# --- Shared internal helpers ---

def _setup_data_source(
    events,
    data_dir: Path,
    stationxml_dir: Optional[Path],
    station_networks: dict,
    output_dir: Path,
    processed_dir: Optional[Path],
    use_daily_processing: bool,
    t_start,
    t_end,
    prefilter_kwargs: Optional[dict],
    filter_kwargs: Optional[dict],
    sampling_rate: Optional[float],
    channel_glob: str,
    n_jobs: int,
):
    """The data source the windows are sliced from: ``(data_dir, inventory, remove_response,
    prefilter_kwargs, filter_kwargs)``.

    With ``use_daily_processing`` the daily files are produced first and the response and filter
    are already applied, so the inventory and filter arguments come back None; otherwise the raw
    directory, the inventory and the filter arguments are returned for slicing to apply.
    """
    if use_daily_processing:
        daily_dir = Path(processed_dir) if processed_dir else Path(output_dir) / "_daily"
        process_daily_files(
            data_dir=data_dir,
            station_networks=station_networks,
            processed_dir=daily_dir,
            t_start=t_start,
            t_end=t_end,
            stationxml_dir=stationxml_dir,
            prefilter_kwargs=prefilter_kwargs,
            filter_kwargs=filter_kwargs,
            sampling_rate=sampling_rate,
            channel_glob=channel_glob,
            n_jobs=n_jobs,
        )
        return daily_dir, None, False, None, None
    else:
        inv = _load_inventory_safe(stationxml_dir)
        return data_dir, inv, inv is not None, prefilter_kwargs, filter_kwargs


def _prepare_stream(stream, use_daily_processing, inventory, remove_response,
                    prefilter_kwargs, filter_kwargs, sampling_rate):
    """Merge a loaded stream, applying deconvolution only when needed.

    When use_daily_processing=True, the stream comes from pre-processed daily
    files: only a merge (gap fill) is required before exporting.
    When use_daily_processing=False, apply full deconvolve_and_filter.
    """
    if use_daily_processing:
        st = stream.copy()
        st.merge(method=0, fill_value="latest")
        return st
    else:
        return deconvolve_and_filter(
            stream,
            inventory=inventory,
            remove_response=remove_response,
            prefilter_kwargs=prefilter_kwargs,
            filter_kwargs=filter_kwargs,
            target_sr=sampling_rate,
        )


def _load_inventory_safe(stationxml_dir):
    if stationxml_dir is None:
        return None
    try:
        return load_inventory(Path(stationxml_dir))
    except FileNotFoundError:
        print(
            f"WARNING: No StationXML in {stationxml_dir} — "
            "skipping response removal."
        )
        return None


def _load_window(data_dir, station_networks, t_start, t_end, duration, channel_glob):
    """Load waveforms for all stations covering [t_start - cov - pad, t_end + pad]."""
    pad_s = 60
    t0_load = t_start - timedelta(seconds=duration.total_seconds() + pad_s)
    t1_load = t_end + timedelta(seconds=pad_s)

    combined = obspy.Stream()
    good_stations = []
    for sta, network in station_networks.items():
        paths = find_mseed_files(
            data_dir, sta, t0_load, t1_load,
            network=network, channel_glob=channel_glob,
        )
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
    return combined, good_stations


def _event_id(event) -> str:
    """Derive a filesystem-safe identifier from an obspy Event."""
    t = event.origins[0].time
    return t.strftime("%Y%m%dT%H%M%S")


def _write_errors(error_log, failures: list) -> None:
    if not error_log or not failures:
        return
    with open(error_log, "a", newline="") as f:
        writer = csv.writer(f)
        for name, reason in failures:
            writer.writerow([name, reason])


def read_stations_file(stations_file: Path) -> dict:
    """Read a stations.txt file → {station: network} dict.

    Expected format: one station per line, whitespace-separated columns with
    station code as the first column and network code as the second.
    Lines starting with '#' are ignored.
    """
    mapping = {}
    with open(stations_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                station, network = parts[0], parts[1]
                mapping[station] = network
    return mapping
