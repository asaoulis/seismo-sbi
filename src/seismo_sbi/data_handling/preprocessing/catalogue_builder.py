"""Build SBI event and noise catalogues from mseed and QuakeML data.

With ``use_daily_processing`` (the default) all raw data in the date range is response-removed,
filtered and resampled in daily chunks, written under ``<output_dir>/_daily/``, and the event
and noise windows are then sliced from those files with no further processing. Each station-day
is processed once however many windows fall in it, the taper needed before response removal only
touches the ends of each daily file, and existing daily files are skipped on a re-run.
"""

from __future__ import annotations

import csv
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

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
from seismo_sbi.data_handling.preprocessing.windowing import (
    compute_event_arrival_windows,
    get_continuous_regions,
    make_noise_windows,
)
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


def build_noise_catalogue(
    noise_start: datetime,
    noise_end: datetime,
    interfering_events,
    data_dir: Path,
    stationxml_dir: Optional[Path],
    station_networks: dict,
    output_dir: Path,
    duration_s: float,
    sampling_rate: float,
    prefilter_kwargs: Optional[dict] = None,
    filter_kwargs: Optional[dict] = None,
    channel_glob: str = "BH?",
    buffer_minutes: float = 20.0,
    min_completeness: float = 0.9,
    max_flat_fraction: float = 0.05,
    taup_model: str = "prem",
    use_taup: bool = False,
    rolling_window_gap_s: float = 30.0,
    n_jobs: int = 1,
    receivers=None,
    error_log: Optional[Path] = None,
    use_daily_processing: bool = True,
    processed_dir: Optional[Path] = None,
) -> List[Path]:
    """Export one HDF5 file per event-free noise window; returns the paths written.

    :param noise_start: start of the search period.
    :param noise_end: end of the search period.
    :param interfering_events: ``obspy.Catalog`` of events to avoid (may be empty).
    :param data_dir: root mseed directory.
    :param stationxml_dir: StationXML directory, or None.
    :param station_networks: ``{station: network}``.
    :param output_dir: destination directory, created if absent.
    :param duration_s: noise window length in seconds.
    :param sampling_rate: target sampling rate in Hz.
    :param prefilter_kwargs: passed to :func:`deconvolve_and_filter`.
    :param filter_kwargs: passed to :func:`deconvolve_and_filter`.
    :param channel_glob: mseed channel glob.
    :param buffer_minutes: gap left at the start and end of each event-free region, in minutes.
    :param min_completeness: minimum sample completeness per trace.
    :param max_flat_fraction: maximum fraction of consecutive identical samples per trace, 0-1
        (default 0.05).
    :param taup_model: TauPy model name, used when ``use_taup``.
    :param use_taup: compute arrival windows with TauPy for event avoidance; False (default) uses
        the onset time, which suits local and regional catalogues.
    :param rolling_window_gap_s: step between consecutive noise windows in seconds (default 30);
        ``duration_s`` gives non-overlapping windows.
    :param n_jobs: parallel workers.
    :param receivers: ``Receivers`` with coordinates for the TauPy distances (``use_taup`` only);
        None uses a zero-latitude, zero-longitude station.
    :param error_log: path to append failures to; None logs nothing.
    :param use_daily_processing: process the raw data in daily files first (default True).
    :param processed_dir: where the daily files live; defaults to ``output_dir / "_daily"``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = timedelta(seconds=duration_s)
    cov_window = duration

    eff_data_dir, inventory, remove_resp, eff_pre, eff_filt = _setup_data_source(
        events=None,  # not used for noise
        data_dir=data_dir,
        stationxml_dir=stationxml_dir,
        station_networks=station_networks,
        output_dir=output_dir,
        processed_dir=processed_dir,
        use_daily_processing=use_daily_processing,
        t_start=noise_start,
        t_end=noise_end,
        prefilter_kwargs=prefilter_kwargs,
        filter_kwargs=filter_kwargs,
        sampling_rate=sampling_rate,
        channel_glob=channel_glob,
        n_jobs=n_jobs,
    )

    # Build event-free continuous regions
    continuous_regions = _compute_continuous_regions(
        interfering_events, noise_start, noise_end,
        station_networks, receivers, taup_model, n_jobs,
        use_taup=use_taup, duration_s=duration_s,
        buffer_minutes=buffer_minutes,
    )

    noise_windows = list(make_noise_windows(
        continuous_regions,
        window_length=duration,
        buffer=timedelta(minutes=buffer_minutes),
        step=timedelta(seconds=rolling_window_gap_s),
    ))

    def _process_window(t_start, t_end):
        label = t_start.strftime("%Y.%m.%d.%H.%M.%S")
        out_path = output_dir / f"{label}.h5"
        if out_path.exists():
            return out_path, True, "already_exists"

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
                full_auto_correlation=False,
            )
            return out_path, True, ""
        except Exception:
            return out_path, False, traceback.format_exc().splitlines()[-1]

    results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_process_window)(t_start, t_end)
        for t_start, t_end in noise_windows
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


def _compute_continuous_regions(
    interfering_events, noise_start, noise_end,
    station_networks, receivers, taup_model, n_jobs,
    use_taup: bool = False,
    duration_s: float = 0.0,
    buffer_minutes: float = 20.0,
):
    """Return event-free continuous regions within [noise_start, noise_end].

    When use_taup=False (default), builds unavailability windows from each
    event's onset time directly — no TauPy model needed.  Each event occupies
    [origin_time, origin_time + duration_s].  The buffer applied by
    make_noise_windows ensures adequate separation from these windows.

    When use_taup=True, uses TauPy to compute precise first/last arrival
    times across all stations and adds the default 5-minute padding.
    """
    if len(interfering_events) == 0:
        return [(noise_start, noise_end)]

    if use_taup:
        _receivers = receivers if receivers is not None else _dummy_receivers(station_networks)
        unavail = compute_event_arrival_windows(
            interfering_events, _receivers,
            taup_model=taup_model, n_jobs=n_jobs,
        )
    else:
        unavail = _simple_event_windows(
            interfering_events, duration_s=duration_s,
        )

    if not unavail:
        return [(noise_start, noise_end)]

    continuous_regions, _ = get_continuous_regions(unavail, noise_start, noise_end)
    return continuous_regions


def _simple_event_windows(interfering_events, duration_s: float) -> List[Tuple]:
    """Build unavailability windows from onset times alone (no TauPy).

    Each window spans [origin_time, origin_time + duration_s].
    Suitable for local/regional events where travel times are negligible.
    """
    windows = []
    for ev in interfering_events:
        t0 = ev.origins[0].time
        windows.append((t0, t0 + duration_s))
    return windows


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


def _dummy_receivers(station_networks):
    """Minimal Receivers with zero lat/lon for TauPy distance calculations."""
    from seismo_sbi.simulators.receivers import Receiver, Receivers
    return Receivers(receivers=[
        Receiver(0.0, 0.0, net, sta, ["Z"])
        for sta, net in station_networks.items()
    ])


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
