"""Build SBI event and noise catalogues from mseed + QuakeML data.

These are the importable library functions.  The CLI entry-point is
``scripts/build_catalogue.py``, which is a thin wrapper around these.

Daily-processing mode (enabled by default)
------------------------------------------
Both ``build_event_catalogue`` and ``build_noise_catalogue`` accept
``use_daily_processing=True`` (default).  When enabled:

  1. All raw data in the relevant date range is processed in daily chunks
     (response removal, filtering, resampling) and saved as intermediate
     mseed files under ``<output_dir>/_daily/`` (overridable via
     ``processed_dir``).
  2. Individual event / noise windows are then sliced from those pre-processed
     files — no further deconvolution or filtering needed.

Benefits:
  - Each station-day is processed once, regardless of how many windows
    fall within it.  For large catalogues this is orders of magnitude faster.
  - Tapering artefacts (required before response removal) only affect the
    very edges of each daily file; all interior windows are clean.
  - Daily files are resumable: existing files are silently skipped on re-runs.
"""

from __future__ import annotations

import csv
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import obspy
from obspy import UTCDateTime

from seismo_sbi.data_handling.preprocessing.io import (
    find_mseed_files,
    load_waveforms,
    load_inventory,
)
from seismo_sbi.data_handling.preprocessing.processing import deconvolve_and_filter
from seismo_sbi.data_handling.preprocessing.sbi_export import export_to_sbi_h5
from seismo_sbi.data_handling.preprocessing.quality import (
    check_window_quality, partition_window_quality)
from seismo_sbi.data_handling.preprocessing.windowing import (
    compute_event_arrival_windows,
    get_continuous_regions,
    make_noise_windows,
)
from seismo_sbi.data_handling.preprocessing.daily import process_daily_files
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_event_catalogue(
    events,
    data_dir: Path,
    stationxml_dir: Optional[Path],
    station_networks: dict,
    output_dir: Path,
    duration_s: float,
    sampling_rate: float,
    covariance_window_s: float = 200.0,
    pre_event_window_s: float = 60.0,   # = wrapper.SYNTHETICS_PRE_EVENT_PAD_S: sims place the
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
    """Export one SBI h5 file per event in *events*.

    Args:
        events: obspy.Catalog or list of obspy Event objects.
        data_dir: Root directory with per-station mseed subdirectories.
        stationxml_dir: Directory with StationXML response files (or None to
            skip response removal).
        station_networks: Mapping ``{station_code: network_code}``.
        output_dir: Destination directory; created if absent.
        duration_s: Event window length in seconds.
        sampling_rate: Target sampling rate (Hz).
        covariance_window_s: Pre-event covariance window length (s).
        pre_event_window_s: Start the event window this many seconds *before*
            the origin time.  DEFAULTS TO 60 s to match the pre-origin pad every
            Instaseis synthetic carries (``SyntheticsPreprocessing`` /
            ``SYNTHETICS_PRE_EVENT_PAD_S`` in ``instaseis_simulator/wrapper.py``):
            the sims place the origin at t=+60 s, so the observations MUST too or
            obs and synthetics are misaligned by 60 s (an out-of-distribution shift
            ~12x beyond the training time-shift augmentation).  Pass 0 only for a
            non-Instaseis convention.
        prefilter_kwargs: Override for ``deconvolve_and_filter`` pre-filter.
        filter_kwargs: Override for ``deconvolve_and_filter`` bandpass.
        channel_glob: Glob for mseed channel codes (default ``'BH?'``).
        min_completeness: Minimum sample completeness fraction (0–1).
        max_flat_fraction: Maximum fraction of consecutive identical samples
            allowed per trace (0–1).  Default 0.05.
        n_jobs: Number of parallel workers.
        error_log: Path to append failures as CSV rows; None = no logging.
        use_daily_processing: Process raw data in daily chunks first, then
            slice windows (default True).  Strongly recommended for large
            catalogues — much faster and avoids taper edge effects.
        processed_dir: Where to store / find daily processed files.  Defaults
            to ``output_dir / "_daily"``.

    Returns:
        List of Paths to successfully written h5 files.
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
    """Export one SBI h5 file per event-free noise window.

    Args:
        noise_start: Start of the search period.
        noise_end: End of the search period.
        interfering_events: obspy.Catalog of events to avoid (may be empty).
        data_dir: Root mseed directory.
        stationxml_dir: StationXML directory (or None).
        station_networks: ``{station: network}`` mapping.
        output_dir: Destination directory; created if absent.
        duration_s: Noise window length (s).
        sampling_rate: Target sampling rate (Hz).
        prefilter_kwargs / filter_kwargs: Passed to ``deconvolve_and_filter``.
        channel_glob: Mseed channel glob.
        buffer_minutes: Gap to leave at the start/end of each event-free
            continuous region (minutes).
        min_completeness: Minimum sample completeness per trace.
        max_flat_fraction: Maximum fraction of consecutive identical samples
            allowed per trace (0–1).  Default 0.05.
        taup_model: TauPy model name; only used when ``use_taup=True``.
        use_taup: If True, use TauPy to compute precise arrival windows for
            event avoidance.  If False (default), use the event onset time
            directly — suitable for local/regional catalogues where travel
            times are negligible.
        rolling_window_gap_s: Step (in seconds) between consecutive noise
            windows.  Defaults to 30 s, producing a dense rolling/sliding
            window catalogue.  Set equal to ``duration_s`` for non-overlapping
            windows, or to ``buffer_minutes * 60`` to match the old behaviour.
        n_jobs: Parallel workers.
        receivers: Receivers object with lat/lon for TauPy distance calc
            (only used when ``use_taup=True``).
            If None, a zero-lat/lon dummy is used for each station.
        error_log: Path to append failures; None = no logging.
        use_daily_processing: Process raw data in daily chunks first (default
            True).  Strongly recommended for large catalogues.
        processed_dir: Where to store / find daily processed files.  Defaults
            to ``output_dir / "_daily"``.

    Returns:
        List of Paths to written h5 files.
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


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

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
    """Set up the data source for window slicing.

    Returns:
        (effective_data_dir, inventory, remove_response,
         effective_prefilter_kwargs, effective_filter_kwargs)

    When use_daily_processing=True:
        - Runs process_daily_files() to populate the daily cache directory.
        - Returns processed_dir as effective_data_dir.
        - inventory and filter kwargs are None (already applied during daily step).

    When use_daily_processing=False:
        - Returns data_dir as effective_data_dir.
        - Loads the inventory and passes filter kwargs through unchanged.
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
    from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
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
