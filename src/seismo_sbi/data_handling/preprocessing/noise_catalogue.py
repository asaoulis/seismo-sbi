"""Build a catalogue of event-free noise windows from mseed data.

:func:`build_noise_catalogue` finds the stretches of continuous data clear of the interfering
events and exports one HDF5 file per clean window. Windows are sliced from daily-processed files
by default, as the event windows of :mod:`~seismo_sbi.data_handling.preprocessing.catalogue_builder`
are; they feed the real-noise models.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import joblib

from seismo_sbi.data_handling.preprocessing.catalogue_builder import (
    _load_window,
    _prepare_stream,
    _setup_data_source,
    _write_errors,
)
from seismo_sbi.data_handling.preprocessing.sbi_export import export_to_sbi_h5
from seismo_sbi.data_handling.preprocessing.quality import partition_window_quality
from seismo_sbi.data_handling.preprocessing.windowing import (
    compute_event_arrival_windows,
    get_continuous_regions,
    make_noise_windows,
)
from seismo_sbi.utils.seismograms import compute_data_vector_length


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


def _dummy_receivers(station_networks):
    """Minimal Receivers with zero lat/lon for TauPy distance calculations."""
    from seismo_sbi.simulators.receivers import Receiver, Receivers
    return Receivers(receivers=[
        Receiver(0.0, 0.0, net, sta, ["Z"])
        for sta, net in station_networks.items()
    ])
