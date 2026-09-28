"""Window selection utilities: event windows, noise windows, continuous regions.

All functions are pure (no file I/O, no global state). Ported from
EventWindowSelector as standalone functions so they can be used without
instantiating an FDSN client.
"""

from datetime import timedelta, datetime
from typing import Iterator, List, Optional, Tuple

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.geodetics import locations2degrees

from seismo_sbi.utils.seismograms import compute_data_vector_length



def slice_event_window(
    stream: Stream,
    t_start,
    t_end,
    sampling_rate: float,
) -> Stream:
    """Slice a ``Stream`` to an event window whose end sample is inclusive.

    The end is ``t_start + n / sr`` with ``n = compute_data_vector_length(duration, sr)`` and the
    slice is inclusive, giving ``n + 1`` samples.

    :param stream: preprocessed ``Stream`` at ``sampling_rate``.
    :param t_start: window start, ``datetime`` or ``UTCDateTime``.
    :param t_end: window end.
    :param sampling_rate: sampling rate in Hz.
    :returns: the sliced ``Stream``.
    """
    duration = (UTCDateTime(t_end) - UTCDateTime(t_start))
    n = compute_data_vector_length(duration, sampling_rate)
    exact_end = UTCDateTime(t_start) + n / sampling_rate
    return stream.slice(UTCDateTime(t_start), exact_end)


def make_noise_windows(
    continuous_regions: List[Tuple],
    window_length: timedelta,
    buffer: timedelta = timedelta(minutes=15),
    step: Optional[timedelta] = None,
) -> Iterator[Tuple[datetime, datetime]]:
    """Yield ``(start, end)`` noise windows from event-free continuous regions.

    :param continuous_regions: ``(start, end)`` pairs of event-free time.
    :param window_length: length of each noise window.
    :param buffer: gap left at the start and end of each continuous region.
    :param step: advance of the window start between consecutive windows; ``buffer`` when None,
        a small value such as ``timedelta(seconds=30)`` for a rolling window.
    """
    advance = step if step is not None else buffer
    for start, end in continuous_regions:
        if isinstance(start, UTCDateTime):
            start = start.datetime
        if isinstance(end, UTCDateTime):
            end = end.datetime
        buffered_start = start + buffer
        buffered_end = end - buffer
        while buffered_start + window_length < buffered_end:
            yield buffered_start, buffered_start + window_length
            buffered_start += advance


def make_daily_overlapping_windows(
    continuous_regions: List[Tuple],
    window_length: timedelta,
    buffer: timedelta = timedelta(minutes=15),
    overlap_offset: timedelta = None,
) -> dict:
    """Per-date lists of overlapping noise windows: ``{date: [(start, end), ...]}``."""
    if overlap_offset is None:
        overlap_offset = buffer

    daily_windows = {}
    for start, end in continuous_regions:
        if isinstance(start, UTCDateTime):
            start = start.datetime
        if isinstance(end, UTCDateTime):
            end = end.datetime
        buffered_start = start + buffer
        buffered_end = end - buffer
        while buffered_start + window_length < buffered_end:
            date = buffered_start.date()
            window_end = buffered_start + window_length
            if date not in daily_windows:
                daily_windows[date] = []
            if date == window_end.date():
                daily_windows[date].append((buffered_start, window_end))
            buffered_start += overlap_offset
    return daily_windows


def compute_event_arrival_windows(
    events,
    receivers,
    taup_model: str = "prem",
    n_jobs: int = 1,
    padding: timedelta = timedelta(minutes=5),
) -> List[Tuple]:
    """``(start, end)`` unavailability windows, one per event, from the earliest and latest TauPy
    arrivals over all stations padded by ``padding`` on both sides.

    :param events: ``obspy.Catalog`` or a list of obspy ``Event`` objects.
    :param receivers: a ``Receivers`` object, or a list of ``(lat, lon)`` tuples.
    :param taup_model: TauPy Earth model name (default ``'prem'``).
    :param n_jobs: parallel jobs (1 = serial).
    :param padding: time added around each arrival window (default 5 min).
    :returns: ``(start, end)`` ``UTCDateTime`` pairs; an event with no computable arrival is skipped.
    """
    from obspy.taup import TauPyModel
    import joblib

    model = TauPyModel(model=taup_model)

    # Normalise receivers to list of (lat, lon)
    if hasattr(receivers, "receivers"):
        station_coords = [
            (r.latitude, r.longitude) for r in receivers.receivers
        ]
    else:
        station_coords = list(receivers)

    pad_s = padding.total_seconds()

    def _one_event(event):
        origin = event.origins[0]
        event_lat = origin.latitude
        event_lon = origin.longitude
        depth_km = origin.depth / 1000.0
        origin_time = origin.time

        earliest = np.inf
        latest = -np.inf
        for lat, lon in station_coords:
            dist_deg = locations2degrees(event_lat, event_lon, lat, lon)
            try:
                arrivals = model.get_travel_times(
                    source_depth_in_km=depth_km,
                    distance_in_degree=dist_deg,
                )
            except Exception:
                continue
            if arrivals:
                earliest = min(earliest, arrivals[0].time)
                latest = max(latest, arrivals[-1].time)

        if earliest == np.inf:
            return None
        return (
            origin_time + earliest - pad_s,
            origin_time + latest + pad_s,
        )

    if n_jobs == 1:
        results = [_one_event(ev) for ev in events]
    else:
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_one_event)(ev) for ev in events
        )

    return [r for r in results if r is not None]


def filter_events_by_distance(
    events,
    center: Tuple[float, float],
    min_radius_deg: Optional[float] = None,
    max_radius_deg: Optional[float] = None,
) -> list:
    """The events within an epicentral-distance range of a point.

    :param events: ``obspy.Catalog`` or a list of obspy ``Event`` objects.
    :param center: ``(latitude, longitude)`` of the reference point in degrees.
    :param min_radius_deg: minimum epicentral distance in degrees, inclusive.
    :param max_radius_deg: maximum epicentral distance in degrees, inclusive.
    :returns: the filtered list of ``Event`` objects.
    """
    center_lat, center_lon = center
    filtered = []
    for ev in events:
        origin = ev.origins[0]
        dist_deg = locations2degrees(
            origin.latitude, origin.longitude,
            center_lat, center_lon,
        )
        if min_radius_deg is not None and dist_deg < min_radius_deg:
            continue
        if max_radius_deg is not None and dist_deg > max_radius_deg:
            continue
        filtered.append(ev)
    return filtered


def get_continuous_regions(
    event_start_end_times: List[Tuple],
    start_time,
    end_time,
) -> Tuple[List[Tuple], List[Tuple]]:
    """``(continuous_regions, gaps)``: the event-free regions between a set of event windows, and the event intervals themselves."""
    continuous_regions = []
    gaps = []

    sorted_ranges = sorted(event_start_end_times, key=lambda x: x[0])
    current_start, current_end = sorted_ranges[0]
    continuous_regions.append((start_time, current_start))

    for start, end in sorted_ranges[1:]:
        if start > current_end:
            continuous_regions.append((current_end, start))
            gaps.append((current_start, current_end))
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)

    gaps.append((current_start, current_end))
    continuous_regions.append((current_end, end_time))
    return continuous_regions, gaps
