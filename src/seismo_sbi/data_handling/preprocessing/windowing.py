"""Window selection utilities: event windows, noise windows, continuous regions.

All functions are pure (no file I/O, no global state). Ported from
EventWindowSelector as standalone functions so they can be used without
instantiating an FDSN client.
"""

import math
from datetime import timedelta, datetime
from typing import Iterator, List, Optional, Tuple

import numpy as np
import obspy
from obspy import Stream, UTCDateTime
from obspy.geodetics import locations2degrees

from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length


def slice_event_window(
    stream: Stream,
    t_start,
    t_end,
    sampling_rate: float,
) -> Stream:
    """Slice a Stream to an event window, reproducing ProcessedDataSlicer semantics.

    The legacy pipeline does:
        fixed_num_seconds = ceil(duration / sr) * sr
        exact_end = t_start + fixed_num_seconds
        data.slice(t_start, exact_end)   # inclusive → +1 sample

    This gives compute_data_vector_length(duration, sr) + 1 samples.

    Args:
        stream: Pre-processed Stream (already at target sampling_rate).
        t_start: Window start (datetime or UTCDateTime).
        t_end: Window end (datetime or UTCDateTime).
        sampling_rate: Target sampling rate (Hz).

    Returns:
        Sliced Stream with the canonical sample count.
    """
    duration = (UTCDateTime(t_end) - UTCDateTime(t_start))
    fixed_num_seconds = math.ceil(duration / sampling_rate) * sampling_rate
    exact_end = UTCDateTime(t_start) + fixed_num_seconds
    return stream.slice(UTCDateTime(t_start), exact_end)


def make_noise_windows(
    continuous_regions: List[Tuple],
    window_length: timedelta,
    buffer: timedelta = timedelta(minutes=15),
    step: Optional[timedelta] = None,
) -> Iterator[Tuple[datetime, datetime]]:
    """Yield (start, end) noise windows from event-free continuous regions.

    Ports EventWindowSelector.create_windows_from_regions as a generator.

    Args:
        continuous_regions: List of (start, end) pairs marking event-free time.
        window_length: Length of each noise window.
        buffer: Gap to leave at the start and end of each continuous region.
        step: How far to advance the window start between consecutive windows.
            Defaults to ``buffer`` when None (original non-rolling behaviour).
            Set to a small value (e.g. ``timedelta(seconds=30)``) for a
            rolling/sliding window that densely covers the available time.
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
    """Build per-date lists of overlapping noise windows.

    Ports EventWindowSelector.create_daily_overlapping_windows_from_regions.

    Returns:
        Dict mapping date → list of (start, end) window tuples.
    """
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
    """Compute (start, end) unavailability windows for each event.

    For each event in *events*, computes the earliest and latest seismic
    arrivals across all stations in *receivers* using TauPy, then pads each
    window by *padding* on both sides.

    Args:
        events: obspy.Catalog or list of obspy Event objects.
        receivers: Receivers object (has .receivers with .latitude/.longitude)
            OR a list of (lat, lon) tuples.
        taup_model: TauPy earth model name (default 'prem').
        n_jobs: Number of parallel jobs for joblib (default 1 = serial).
        padding: Extra time to add around each arrival window (default 5 min).

    Returns:
        List of (start_UTCDateTime, end_UTCDateTime) tuples, one per event.
        Events with no computable arrivals are silently skipped.
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
    """Filter an obspy Catalog/list to events within a distance range.

    Args:
        events: obspy.Catalog or list of obspy Event objects.
        center: (latitude, longitude) of the reference point in degrees.
        min_radius_deg: Minimum epicentral distance in degrees (inclusive).
        max_radius_deg: Maximum epicentral distance in degrees (inclusive).

    Returns:
        Filtered list of Event objects.
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
    """Compute event-free continuous regions between a set of event windows.

    Ports EventWindowSelector.get_continuous_regions.

    Returns:
        (continuous_regions, gaps) where gaps are the event intervals.
    """
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
