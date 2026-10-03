"""Recorded noise windows cut from a continuous processed ``Stream``, in memory.

:func:`noise_windows_from_stream` lays each event-free window out as
:func:`~seismo_sbi.data_handling.preprocessing.sbi_export.observation_from_stream` lays out an event:
``(n_windows, n_traces * n_samples)`` rows in receiver order with a ``(n_windows, n_stations)``
presence mask, the layout ``RealNoiseSampler.from_windows`` and
``EmpiricalCovarianceEstimator.estimate_from_windows`` take. :func:`quiet_window_mask` screens a
pool of windows for the earthquakes it holds.
"""
from datetime import timedelta
from typing import Optional, Sequence, Tuple

import numpy as np
from obspy import Stream, UTCDateTime

from seismo_sbi.data_handling.preprocessing.sbi_export import observation_from_stream
from seismo_sbi.data_handling.preprocessing.windowing import get_continuous_regions, make_noise_windows
from seismo_sbi.simulators.receivers import Receivers
from seismo_sbi.utils.seismograms import compute_data_vector_length


def noise_windows_from_stream(stream: Stream, receivers: Receivers, window_length_s: float, sampling_rate_hz: float,
                              avoid_windows_utc: Sequence[Tuple] = (), buffer_s: float = 900.0,
                              step_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(noise_windows, present)`` of the processed ``stream``, filtered and resampled to
    ``sampling_rate_hz``: ``(n_windows, n_traces * n_samples)`` and ``(n_windows, n_stations)``.

    Windows of ``window_length_s`` start every ``step_s`` (``buffer_s`` when None) and keep
    ``buffer_s`` clear of the ends of the stream and of each ``(start, end)`` in
    ``avoid_windows_utc``, such as ``(origin_time, origin_time + window_length_s)`` for every event
    in the span. A station without all three components in a window is absent from it and its
    samples are zeros; a window holding no station, or a trace cut short by a data gap, is dropped.
    """
    span_start = min(trace.stats.starttime for trace in stream)
    span_end = max(trace.stats.endtime for trace in stream)
    regions = [(span_start, span_end)]
    if len(avoid_windows_utc) > 0:
        avoided = [(UTCDateTime(start), UTCDateTime(end)) for start, end in avoid_windows_utc]
        regions, _ = get_continuous_regions(avoided, span_start, span_end)
    windows_utc = make_noise_windows(regions, timedelta(seconds=window_length_s), buffer=timedelta(seconds=buffer_s),
                                     step=None if step_s is None else timedelta(seconds=step_s))

    n_samples = compute_data_vector_length(window_length_s, sampling_rate_hz) + 1
    row_length = sum(len(receiver.components) for receiver in receivers) * n_samples
    rows, present_masks = [], []
    for window_utc in windows_utc:
        noise_vector, present = observation_from_stream(stream, receivers, window_utc, sampling_rate_hz)
        if noise_vector.shape[0] == row_length and present.any():
            rows.append(noise_vector)
            present_masks.append(present)
    return (np.array(rows).reshape(len(rows), row_length),
            np.array(present_masks, dtype=bool).reshape(len(present_masks), len(receivers)))


def quiet_window_mask(vertical_rms: np.ndarray, max_rms_ratio: float = 5.0) -> np.ndarray:
    """Which windows of a recorded noise pool to keep: ``False`` where any station's vertical rms
    exceeds ``max_rms_ratio`` times that station's median over the pool, as it does when an
    earthquake falls inside a noise window.

    :param vertical_rms: ``(n_windows, n_stations)`` rms of each window's vertical trace.
    :returns: ``(n_windows,)`` boolean, ``True`` for a window kept.
    """
    station_median_rms = np.median(vertical_rms, axis=0)
    return np.all(vertical_rms <= max_rms_ratio * station_median_rms, axis=1)
