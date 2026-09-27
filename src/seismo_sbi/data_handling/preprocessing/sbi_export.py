"""Write preprocessed streams to the HDF5 format the rest of the library reads.

:func:`stream_to_seismogram_map` turns an obspy ``Stream`` into the ``{station: {component:
waveform}}`` map the simulators produce, and :func:`export_to_sbi_h5` writes it. The schema matches what ``SimulationSaver.dump_data_as_hdf5`` produces, so ``RealNoiseSampler``
and ``SimulationDataLoader`` consume these files unchanged. Channel keys on disk are ``Z``,
``1`` and ``2``, never ``E`` or ``N``. Each array is ``compute_data_vector_length(duration, sr)
+ 1`` samples long, the slice being inclusive. The autocorrelation in ``/misc`` is taken over
the pre-event window and averaged as ``auto_correlate[:n][::-1] / arange(n, 0, -1)``.
"""

import math
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
from obspy import Stream, UTCDateTime

from seismo_sbi.simulators.simulation_io import SimulationSaver, component_alias


# --- Internal helpers ---

def _rename_component(channel: str) -> str:
    """The component key (``Z``, ``1`` or ``2``) of a SEED channel code such as ``BHE``."""
    component = component_alias(channel[-1])
    if component not in ("Z", "1", "2"):
        raise ValueError(f"Cannot map channel '{channel}' to Z/1/2")
    return component


def _compute_autocorrelation(data: np.ndarray) -> np.ndarray:
    """Compute the normalised one-sided autocorrelation used for /misc.

    That is:
        auto_correlate = np.correlate(data, data, mode='full')
        averaged = auto_correlate[:n][::-1] / np.arange(n, 0, -1)
    """
    n = data.shape[0]
    full = np.correlate(data, data, mode="full")
    return full[:n][::-1] / np.arange(n, 0, -1)


def _exact_end_time(t_start: UTCDateTime, t_end: UTCDateTime, sampling_rate: float) -> UTCDateTime:
    """Compute the inclusive slice end time that matches legacy behaviour."""
    duration = t_end - t_start
    fixed_seconds = math.ceil(duration / sampling_rate) * sampling_rate
    return t_start + fixed_seconds


# --- Public API ---

def stream_to_seismogram_map(stream: Stream, station_names: List[str], t_start, t_end) -> dict:
    """``{station: {component: waveform}}`` for ``station_names`` from the traces of ``stream``
    between ``t_start`` and ``t_end`` (inclusive), component keys ``Z``, ``1``, ``2``.

    A station absent from the stream is absent from the map; a component absent from the
    window is absent from its station.
    """
    station_channel_map: dict = {}
    for trace in stream:
        station = trace.stats.station
        if station not in station_names:
            continue
        station_channel_map.setdefault(station, {})[trace.stats.channel] = _rename_component(
            trace.stats.channel)

    window = stream.slice(UTCDateTime(t_start), UTCDateTime(t_end))
    seismogram_map: dict = {}
    for station in station_names:
        if station not in station_channel_map:
            continue
        station_traces: dict = {}
        for channel, renamed in station_channel_map[station].items():
            traces = window.select(station=station, channel=channel)
            if len(traces) == 0:
                continue
            station_traces[renamed] = traces[0].data.copy()
        seismogram_map[station] = station_traces
    return seismogram_map

def export_to_sbi_h5(
    stream: Stream,
    receivers: List[str],
    event_window,
    out_path: Path,
    sampling_rate: float,
    covariance_window: Optional[timedelta] = None,
    full_auto_correlation: bool = True,
) -> None:
    """Write a preprocessed Stream to seismo-sbi HDF5 format.

    The stream must already be preprocessed (filtered, resampled to
    sampling_rate) and must cover at least [event_start - covariance_window,
    event_end] to allow pre-event covariance estimation.

    Args:
        stream: Preprocessed Stream (all stations, all components).
        receivers: Ordered list of station names to include.
        event_window: (t_start, t_end) as datetime or UTCDateTime.
        out_path: Destination .h5 file path.
        sampling_rate: Target sampling rate (Hz); must match stream traces.
        covariance_window: Pre-event window length for autocorrelation.
            If None, no /misc group is written.
        full_auto_correlation: When True write the full normalised
            autocorrelation array; when False write a scalar variance.
    """
    t_start = UTCDateTime(event_window[0])
    t_end = UTCDateTime(event_window[1])
    exact_end = _exact_end_time(t_start, t_end, sampling_rate)

    data_map = {station: traces for station, traces
                in stream_to_seismogram_map(stream, receivers, t_start, exact_end).items()
                if len(traces) == 3}

    variance_dict: Optional[dict] = None
    if covariance_window is not None:
        cov_start = t_start - covariance_window.total_seconds()
        variance_dict = {}
        for sta, traces in stream_to_seismogram_map(stream, receivers, cov_start, t_start).items():
            sta_misc = {renamed: _compute_autocorrelation(data) if full_auto_correlation else np.var(data)
                        for renamed, data in traces.items()}
            if sta_misc:
                variance_dict[sta] = sta_misc

    saver = SimulationSaver(output_data=data_map, misc_data=variance_dict)
    saver.dump_data_as_hdf5(out_path)
