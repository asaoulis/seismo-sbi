"""SBI boundary: convert preprocessed Stream data to the seismo-sbi HDF5 format.

This is the ONLY place in the new pipeline that writes h5 files.
The schema written here is identical to what SimulationSaver.dump_data_as_hdf5
produces, so RealNoiseSampler and SimulationDataLoader can consume these files
without modification.

Key behavioural contracts (locked in by Phase 0 tests):
- Channel keys on disk are 'Z', '1', '2' — never 'E' or 'N'.
  (BHE/HHE/… → '1', BHN/HHN/… → '2')
- Array length = compute_data_vector_length(duration, sr) + 1 (inclusive slice).
- Autocorrelation in /misc is computed from the pre-event window
  [event_start - covariance_window, event_start], averaged as:
      auto_correlate[:n][::-1] / arange(n, 0, -1)
"""

import math
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import obspy
from obspy import Stream, UTCDateTime

from seismo_sbi.instaseis_simulator.simulation_saver import SimulationSaver


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rename_component(channel: str) -> str:
    """Map a full channel code (e.g. 'BHE') to the SBI component key ('1').

    Rules (matching ProcessedDataSlicer.rename_component):
        Z in channel  → 'Z'
        1 in channel or channel ends with E → '1'
        2 in channel or channel ends with N → '2'
    """
    if "Z" in channel:
        return "Z"
    if "1" in channel or channel[-1] == "E":
        return "1"
    if "2" in channel or channel[-1] == "N":
        return "2"
    raise ValueError(f"Cannot map channel '{channel}' to Z/1/2")


def _compute_autocorrelation(data: np.ndarray) -> np.ndarray:
    """Compute the normalised one-sided autocorrelation used for /misc.

    Reproduces ProcessedDataSlicer lines 374-376:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    # Build component map from channel codes present in the stream
    # station → {channel_code → renamed_key}
    station_channel_map: dict = {}
    for tr in stream:
        sta = tr.stats.station
        if sta not in receivers:
            continue
        cha = tr.stats.channel
        renamed = _rename_component(cha)
        if sta not in station_channel_map:
            station_channel_map[sta] = {}
        station_channel_map[sta][cha] = renamed

    # Slice event window (inclusive endpoints, legacy behaviour)
    event_stream = stream.slice(t_start, exact_end)

    # Build output data_map: {station: {renamed_component: np.ndarray}}
    data_map: dict = {}
    for sta in receivers:
        if sta not in station_channel_map:
            continue
        ch_map = station_channel_map[sta]
        sta_data: dict = {}
        for cha, renamed in ch_map.items():
            traces = event_stream.select(station=sta, channel=cha)
            if len(traces) == 0:
                continue
            sta_data[renamed] = traces[0].data.copy()
        if len(sta_data) == 3:
            data_map[sta] = sta_data

    # Build /misc: autocorrelation from pre-event window
    variance_dict: Optional[dict] = None
    if covariance_window is not None:
        cov_start = t_start - covariance_window.total_seconds()
        cov_stream = stream.slice(cov_start, t_start)
        variance_dict = {}
        for sta in receivers:
            if sta not in station_channel_map:
                continue
            ch_map = station_channel_map[sta]
            sta_misc: dict = {}
            for cha, renamed in ch_map.items():
                traces = cov_stream.select(station=sta, channel=cha)
                if len(traces) == 0:
                    continue
                data = traces[0].data
                if full_auto_correlation:
                    auto_cov = _compute_autocorrelation(data)
                else:
                    auto_cov = np.var(data)
                sta_misc[renamed] = auto_cov
            if sta_misc:
                variance_dict[sta] = sta_misc

    saver = SimulationSaver(output_data=data_map, misc_data=variance_dict)
    saver.dump_data_as_hdf5(out_path)
