"""Seismogram array helpers shared by the simulators and the data pipeline.

Trace length from a duration and a sampling rate, a zero-padded integer-sample shift of one
trace, and the per-station application of ``receiver.time_shift`` (in samples) to a
``{station: {component: waveform}}`` map.
"""

import numpy as np


def compute_data_vector_length(data_length, sampling_rate):
    return int(data_length * sampling_rate)

def shift_1d_with_padding(x: np.ndarray, shift: int) -> np.ndarray:
    """``x`` shifted by ``shift`` samples and zero-padded; positive delays, negative advances."""
    if shift > 0:
        return np.concatenate([np.zeros(shift), x[:-shift]])
    elif shift < 0:
        s = abs(shift)
        return np.concatenate([x[s:], np.zeros(s)])
    else:
        return x.copy()

def apply_station_time_shifts(receivers, all_seismograms_map: dict) -> dict:
    """A new ``{station: {component: waveform}}`` map with each station's
    ``receiver.time_shift``, in samples, applied. The input map is not modified.
    """
    shifted_map = {}

    for receiver in receivers.iterate():
        station = receiver.station_name
        shift = int(receiver.time_shift)

        station_dict = all_seismograms_map.get(station, {})
        shifted_map[station] = {}

        for comp in station_dict.keys():
            data = station_dict[comp]
            shifted_map[station][comp] = shift_1d_with_padding(data, shift)

    return shifted_map
