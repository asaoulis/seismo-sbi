import numpy as np

def compute_data_vector_length(data_length, sampling_rate):
    return int(data_length * sampling_rate)

def shift_1d_with_padding(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift 1D array by `shift` samples using zero padding.
      shift > 0 : delay (shift right)
      shift < 0 : advance (shift left)
    """
    n = len(x)
    if shift > 0:
        return np.concatenate([np.zeros(shift), x[:-shift]])
    elif shift < 0:
        s = abs(shift)
        return np.concatenate([x[s:], np.zeros(s)])
    else:
        return x.copy()

def apply_station_time_shifts(receivers, all_seismograms_map: dict) -> dict:
    """
    all_seismograms_map: {station_name: {component: waveform_array}}
    Uses receiver.time_shift (int, in samples) for each station.
    Returns a NEW shifted map (does not modify input in-place).
    """
    shifted_map = {}

    for receiver in receivers.iterate():
        station = receiver.station_name
        shift = int(receiver.time_shift)  # in samples

        # Copy to avoid modifying original
        station_dict = all_seismograms_map.get(station, {})
        shifted_map[station] = {}

        for comp in station_dict.keys():
            data = station_dict[comp]
            shifted_map[station][comp] = shift_1d_with_padding(data, shift)

    return shifted_map