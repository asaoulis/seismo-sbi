"""Simulated seismograms as an ObsPy ``Stream``.

:func:`seismogram_map_to_stream` turns a simulator's ``{station: {component: waveform}}`` map into
one trace per station and component, with the receivers' network and station codes, channels
``<band><component>``, the simulator's output sampling rate and a start time ``pre_event_pad_s``
before the origin. :func:`seismogram_map_from_traces` rebuilds that map from a flat data vector.
"""
import numpy as np
from obspy import Stream, Trace, UTCDateTime

from .simulation_io import component_alias


def seismogram_map_from_traces(data_vector, traces) -> dict:
    """``{station: {component: waveform}}`` of a flat ``data_vector``, shape ``(n_traces * n_samples,)``,
    whose traces are ``[(station, component)]`` in order, as ``simulate_at(..., return_traces=True)`` returns.
    """
    per_trace = np.asarray(data_vector).reshape(len(traces), -1)
    seismogram_map = {}
    for (station, component), waveform in zip(traces, per_trace):
        seismogram_map.setdefault(station, {})[component] = waveform
    return seismogram_map


def seismogram_map_to_stream(seismogram_map: dict, simulator, origin_time_utc, channel_band: str = "BH") -> Stream:
    """The ``Stream`` of ``seismogram_map`` as ``simulator`` (a :class:`~seismo_sbi.simulators.base.Simulator`)
    produced it, for a source at ``origin_time_utc``.

    Traces follow receiver order; stations absent from the map are left out. The sampling rate is
    the simulator's ``synthetics_processing['sampling_rate']`` in Hz and each trace starts
    ``simulator.pre_event_pad_s`` before ``origin_time_utc`` (60 s for Instaseis, 0 for CPS); a
    simulator whose ``pre_event_pad_s`` is None raises ``ValueError``.
    """
    if simulator.pre_event_pad_s is None:
        raise ValueError(f"{type(simulator).__name__} does not state how long before the origin "
                         "its seismograms start (pre_event_pad_s)")
    sampling_rate_hz = float(simulator.synthetics_processing["sampling_rate"])
    start_time_utc = UTCDateTime(origin_time_utc) - simulator.pre_event_pad_s

    stream = Stream()
    for receiver in simulator.receivers.iterate():
        station_traces = seismogram_map.get(receiver.station_name)
        if station_traces is None:
            continue
        for component in receiver.components:
            waveform = station_traces.get(component)
            if waveform is None:
                waveform = station_traces[component_alias(component)]
            header = {"network": receiver.network, "station": receiver.station_name,
                      "channel": channel_band + component, "sampling_rate": sampling_rate_hz,
                      "starttime": start_time_utc}
            stream.append(Trace(data=np.asarray(waveform, dtype=float), header=header))
    return stream
