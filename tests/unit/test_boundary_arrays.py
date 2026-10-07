"""In-memory seismogram maps and obspy streams become the same arrays as the HDF5 files."""
from datetime import timedelta

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from seismo_sbi.data_handling.preprocessing.sbi_export import (
    _rename_component, export_to_sbi_h5, observation_from_stream, pre_event_autocorrelations,
    stream_to_seismogram_map)
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.simulation_io import (
    SimulationDataLoader, SimulationSaver, component_alias, seismogram_map_to_array)
from seismo_sbi.utils.errors import InvalidConfiguration

N_SAMPLES = 16


def three_stations():
    return Receivers(receivers=[
        Receiver(37.0, -118.0, "XX", "AAA", ["Z", "E", "N"]),
        Receiver(38.0, -119.0, "XX", "BBB", ["Z", "E", "N"]),
        Receiver(39.0, -120.0, "XX", "CCC", ["Z"]),
    ])


def seismogram_map(receivers):
    rng = np.random.default_rng(0)
    return {receiver.station_name: {component: rng.normal(size=N_SAMPLES)
                                    for component in receiver.components}
            for receiver in receivers.iterate()}


def test_seismogram_map_to_array_matches_the_hdf5_round_trip(tmp_path):
    receivers = three_stations()
    waveforms = seismogram_map(receivers)
    path = tmp_path / "sim.h5"
    SimulationSaver(output_data=waveforms).dump_data_as_hdf5(path)
    loader = SimulationDataLoader("ZEN", receivers)

    flat = seismogram_map_to_array(waveforms, receivers)
    assert flat.shape == (7 * N_SAMPLES,)
    assert np.array_equal(flat, loader.load_simulation_data_array(path))
    assert np.array_equal(flat[:N_SAMPLES], waveforms["AAA"]["Z"])
    assert np.array_equal(flat[-N_SAMPLES:], waveforms["CCC"]["Z"])


def test_seismogram_map_to_array_stacks_equal_component_counts():
    receivers = Receivers(receivers=three_stations().receivers[:2])
    waveforms = seismogram_map(receivers)
    stacked = seismogram_map_to_array(waveforms, receivers, stacked=True)
    assert stacked.shape == (2, 3, N_SAMPLES)
    assert np.array_equal(stacked[1, 2], waveforms["BBB"]["N"])


def test_seismogram_map_to_array_raises_on_a_missing_station():
    receivers = three_stations()
    waveforms = seismogram_map(receivers)
    del waveforms["BBB"]
    with pytest.raises(KeyError):
        seismogram_map_to_array(waveforms, receivers)


def test_zero_fill_pads_a_z_only_station_in_the_layout_order(tmp_path):
    receivers = three_stations()
    waveforms = seismogram_map(receivers)
    path = tmp_path / "sim.h5"
    SimulationSaver(output_data=waveforms).dump_data_as_hdf5(path)
    stacked = SimulationDataLoader("ZEN", receivers).load_simulation_data_array(
        path, stacked=True, fill_unused=True)
    assert stacked.shape == (3, 3, N_SAMPLES)
    assert np.array_equal(stacked[2, 0], waveforms["CCC"]["Z"])
    assert not stacked[2, 1:].any()


def test_zero_fill_rejects_a_station_ordered_against_the_layout(tmp_path):
    receivers = Receivers(receivers=[Receiver(37.0, -118.0, "XX", "AAA", ["Z", "N", "E"])])
    waveforms = seismogram_map(receivers)
    path = tmp_path / "sim.h5"
    SimulationSaver(output_data=waveforms).dump_data_as_hdf5(path)
    with pytest.raises(InvalidConfiguration):
        SimulationDataLoader("ZEN", receivers).load_simulation_data_array(
            path, stacked=True, fill_unused=True)


@pytest.mark.parametrize("channel, component", [
    ("BHZ", "Z"), ("BHE", "1"), ("BHN", "2"), ("HHZ", "Z"), ("EHE", "1"),
])
def test_rename_component_reads_the_orientation_code(channel, component):
    assert _rename_component(channel) == component


@pytest.mark.parametrize("channel", ["LOG", "BH1", "BH2"])
def test_rename_component_rejects_a_channel_not_coded_z_e_or_n(channel):
    with pytest.raises(ValueError):
        _rename_component(channel)


def synthetic_stream(station_names, t_start, sampling_rate_hz=1.0, n_samples=120):
    rng = np.random.default_rng(1)
    stream = Stream()
    for station in station_names:
        for channel in ("BHZ", "BHE", "BHN"):
            trace = Trace(data=rng.normal(size=n_samples))
            trace.stats.update({"station": station, "channel": channel, "network": "XX",
                                "sampling_rate": sampling_rate_hz, "starttime": t_start})
            stream.append(trace)
    return stream


def test_stream_to_seismogram_map_matches_the_exported_file(tmp_path):
    t_start = UTCDateTime(2020, 1, 1)
    stream = synthetic_stream(["AAA", "BBB"], t_start - 60)
    event_window = (t_start, t_start + 30)
    path = tmp_path / "event.h5"
    export_to_sbi_h5(stream, ["AAA", "BBB"], event_window, path, sampling_rate=1.0)

    receivers = Receivers.from_arrays(["AAA", "BBB"], ["XX", "XX"], [0.0, 1.0], [0.0, 1.0])
    from_file = SimulationDataLoader("ZEN", receivers).load_simulation_data_array(path, stacked=True)
    from_stream = stream_to_seismogram_map(stream, ["AAA", "BBB"], *event_window)
    assert set(from_stream["AAA"]) == {"Z", "1", "2"}
    assert np.array_equal(from_stream["BBB"]["2"], from_file[1, 2])
    assert from_stream["AAA"]["Z"].shape == (31,)


def test_stream_to_seismogram_map_skips_absent_stations_and_components():
    t_start = UTCDateTime(2020, 1, 1)
    stream = synthetic_stream(["AAA"], t_start)
    stream.remove(stream.select(channel="BHN")[0])
    seismogram_map = stream_to_seismogram_map(stream, ["AAA", "ZZZ"], t_start, t_start + 10)
    assert list(seismogram_map) == ["AAA"]
    assert set(seismogram_map["AAA"]) == {"Z", "1"}


def test_event_window_has_n_plus_one_samples_at_20_hz():
    from seismo_sbi.data_handling.preprocessing.sbi_export import _exact_end_time
    from seismo_sbi.data_handling.preprocessing.windowing import slice_event_window

    start = UTCDateTime("2020-03-01T12:00:00")
    stream = Stream([Trace(np.zeros(2000), header=dict(station="AAA", channel=f"BH{component}",
                                                        sampling_rate=20.0, starttime=start - 10))
                     for component in "ZEN"])

    sliced = slice_event_window(stream, start, start + 30.3, 20.0)

    assert [trace.stats.npts for trace in sliced] == [607, 607, 607]
    assert _exact_end_time(start, start + 30.3, 20.0) - start == pytest.approx(30.3)


def test_observation_from_stream_equals_the_exported_file_read_back(tmp_path):
    t_start = UTCDateTime(2020, 1, 1)
    stream = synthetic_stream(["AAA", "BBB"], t_start - 60)
    event_window = (t_start, t_start + 30)
    path = tmp_path / "event.h5"
    export_to_sbi_h5(stream, ["ZZZ", "AAA", "BBB"], event_window, path, sampling_rate=1.0)
    receivers = Receivers.from_arrays(["ZZZ", "AAA", "BBB"], ["XX"] * 3, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    from_file, file_mask = SimulationDataLoader("ZEN", receivers, data_length=31) \
        .load_simulation_data_array_with_presence(path)
    in_memory, memory_mask = observation_from_stream(stream, receivers, event_window, sampling_rate_hz=1.0)

    assert np.array_equal(in_memory, from_file)
    assert memory_mask.tolist() == file_mask.tolist() == [False, True, True]


def test_observation_from_stream_returns_the_window_of_each_trace_in_receiver_order():
    t_start = UTCDateTime(2020, 1, 1)
    stream = synthetic_stream(["AAA", "BBB"], t_start - 60)
    receivers = Receivers.from_arrays(["BBB", "AAA"], ["XX", "XX"], [0.0, 1.0], [0.0, 1.0])

    stacked, mask = observation_from_stream(stream, receivers, (t_start, t_start + 30), 1.0, stacked=True)

    expected = [[stream.select(station=station, channel=f"BH{component}")[0].data[60:91]
                 for component in "ZEN"] for station in ("BBB", "AAA")]
    assert np.array_equal(stacked, np.array(expected))
    assert mask.all()


def test_pre_event_autocorrelations_are_the_exported_misc_group(tmp_path):
    t_start = UTCDateTime(2020, 1, 1)
    stream = synthetic_stream(["AAA", "BBB"], t_start - 60)
    path = tmp_path / "event.h5"
    export_to_sbi_h5(stream, ["AAA", "BBB"], (t_start, t_start + 30), path, sampling_rate=1.0,
                     covariance_window=timedelta(seconds=40))
    receivers = Receivers.from_arrays(["AAA", "BBB"], ["XX", "XX"], [0.0, 1.0], [0.0, 1.0])

    from_file = SimulationDataLoader("ZEN", receivers).load_misc_data(path)
    in_memory = pre_event_autocorrelations(stream, ["AAA", "BBB"], t_start, covariance_window_s=40.0)

    for station, components in from_file.items():
        for component, autocorrelation in components.items():
            assert np.array_equal(in_memory[station][component_alias(component)], autocorrelation)
