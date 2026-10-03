"""Simulated seismograms become an ObsPy Stream on the simulator's own time base."""
import numpy as np
import pytest
from obspy import UTCDateTime

from seismo_sbi.data_handling.preprocessing.sbi_export import stream_to_seismogram_map
from seismo_sbi.simulators.base import Simulator
from seismo_sbi.simulators.cps.simulator import CPSPrecomputedSimulator, CPSSimulator, MultiModelCPSSimulator
from seismo_sbi.simulators.instaseis.ensemble import InstaseisEnsembleSimulator
from seismo_sbi.simulators.instaseis.multi_model import InstaseisMultiModelSimulator
from seismo_sbi.simulators.instaseis.querier import SYNTHETICS_PRE_EVENT_PAD_S
from seismo_sbi.simulators.instaseis.simulator import InstaseisSourceSimulator
from seismo_sbi.simulators.kernel import FixedLocationKernelSimulator
from seismo_sbi.simulators.receivers import Receivers
from seismo_sbi.simulators.simulation_io import component_alias
from seismo_sbi.simulators.synthetic_stream import seismogram_map_from_traces, seismogram_map_to_stream

ORIGIN_TIME = UTCDateTime(2020, 1, 1, 12)
DURATION_S = 30.0
SAMPLING_RATE_HZ = 2.0


class PaddedSimulator(Simulator):
    pre_event_pad_s = 60.0

    def generic_point_source_simulation(self, source, **kwargs):
        raise NotImplementedError


def padded_simulator():
    receivers = Receivers.from_arrays(["AAA", "BBB"], ["XX", "YY"], [0.0, 1.0], [0.0, 1.0])
    return PaddedSimulator("ZEN", receivers, DURATION_S, {"sampling_rate": SAMPLING_RATE_HZ})


def seismogram_map(simulator, horizontal_keys="12"):
    rng = np.random.default_rng(0)
    n_samples = int(DURATION_S * SAMPLING_RATE_HZ) + 1
    return {receiver.station_name: {key: rng.normal(size=n_samples) for key in "Z" + horizontal_keys}
            for receiver in simulator.receivers}


@pytest.mark.parametrize("horizontal_keys", ["12", "EN"])
def test_stream_reads_back_as_the_seismogram_map(horizontal_keys):
    simulator = padded_simulator()
    expected = seismogram_map(simulator, horizontal_keys)

    stream = seismogram_map_to_stream(expected, simulator, ORIGIN_TIME)
    start = ORIGIN_TIME - 60.0
    read_back = stream_to_seismogram_map(stream, ["AAA", "BBB"], start, start + DURATION_S)

    for station, components in expected.items():
        for key, waveform in components.items():
            assert np.array_equal(read_back[station][component_alias(key)], waveform)


def test_stream_carries_codes_rate_and_the_pre_origin_pad():
    simulator = padded_simulator()
    stream = seismogram_map_to_stream(seismogram_map(simulator), simulator, ORIGIN_TIME)

    assert [trace.id for trace in stream] == ["XX.AAA..BHZ", "XX.AAA..BHE", "XX.AAA..BHN",
                                              "YY.BBB..BHZ", "YY.BBB..BHE", "YY.BBB..BHN"]
    assert all(trace.stats.sampling_rate == SAMPLING_RATE_HZ for trace in stream)
    assert all(trace.stats.starttime == ORIGIN_TIME - 60.0 for trace in stream)
    assert stream[0].stats.endtime == ORIGIN_TIME - 60.0 + DURATION_S


def test_stations_absent_from_the_map_are_left_out():
    simulator = padded_simulator()
    subset = {"BBB": seismogram_map(simulator)["BBB"]}
    assert {trace.stats.station for trace in seismogram_map_to_stream(subset, simulator, ORIGIN_TIME)} == {"BBB"}


def test_flat_data_vector_rebuilds_the_seismogram_map():
    traces = [("AAA", "Z"), ("AAA", "E"), ("BBB", "Z")]
    data_vector = np.arange(12.0)
    rebuilt = seismogram_map_from_traces(data_vector, traces)
    assert np.array_equal(rebuilt["AAA"]["E"], [4.0, 5.0, 6.0, 7.0])
    assert np.array_equal(rebuilt["BBB"]["Z"], [8.0, 9.0, 10.0, 11.0])


def test_backends_state_their_pre_origin_pad():
    for instaseis_backend in (InstaseisSourceSimulator, InstaseisEnsembleSimulator, InstaseisMultiModelSimulator):
        assert instaseis_backend.pre_event_pad_s == SYNTHETICS_PRE_EVENT_PAD_S
    for cps_backend in (CPSSimulator, CPSPrecomputedSimulator, MultiModelCPSSimulator):
        assert cps_backend.pre_event_pad_s == 0.0
    assert FixedLocationKernelSimulator.pre_event_pad_s is None


def test_simulator_without_a_stated_pad_is_rejected():
    simulator = padded_simulator()
    simulator.pre_event_pad_s = None
    with pytest.raises(ValueError):
        seismogram_map_to_stream(seismogram_map(simulator), simulator, ORIGIN_TIME)
