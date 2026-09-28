"""A forward model run from a point-source record, and one synthetic at a pinned hypocentre."""
import numpy as np
import pytest

from seismo_sbi.sbi.simulator_wrapper import GeneralSimulatorWrapper
from seismo_sbi.simulators.base import Simulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.simulation_io import SimulationDataLoader, seismogram_map_to_array
from seismo_sbi.simulators.sources import (
    GeneralMomentTensor, GenericPointSource, SimpleMomentTensor, SourceLocation)

MOMENT_TENSOR = [1.0e15, -2.0e15, 1.0e15, 3.0e14, -4.0e14, 5.0e14]
LOCATION = SourceLocation(37.6, -118.9, 7.5, 0.0)


class SourceEchoSimulator(Simulator):
    """Each trace is the source location, the six tensor components and the model choice."""

    def generic_point_source_simulation(self, source, *, stf_duration=None, use_fiducial=None, **kwargs):
        model = 0.0 if use_fiducial else np.random.uniform()
        trace = np.r_[list(source.source_location), source.moment_tensor.components, model]
        return {receiver.station_name: {component: trace * (index + 1) for component in receiver.components}
                for index, receiver in enumerate(self.receivers.iterate())}


class RandomGainEffect:
    """A nuisance effect that multiplies every trace by a random gain whatever it is passed."""

    def __call__(self, seismograms_map, receivers, **nuisance_params):
        return {station: {component: trace * np.random.uniform(0.5, 1.5) for component, trace in traces.items()}
                for station, traces in seismograms_map.items()}


@pytest.fixture
def receivers():
    return Receivers(receivers=[Receiver(37.0, -118.0, "XX", "AAA", ["Z", "E", "N"]),
                                Receiver(38.0, -119.0, "XX", "BBB", ["Z"])])


@pytest.fixture
def simulator(receivers):
    return SourceEchoSimulator(["Z", "E", "N"], receivers, 11, {"sampling_rate": 1.0},
                               post_processing_effects=[RandomGainEffect()])


def wrapper_around(simulator):
    wrapper = object.__new__(GeneralSimulatorWrapper)
    wrapper.simulator = simulator
    wrapper.data_loader_callable = SimulationDataLoader("ZEN", simulator.receivers).convert_sim_data_to_array
    return wrapper


@pytest.mark.parametrize("moment_tensor, as_dict", [
    (GeneralMomentTensor(MOMENT_TENSOR), {"moment_tensor": MOMENT_TENSOR}),
    (SimpleMomentTensor(2.0e15), {"earthquake_magnitude": [2.0e15]}),
])
def test_run_simulation_takes_a_point_source_as_the_dict_it_stands_for(simulator, moment_tensor, as_dict):
    simulator.post_processing_chain.effects = []
    source = GenericPointSource(LOCATION, moment_tensor)

    _, from_record = simulator.run_simulation(source, use_fiducial=True)
    _, from_dict = simulator.run_simulation({"source_location": list(LOCATION), **as_dict}, use_fiducial=True)

    for station in from_dict:
        for component in from_dict[station]:
            np.testing.assert_array_equal(from_record[station][component], from_dict[station][component])


def test_simulate_at_is_the_fiducial_run_without_nuisance_effects(simulator, receivers):
    expected_simulator = SourceEchoSimulator(["Z", "E", "N"], receivers, 11, {"sampling_rate": 1.0})
    _, expected_map = expected_simulator.run_simulation(
        {"source_location": list(LOCATION), "moment_tensor": MOMENT_TENSOR}, use_fiducial=True)

    data_vector = wrapper_around(simulator).simulate_at(LOCATION, MOMENT_TENSOR)

    np.testing.assert_array_equal(data_vector, seismogram_map_to_array(expected_map, receivers))


def test_simulate_at_draws_no_random_numbers_and_leaves_the_simulator_alone(simulator):
    wrapper = wrapper_around(simulator)
    np.random.seed(0)
    state = np.random.get_state()[1].copy()

    first = wrapper.simulate_at(list(LOCATION), MOMENT_TENSOR)
    second = wrapper.simulate_at(list(LOCATION), MOMENT_TENSOR)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.random.get_state()[1], state)
    assert len(simulator.post_processing_chain.effects) == 1


def test_simulate_at_keeps_the_named_stations_in_receiver_order(simulator):
    wrapper = wrapper_around(simulator)
    every_station = wrapper.simulate_at(LOCATION, MOMENT_TENSOR).reshape(4, -1)

    data_vector, traces = wrapper.simulate_at(LOCATION, MOMENT_TENSOR, stations=["BBB"], return_traces=True)

    assert traces == [("BBB", "Z")]
    np.testing.assert_array_equal(data_vector, every_station[3])
