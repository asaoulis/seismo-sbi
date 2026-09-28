"""A forward model run from a point-source record."""
import numpy as np
import pytest

from seismo_sbi.simulators.base import Simulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
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
