"""CPS seismograms follow the receivers the simulator holds when it runs."""
import numpy as np

from seismo_sbi.simulators.cps.simulator import CPSSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.sources import GeneralMomentTensor, GenericPointSource, SourceLocation

N_SAMPLES = 20


class StationSeededCPSSimulator(CPSSimulator):
    """Green's functions ``(n_stations, 3, 6, n_samples)`` drawn from a seed set by each station's latitude."""

    def compute_or_load_greens_functions(self, objstats, velocity_model, **kwargs):
        return np.stack([np.random.default_rng(int(receiver.latitude)).normal(size=(3, 6, N_SAMPLES))
                         for receiver in self.receivers.iterate()])


def test_cps_seismograms_follow_a_receiver_list_shortened_after_construction():
    receivers = Receivers(receivers=[Receiver(float(latitude), 0.0, "XX", name, ["Z", "E", "N"])
                                     for latitude, name in [(10, "AAA"), (20, "BBB"), (30, "CCC")]])
    simulator = StationSeededCPSSimulator(None, None, "ZEN", receivers, N_SAMPLES, {"sampling_rate": 1.0})
    source = GenericPointSource(SourceLocation(0.0, 1.0, 10.0, 0.0),
                                GeneralMomentTensor([1e15, -2e15, 1e15, 3e14, -4e14, 5e14]))
    every_station = simulator.generic_point_source_simulation(source)

    receivers.receivers = receivers.receivers[1:2]
    one_station = simulator.generic_point_source_simulation(source)

    assert list(one_station) == ["BBB"]
    for component in "ZEN":
        np.testing.assert_array_equal(one_station["BBB"][component], every_station["BBB"][component])
