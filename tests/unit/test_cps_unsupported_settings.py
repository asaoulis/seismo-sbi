"""CPS backends refuse what they cannot apply: a sampling rate other than 1 Hz, a depth offset, a source time shift."""
import pytest

from seismo_sbi.simulators.cps.simulator import CPSSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.sources import GeneralMomentTensor, GenericPointSource, SourceLocation
from seismo_sbi.utils.errors import InvalidConfiguration


class NoKernelCPSSimulator(CPSSimulator):

    def compute_or_load_greens_functions(self, objstats, velocity_model, **kwargs):
        raise NotImplementedError


RECEIVERS = Receivers(receivers=[Receiver(10.0, 0.0, "XX", "AAA", ["Z", "E", "N"])])


def test_cps_simulator_rejects_a_sampling_rate_other_than_1_hz():
    with pytest.raises(InvalidConfiguration, match="sampling_rate"):
        NoKernelCPSSimulator(None, None, "ZEN", RECEIVERS, 200, {"sampling_rate": 0.5})


def test_cps_simulator_accepts_1_hz():
    simulator = NoKernelCPSSimulator(None, None, "ZEN", RECEIVERS, 200, {"sampling_rate": 1})
    assert simulator.num_traces == 3


def test_cps_simulator_rejects_a_source_depth_offset():
    with pytest.raises(InvalidConfiguration, match="source_depth_offset_km"):
        NoKernelCPSSimulator(None, None, "ZEN", RECEIVERS, 200, {"sampling_rate": 1}, source_depth_offset_km=0.5)


def test_cps_simulation_rejects_a_source_time_shift():
    simulator = NoKernelCPSSimulator(None, None, "ZEN", RECEIVERS, 200, {"sampling_rate": 1})
    source = GenericPointSource(SourceLocation(0.0, 1.0, 10.0, 2.0), GeneralMomentTensor([1e15] * 6))
    with pytest.raises(InvalidConfiguration, match="time shift"):
        simulator.generic_point_source_simulation(source)
