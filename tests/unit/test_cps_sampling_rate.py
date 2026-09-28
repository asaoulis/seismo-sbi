"""CPS backends refuse a configured sampling rate other than the 1 Hz of their Green's functions."""
import pytest

from seismo_sbi.simulators.cps.simulator import CPSSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
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
