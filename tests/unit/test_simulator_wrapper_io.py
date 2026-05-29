"""Characterization tests for GeneralSimulatorWrapper.input_output_simulation().

input_output_simulation() is a pure data-routing function: it samples nuisance
parameters, merges them with the theta vector, runs the simulator, and returns a
flat array.  We test this logic directly — without constructing a real
GeneralSimulatorWrapper (which would require a live Instaseis / CPS installation)
— by calling the method as an unbound function with mock collaborators.
"""

import numpy as np
import pytest
from functools import partial
from copy import deepcopy

from seismo_sbi.instaseis_simulator.simulator import Simulator
from seismo_sbi.instaseis_simulator.wrapper import GenericPointSource
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
from seismo_sbi.sbi.types.parameters import ModelParameters
from seismo_sbi.sbi.simulator_wrapper import GeneralSimulatorWrapper

TRACE_LEN = 40
_SOURCE_LOC = [0.0, 0.0, 10.0, 0.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockSimulator(Simulator):
    """Returns canned seismograms and records the last inputs_map it received."""

    def __init__(self, receivers, trace_len=TRACE_LEN):
        super().__init__(
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=trace_len,
            synthetics_processing={
                "sampling_rate": 1.0,
                "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
            },
        )
        self._trace_len = trace_len
        self.received_inputs = {}

    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        return {
            rec.station_name: {comp: np.ones(self._trace_len) for comp in rec.components}
            for rec in self.receivers.iterate()
        }

    # Wrap run_simulation to capture the full inputs dict
    def run_simulation(self, source_parameters, **kwargs):
        self.received_inputs = deepcopy(dict(source_parameters))
        return super().run_simulation(source_parameters, **kwargs)


def _constant_sampler(value, n):
    """Yields value n times — mirrors dataset_generator.constant_sampler."""
    for _ in range(n):
        yield value


def _make_mt_model_parameters():
    """ModelParameters with 6-component MT inference and fixed source_location nuisance."""
    mp = ModelParameters()
    mp.names["moment_tensor"] = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    mp.theta_fiducial["moment_tensor"] = [1e14] * 6
    mp.nuisance["source_location"] = _SOURCE_LOC
    mp.bounds["source_location"] = _SOURCE_LOC
    return mp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def one_station_receivers():
    r = Receiver(0.0, 0.0, "XX", "STA1", ["Z"])
    return Receivers(receivers=[r])


@pytest.fixture
def mock_sim(one_station_receivers):
    return MockSimulator(one_station_receivers)


@pytest.fixture
def mt_parameters():
    return _make_mt_model_parameters()


@pytest.fixture
def data_loader(one_station_receivers):
    return SimulationDataLoader(components=["Z"], receivers=one_station_receivers)


@pytest.fixture
def samplers():
    """Constant sampler for source_location."""
    return {"source_location": partial(_constant_sampler, np.array(_SOURCE_LOC))}


# ---------------------------------------------------------------------------
# Helpers to call input_output_simulation without a real wrapper instance
# ---------------------------------------------------------------------------

def _call_io_sim(parameters, data_loader, samplers, simulator, theta, **kwargs):
    """Invoke input_output_simulation as an unbound call (self is unused)."""
    return GeneralSimulatorWrapper.input_output_simulation(
        None,  # self — not referenced in the method body
        parameters,
        data_loader,
        samplers,
        simulator,
        theta,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInputOutputSimulation:

    def test_output_is_flat_1d_array(self, mt_parameters, data_loader, samplers, mock_sim):
        theta = np.array([1e14] * 6)
        result = _call_io_sim(mt_parameters, data_loader, samplers, mock_sim, theta)
        assert result.ndim == 1

    def test_output_length_matches_trace_len(self, mt_parameters, data_loader, samplers, mock_sim):
        theta = np.array([1e14] * 6)
        result = _call_io_sim(mt_parameters, data_loader, samplers, mock_sim, theta)
        # One station, one component, TRACE_LEN samples
        assert len(result) == TRACE_LEN

    def test_theta_vector_reaches_simulator_as_moment_tensor(
        self, mt_parameters, data_loader, samplers, mock_sim
    ):
        """The theta values must appear in the moment_tensor entry passed to run_simulation."""
        mt_values = [2e14, 3e14, 4e14, 5e14, 6e14, 7e14]
        theta = np.array(mt_values)
        _call_io_sim(mt_parameters, data_loader, samplers, mock_sim, theta)
        assert np.allclose(
            mock_sim.received_inputs["moment_tensor"], mt_values
        ), "moment_tensor in simulator inputs must match the theta vector"

    def test_nuisance_source_location_reaches_simulator(
        self, mt_parameters, data_loader, samplers, mock_sim
    ):
        """The sampled source_location must appear in inputs passed to run_simulation."""
        theta = np.array([1e14] * 6)
        _call_io_sim(mt_parameters, data_loader, samplers, mock_sim, theta)
        assert "source_location" in mock_sim.received_inputs
        assert np.allclose(
            mock_sim.received_inputs["source_location"], _SOURCE_LOC
        )

    def test_empty_nuisance_dict_works(self, data_loader, mock_sim, one_station_receivers):
        """Backward compat: ModelParameters with no nuisance params must not error."""
        mp = ModelParameters()
        mp.names["moment_tensor"] = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
        mp.theta_fiducial["moment_tensor"] = [1e14] * 6
        # No nuisance — simulate passing source_location directly via kwargs
        empty_samplers = {}
        theta = np.array([1e14] * 6)
        # source_location must be provided somehow — here as a kwarg
        result = _call_io_sim(
            mp, data_loader, empty_samplers, mock_sim, theta,
            source_location=np.array(_SOURCE_LOC),
        )
        assert result.ndim == 1

    def test_theta_1d_input_accepted(self, mt_parameters, data_loader, samplers, mock_sim):
        """input_output_simulation must handle 1-D theta without raising."""
        theta_1d = np.array([1e14] * 6)
        result = _call_io_sim(mt_parameters, data_loader, samplers, mock_sim, theta_1d)
        assert result.ndim == 1

    def test_theta_2d_input_accepted(self, mt_parameters, data_loader, samplers, mock_sim):
        """input_output_simulation must handle 2-D theta (1, n_params)."""
        theta_2d = np.array([[1e14] * 6])
        result = _call_io_sim(mt_parameters, data_loader, samplers, mock_sim, theta_2d)
        assert result.ndim == 1

    def test_different_theta_values_produce_same_output_shape(
        self, mt_parameters, data_loader, samplers, mock_sim
    ):
        """With a mock simulator returning constant traces, shape never varies."""
        theta_a = np.array([1e14] * 6)
        theta_b = np.array([5e13] * 6)
        r_a = _call_io_sim(mt_parameters, data_loader, samplers, mock_sim, theta_a)
        r_b = _call_io_sim(mt_parameters, data_loader, samplers, mock_sim, theta_b)
        assert r_a.shape == r_b.shape
