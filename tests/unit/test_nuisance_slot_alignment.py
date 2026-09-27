"""Nuisance values reach the simulator in their own slot of the sample vector.

A `constant` nuisance must contribute exactly ``len(fiducial)`` slots — the number
``vector_to_simulation_inputs`` consumes — so that no parameter declared after it is read one
slot late. Pins the generation chain and the forward path against each other, dependency-free:
the simulator is a stub, no Instaseis or CPS.
"""

import numpy as np
import pytest
from copy import deepcopy

from seismo_sbi.sbi.dataset_generator import (
    DatasetGenerator,
    flatten_sample,
    transform_sampling_func,
)
from seismo_sbi.sbi.types.parameters import ModelParameters
from seismo_sbi.sbi.simulator_wrapper import GeneralSimulatorWrapper
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.simulation_io import SimulationDataLoader

from tests.unit.test_simulator_wrapper_io import MockSimulator

_SOURCE_LOC = [0.0, 0.0, 10.0, 0.0]
_STF_BOUNDS = [0.3, 1.2]

#: Constants both BEFORE and AFTER the one sampled nuisance: the bug shifted every
#: parameter following a constant, so an ordering with constants on one side only would miss it.
_NUISANCE_FIDUCIALS = {
    "source_location": _SOURCE_LOC,
    "time_shift_error": [1.0],
    "stf_duration": [0.9],
    "scattering_coda": [0.4],
    "amplitude_error": [0.2],
}
_SAMPLING_METHOD = {
    "moment_tensor": "uniform",
    "source_location": "constant",
    "time_shift_error": "constant",
    "stf_duration": "uniform",
    "scattering_coda": "constant",
    "amplitude_error": "constant",
}


def _parameters():
    params = ModelParameters()
    params.names["moment_tensor"] = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    params.theta_fiducial["moment_tensor"] = [1e14] * 6
    params.bounds["moment_tensor"] = [[-1e16] * 6, [1e16] * 6]
    for key, fiducial in _NUISANCE_FIDUCIALS.items():
        params.nuisance[key] = list(fiducial)
        params.bounds[key] = [0.0, 1.0]
    params.bounds["source_location"] = [[-1.0, -1.0, 0.0, -2.0], [1.0, 1.0, 20.0, 2.0]]
    params.bounds["stf_duration"] = _STF_BOUNDS
    return params


class _CapturingGenerator(DatasetGenerator):
    """Records the simulation inputs instead of running the forward model."""

    def __init__(self):
        super().__init__(lambda *args, **kwargs: None, output_base_path="/unused")
        self.captured = []

    def run_parallel_simulations(self, simulation_job_args_list):
        self.captured = [inputs for inputs, _ in simulation_job_args_list]


def _generated_inputs(num_samples=8, seed=0, sampling_method=None):
    params = _parameters()
    generator = _CapturingGenerator()
    np.random.seed(seed)
    generator.run_and_save_simulations(
        params, sampling_method or _SAMPLING_METHOD, (0, num_samples)
    )
    return generator.captured


def test_each_nuisance_receives_its_own_configured_value():
    for inputs in _generated_inputs():
        assert np.allclose(inputs["source_location"], _SOURCE_LOC)
        assert np.allclose(inputs["time_shift_error"], [1.0])
        assert np.allclose(inputs["scattering_coda"], [0.4])
        assert np.allclose(inputs["amplitude_error"], [0.2])


def test_sampled_nuisance_after_a_constant_spans_its_bounds():
    durations = np.array([inputs["stf_duration"][0] for inputs in _generated_inputs()])
    assert np.all((durations >= _STF_BOUNDS[0]) & (durations <= _STF_BOUNDS[1]))
    # The bug pinned this to the preceding constant's upper bound, i.e. one value for every draw.
    assert durations.std() > 0.0


def test_sample_vector_length_equals_the_register_slots():
    params = _parameters()
    samplers = DatasetGenerator.create_samplers(params, _SAMPLING_METHOD)
    draw = [next(sampler(1)) for sampler in samplers.values()]
    n_slots = 6 + sum(len(fiducial) for fiducial in _NUISANCE_FIDUCIALS.values())
    assert len(flatten_sample(draw)) == n_slots


def _forward_path_inputs(params, sampling_method):
    """The nuisance map ``input_output_simulation`` hands the simulator."""
    samplers = DatasetGenerator.create_samplers(params, sampling_method)
    receivers = Receivers(receivers=[Receiver(0.0, 0.0, "XX", "STA1", ["Z"])])
    simulator = MockSimulator(receivers)
    GeneralSimulatorWrapper.input_output_simulation(
        None, params, SimulationDataLoader(components=["Z"], receivers=receivers),
        samplers, simulator, np.array(params.theta_fiducial["moment_tensor"]),
    )
    return simulator.received_inputs


def test_generation_and_forward_paths_agree_on_the_nuisance_map():
    # Every nuisance constant, so the map is fully determined and the two paths must match
    # value for value; a slot shift would move a value into the neighbouring nuisance.
    constants_only = {**_SAMPLING_METHOD, "stf_duration": "constant"}
    params = _parameters()
    generated = _generated_inputs(num_samples=1, sampling_method=constants_only)[0]
    forward = _forward_path_inputs(params, constants_only)

    for key, fiducial in _NUISANCE_FIDUCIALS.items():
        assert np.allclose(generated[key], fiducial), key
        assert np.allclose(forward[key], generated[key]), key


def test_forward_path_shapes_match_generation_when_a_nuisance_is_sampled():
    params = _parameters()
    generated = _generated_inputs(num_samples=1)[0]
    forward = _forward_path_inputs(params, _SAMPLING_METHOD)

    for key in params.nuisance:
        assert np.shape(forward[key]) == np.shape(generated[key]), key
    duration = np.asarray(forward["stf_duration"])[0]
    assert _STF_BOUNDS[0] <= duration <= _STF_BOUNDS[1]
