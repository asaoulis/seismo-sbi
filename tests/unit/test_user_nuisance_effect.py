"""A nuisance effect written outside the library, registered once, runs at either stage a
configuration puts it at."""
from types import SimpleNamespace

import numpy as np
import pytest

import seismo_sbi.sbi.simulator_wrapper as simulator_wrapper
from seismo_sbi.nuisance_effects.post_processing import (
    EFFECT_REGISTRY, build_augmentation_chain_from_parameters, register_nuisance_effect)
from seismo_sbi.nuisance_effects.seismogram_effect import SeismogramEffect
from seismo_sbi.sbi.configuration import SBI_Configuration

KEY = "station_gain_error"


class StationGainEffect(SeismogramEffect):
    """Multiplies every trace by ``gain`` when the key is active."""

    def __init__(self, gain=2.0):
        self.gain = gain

    def __call__(self, seismograms_map, receivers, **nuisance_params):
        if not nuisance_params.get(KEY):
            return seismograms_map
        return {station: {component: trace * self.gain for component, trace in components.items()}
                for station, components in seismograms_map.items()}


@pytest.fixture
def registered():
    register_nuisance_effect(KEY, StationGainEffect, stages=("simulation", "training_augmentation"))
    yield
    EFFECT_REGISTRY.pop(KEY)


def parsed(stage):
    config = SBI_Configuration()
    config.parse_parameters({
        "inference": {"moment_tensor": {"fiducial": [1e13] * 6, "stencil_deltas": [1e10] * 6,
                                        "bounds": [[-5e13] * 6, [5e13] * 6]}},
        "nuisance": {KEY: {"fiducial": [1.0], "bounds": [0.0, 1.0], "gain": 3.0, "stage": stage}},
    })
    return config.model_parameters


def test_a_registered_effect_runs_at_simulation(registered, monkeypatch):
    built = []
    monkeypatch.setattr(simulator_wrapper, "build_simulator",
                        lambda config, parameters, effects, data_flattening=None:
                        built.extend(effects) or SimpleNamespace(execute_sim_and_save_outputs=None))
    simulator_wrapper.GeneralSimulatorWrapper(
        SimpleNamespace(simulation_type="instaseis", sampling_rate=1.0), parsed("simulation"),
        SimpleNamespace(convert_sim_data_to_array=None), {})

    assert [type(effect) for effect in built] == [StationGainEffect]
    assert built[0].gain == 3.0


def test_a_registered_effect_runs_as_a_training_augmentation(registered):
    chain, nuisance_params = build_augmentation_chain_from_parameters(parsed("training_augmentation"))

    assert nuisance_params == {KEY: 1.0}
    traces = {"AAA": {"Z": np.ones(4)}}
    np.testing.assert_array_equal(chain(traces, None, nuisance_params)["AAA"]["Z"], 3.0 * np.ones(4))


def test_a_registered_effect_is_refused_at_a_stage_it_was_not_registered_for(registered):
    with pytest.raises(Exception, match="cannot use stage"):
        parsed("training_augmentation_post_noise")


def test_registration_refuses_a_taken_key_an_unknown_stage_and_a_non_effect():
    with pytest.raises(ValueError, match="already registered"):
        register_nuisance_effect("time_shift_error", StationGainEffect)
    with pytest.raises(ValueError, match="stages"):
        register_nuisance_effect(KEY, StationGainEffect, stages=("training",))
    with pytest.raises(ValueError, match="SeismogramEffect"):
        register_nuisance_effect(KEY, dict)
    assert KEY not in EFFECT_REGISTRY
