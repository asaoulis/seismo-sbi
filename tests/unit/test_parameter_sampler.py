"""A velocity-model prior reads its reference model once and draws a fresh perturbation each time."""
import numpy as np

from seismo_sbi.priors import parameter_sampler
from seismo_sbi.priors.parameter_sampler import ParameterSampler
from seismo_sbi.sbi.types.parameters import ModelParameters


def test_the_reference_velocity_model_is_read_once_for_every_draw(monkeypatch):
    reads = []
    reference_model = np.vstack([np.full(3, 2.0), np.full(3, 6.0), np.full(3, 3.5), np.full(3, 2.7),
                                 np.full(3, 500.0), np.full(3, 250.0)])
    monkeypatch.setattr(parameter_sampler, "load_velocity_model", lambda path: reads.append(path) or reference_model)
    parameters = ModelParameters()
    parameters.nuisance = {"velocity_model": reference_model}
    parameters.bounds = {"velocity_model": ["model.txt", 5]}

    sampler = ParameterSampler.from_configuration(parameters, {"velocity_model": "velocity model"})
    draws = [np.asarray(sampler.draw_nuisance_inputs()["velocity_model"], dtype=float) for _ in range(3)]

    assert reads == ["model.txt"]
    assert not np.allclose(draws[0], draws[1]) and np.allclose(reference_model[1], 6.0)
