"""Which training noise models are rescaled to a real event's pre-event variance."""
from types import SimpleNamespace

import pytest

from seismo_sbi.sbi.training_data import rescale_training_noise_to_event
from seismo_sbi.utils.errors import InvalidConfiguration


def configuration(noise_model, real_event_jobs=None):
    return SimpleNamespace(sbi_noise_model=noise_model, real_event_jobs=real_event_jobs or {})


def test_white_gaussian_noise_trains_without_a_real_event():
    rescale_training_noise_to_event(None, configuration({"type": "gaussian", "noise_level": 1e-6}))


def test_frozen_real_noise_trains_without_a_real_event():
    rescale_training_noise_to_event(None, configuration({"type": "real_noise", "rescale": False}))


def test_a_rescaled_noise_model_needs_a_real_event():
    with pytest.raises(InvalidConfiguration, match="real_events"):
        rescale_training_noise_to_event(None, configuration({"type": "gaussian_filtered"}))
