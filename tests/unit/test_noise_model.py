"""The training noise model block: what it accepts, when it follows an event, and the samplers it builds."""
import pytest

from seismo_sbi.sbi.noises.noise_model import NoiseModelConfiguration, build_noise_sampler, build_test_noise_samplers
from seismo_sbi.utils.errors import InvalidConfiguration


@pytest.mark.parametrize("block, follows", [
    ({"type": "gaussian", "noise_level": 1e-6}, False),
    ({"type": "gaussian_filtered"}, True),
    ({"type": "empirical_gaussian"}, True),
    ({"type": "real_noise", "noise_catalogue_path": "/x"}, True),
    ({"type": "real_noise", "noise_catalogue_path": "/x", "rescale": False}, False),
])
def test_recorded_and_covariance_noise_follow_the_event_unless_rescale_is_off(block, follows):
    assert NoiseModelConfiguration.from_yaml_block(block).follows_event is follows


@pytest.mark.parametrize("block, message", [
    ({"type": "laplace", "noise_level": 1.0}, "noise_model type 'laplace'"),
    ({"type": "gaussian", "noise_level": 1.0, "noise_levle": 2.0}, "unknown keys \\['noise_levle'\\]"),
    ({"type": "gaussian"}, "needs noise_level"),
    ({"type": "real_noise"}, "needs noise_catalogue_path"),
    ({"type": "real_noise", "noise_catalogue_path": "/x", "allow_incomplete": True}, "allow_incomplete needs rescale: false"),
])
def test_an_inconsistent_noise_model_is_rejected_when_parsed(block, message):
    with pytest.raises(InvalidConfiguration, match=message):
        NoiseModelConfiguration.from_yaml_block(block)


@pytest.mark.parametrize("noise_type", ["gaussian_filtered", "empirical_gaussian"])
def test_covariance_noise_needs_a_loaded_covariance(noise_type):
    noise_model = NoiseModelConfiguration(noise_type)
    with pytest.raises(InvalidConfiguration, match="none is loaded"):
        build_noise_sampler(noise_model, object(), 100, 100)
    with pytest.raises(InvalidConfiguration, match="none is loaded"):
        build_test_noise_samplers([(noise_type, None)], noise_model, object(), 100, 100)
