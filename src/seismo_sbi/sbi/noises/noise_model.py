"""The training noise model (``inference.sbi.noise_model``) and the samplers built from it.

:class:`NoiseModelConfiguration` mirrors the YAML block. :func:`build_noise_sampler` returns the
sampler that adds noise to training simulations, and :func:`build_test_noise_samplers` the samplers
of the test jobs (``jobs.noise_models``).
"""
from dataclasses import dataclass

from seismo_sbi.sbi.noises.noise_samplers import WhiteNoiseSampler
from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
from seismo_sbi.utils.errors import InvalidConfiguration

NOISE_MODEL_TYPES = ("gaussian", "gaussian_filtered", "real_noise", "empirical_gaussian")


@dataclass(frozen=True)
class NoiseModelConfiguration:
    """``inference.sbi.noise_model``: the noise added to every training simulation.

    ``gaussian`` is white noise of standard deviation ``noise_level`` (m). ``gaussian_filtered`` and
    ``empirical_gaussian`` draw from the band-passed and the empirical covariance of the compressors.
    ``real_noise`` draws recorded windows from ``noise_catalogue_path``; with ``rescale`` they are
    scaled to an event's pre-event variance, and ``allow_incomplete`` admits windows that lack stations.
    """

    type: str
    noise_level: float = None
    noise_catalogue_path: str = None
    rescale: bool = True
    allow_incomplete: bool = False

    @classmethod
    def from_yaml_block(cls, block):
        return cls(**{name: block[name] for name in cls.__dataclass_fields__ if name in block})

    @property
    def follows_event(self):
        """Whether the training noise is rescaled to the pre-event noise of a real event."""
        return (self.type in ("gaussian_filtered", "empirical_gaussian")
                or (self.type == "real_noise" and self.rescale))


def build_noise_sampler(noise_model, simulation_parameters, trace_length, data_vector_length,
                        data_covariance=None, empirical_covariance=None):
    """The training noise sampler of ``noise_model`` (:class:`NoiseModelConfiguration`).

    ``data_covariance`` and ``empirical_covariance`` are the compressors' band-passed and empirical
    covariances, which the ``gaussian_filtered`` and ``empirical_gaussian`` models draw from.
    """
    train_noise_type = noise_model.type
    if train_noise_type == 'gaussian':
        return WhiteNoiseSampler(noise_model.noise_level, data_vector_length)
    elif train_noise_type == 'gaussian_filtered':
        return data_covariance.create_sampler()
    elif train_noise_type == 'real_noise':
        if noise_model.allow_incomplete and noise_model.follows_event:
            raise InvalidConfiguration(
                "inference.sbi.noise_model: allow_incomplete needs rescale: false. Rescaling a noise "
                "window to an event needs the pre-event variance of every station, which an "
                "incomplete window lacks.")
        return RealNoiseSampler(simulation_parameters, noise_model.noise_catalogue_path, trace_length,
                                allow_incomplete=noise_model.allow_incomplete)
    elif train_noise_type == 'empirical_gaussian':
        return empirical_covariance.create_sampler()
    raise InvalidConfiguration(
        f"Unknown inference.sbi.noise_model type {train_noise_type!r}: expected one of "
        "'gaussian', 'gaussian_filtered', 'real_noise' or 'empirical_gaussian'.")


def build_test_noise_samplers(test_noise_models, noise_model, simulation_parameters, trace_length,
                              data_vector_length, data_covariance=None, empirical_covariance=None):
    """``{name: sampler}`` for each ``(type, options)`` of ``jobs.noise_models``.

    ``gaussian_noises`` entries are white noise at ``options`` times the training ``noise_level``.
    """
    test_noises = {}
    for noise_type, noise_options in test_noise_models:
        if noise_type == "gaussian_noises":
            train_noise_level = noise_model.noise_level
            noise_factor = noise_options
            test_noises[f"{noise_type}_x{noise_factor}"] = WhiteNoiseSampler(
                noise_factor * train_noise_level, data_vector_length)
        elif noise_type == "gaussian_filtered":
            test_noises[noise_type] = data_covariance.create_sampler()
        elif noise_type == "real_noise":
            noise_catalogue_path = noise_options
            test_noises[noise_type] = RealNoiseSampler(simulation_parameters, noise_catalogue_path, trace_length)
        elif noise_type == "empirical_gaussian":
            test_noises[noise_type] = empirical_covariance.create_sampler()
    return test_noises
