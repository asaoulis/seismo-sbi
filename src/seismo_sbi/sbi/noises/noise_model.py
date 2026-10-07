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
        unknown = sorted(set(block) - set(cls.__dataclass_fields__))
        if unknown:
            raise InvalidConfiguration(f"inference.sbi.noise_model: unknown keys {unknown}; allowed: "
                                       f"{sorted(cls.__dataclass_fields__)}.")
        return cls(**block)

    def __post_init__(self):
        if self.type not in NOISE_MODEL_TYPES:
            raise InvalidConfiguration(
                f"Unknown inference.sbi.noise_model type {self.type!r}: expected one of "
                "'gaussian', 'gaussian_filtered', 'real_noise' or 'empirical_gaussian'.")
        if self.type == "gaussian" and self.noise_level is None:
            raise InvalidConfiguration("inference.sbi.noise_model: type 'gaussian' needs noise_level (m).")
        if self.type == "real_noise" and not self.noise_catalogue_path:
            raise InvalidConfiguration("inference.sbi.noise_model: type 'real_noise' needs noise_catalogue_path.")

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
        return _sampler_of(data_covariance, "inference.sbi.noise_model type 'gaussian_filtered'")
    elif train_noise_type == 'real_noise':
        return RealNoiseSampler(simulation_parameters, noise_model.noise_catalogue_path, trace_length,
                                allow_incomplete=noise_model.allow_incomplete)
    return _sampler_of(empirical_covariance, "inference.sbi.noise_model type 'empirical_gaussian'")


def _sampler_of(covariance, what):
    """``covariance``'s sampler; ``what`` names the noise model in the error when there is none."""
    if covariance is None:
        raise InvalidConfiguration(f"{what} draws from a compressor's covariance, and none is loaded: "
                                   "load the compressors first.")
    return covariance.create_sampler()


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
            test_noises[noise_type] = _sampler_of(data_covariance, "jobs.noise_models gaussian_filtered")
        elif noise_type == "real_noise":
            noise_catalogue_path = noise_options
            test_noises[noise_type] = RealNoiseSampler(simulation_parameters, noise_catalogue_path, trace_length)
        elif noise_type == "empirical_gaussian":
            test_noises[noise_type] = _sampler_of(empirical_covariance, "jobs.noise_models empirical_gaussian")
    return test_noises
