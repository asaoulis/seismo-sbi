"""Parse a pipeline YAML file into typed parameter records.

:class:`SBI_Configuration` reads the main options, the parameters with their priors and fiducial
values, the simulation and sampling options, the compression and SBI settings, and the test
jobs, into the records of :mod:`seismo_sbi.sbi.types.parameters`.
"""

import yaml
from dataclasses import dataclass
from functools import partial
from copy import copy

from seismo_sbi.sbi.types.parameter_labels import ParameterInformation, DegreeKMConverter, DegreeType
from seismo_sbi.simulators.receivers import Receivers
from seismo_sbi.sbi.types.parameters import PIPELINE_TYPES, ModelParameters, PipelineParameters, \
    SimulationParameters, DatasetGenerationParameters, TestJobs, IterativeLeastSquaresParameters
from seismo_sbi.simulators.cps.compatibility import load_velocity_model
from seismo_sbi.nuisance_effects.post_processing import CONDITIONING_AUGMENTABLE_KEYS, EFFECT_REGISTRY
from seismo_sbi.priors.catalogue import load_catalogue
from seismo_sbi.priors.samplers import (
    make_catalogue_location_sampler,
    make_gutenberg_richter_mt_sampler,
)
from seismo_sbi.sbi.training_configuration import TrainingConfiguration
from seismo_sbi.sbi.noises.noise_model import NoiseModelConfiguration
from seismo_sbi.sbi.scalers import ScalerConfiguration
from seismo_sbi.sbi.compression.compressor_options import compressor_options
from seismo_sbi.utils.errors import InvalidConfiguration

#: Catalogue-driven sampler factories selectable via a dict-form
#: ``simulations.sampling_method`` entry (``type: <name>`` + factory kwargs).
SAMPLER_FACTORIES = {
    "catalogue_kde": make_catalogue_location_sampler,
    "gutenberg_richter": make_gutenberg_richter_mt_sampler,
}


@dataclass(frozen=True)
class LikelihoodConfiguration:
    """``inference.likelihood``: the Gaussian-likelihood inversion run beside the SBI one.

    With ``run`` the posterior is sampled with ``num_samples`` samples after ``walker_burn_in`` steps,
    split between ``num_processes`` chains (None: the run's ``num_jobs``), under the compressor's
    ``covariance`` (``empirical``) or white noise of that standard deviation (m). ``ensemble`` uses
    emcee's affine-invariant ensemble, otherwise independent Gaussian-move chains of step ``move_size``
    (one value, or ``[burn-in step, sampling step]`` in unit-box coordinates). ``mle_use_best`` takes the
    highest-probability sample of an MCMC refinement of the MLE instead of the chain mean.
    """

    run: bool = False
    covariance: object = "empirical"
    ensemble: bool = True
    walker_burn_in: int = None
    num_samples: int = None
    move_size: object = None
    num_processes: int = None
    mle_use_best: bool = False

    @classmethod
    def from_yaml_block(cls, block):
        unknown = sorted(set(block) - set(cls.__dataclass_fields__))
        if unknown:
            raise InvalidConfiguration(f"inference.likelihood: unknown keys {unknown}; allowed: "
                                       f"{sorted(cls.__dataclass_fields__)}.")
        configuration = cls(**block)
        if configuration.run and (configuration.walker_burn_in is None or configuration.num_samples is None):
            raise InvalidConfiguration("inference.likelihood: run: true needs walker_burn_in and num_samples.")
        return configuration


@dataclass(frozen=True)
class PlotsConfiguration:
    """``jobs.plots``: whether the inversions are plotted (``disable_plotting``), whether the plots are made
    while the next job runs (``async_plotting``), and the posterior comparisons
    (``test_posteriors.chain_consumer``: one list of ``[compressor, test noise]`` pairs per figure)."""

    disable_plotting: bool = False
    async_plotting: bool = True
    chain_consumer: tuple = ()

    @classmethod
    def from_yaml_block(cls, block):
        block = dict(block or {})
        test_posteriors = block.pop("test_posteriors", None) or {}
        unknown = sorted(set(block) - {"disable_plotting", "async_plotting"}) + sorted(
            f"test_posteriors.{key}" for key in set(test_posteriors) - {"chain_consumer"})
        if unknown:
            raise InvalidConfiguration(f"jobs.plots: unknown keys {unknown}.")
        return cls(chain_consumer=tuple(test_posteriors.get("chain_consumer") or ()), **block)


class SBI_Configuration:
    """Every option of a pipeline YAML file, parsed once into typed records.

    Build it with :meth:`from_file`; the parsed blocks are attributes (``pipeline_parameters``,
    ``model_parameters``, ``sim_parameters``, ``dataset_parameters``, ``compression_methods``,
    ``sbi_method``, ``test_job_simulations``, ``training`` and the like).
    """

    parameter_types = [
        # Core source parameters
        "source_location", "earthquake_magnitude", "moment_tensor", "velocity_model",
        # Simulator-level nuisance (Category 1 — modify forward model inputs)
        "stf_duration",
        # Post-processing nuisance (Category 2 — modify synthetic seismograms)
        "amplitude_error", "instrument_dropout", "scattering_coda", "time_shift_error",
        # Per-octave dispersion-spread phase delays (DispersionSpreadEffect)
        "dispersion_spread",
        # Post-noise augmentation (applied after sensor noise; see ComponentDropoutEffect)
        "component_dropout",
        # Conditioning augmentation (perturbs the source-location CONDITIONING vector in the ML
        # dataloader, NOT the waveform; see CONDITIONING_AUGMENTABLE_KEYS).
        "source_location_error",
    ]

    param_names_map = {
        "source_location": ["latitude", "longitude", "depth", "time_shift"],
        "moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
        "earthquake_magnitude": ["earthquake_magnitude"],
        "velocity_model": ["velocity_model"],
        # Nuisance types: a single scalar per entry
        "stf_duration": ["stf_duration"],
        "amplitude_error": ["amplitude_error"],
        "instrument_dropout": ["instrument_dropout"],
        "scattering_coda": ["scattering_coda"],
        "time_shift_error": ["time_shift_error"],
        "dispersion_spread": ["dispersion_spread"],
        "component_dropout": ["component_dropout"],
        "source_location_error": ["source_location_error"],
    }

    #: YAML keys consumed by the parameter machinery — any other keys in a
    #: nuisance config block are treated as effect-level constructor kwargs
    #: and stored in ``ModelParameters.nuisance_effect_config``.
    _STANDARD_NUISANCE_KEYS = frozenset({"fiducial", "bounds", "stage"})


    test_noise_models = ['gaussian_noises', 'real_noise', 'empirical_gaussian', 'gaussian_filtered']

    
    def __init__(self) -> None:

        self.pipeline_parameters = None
        self.training = TrainingConfiguration()
        self.ml_scaler = ScalerConfiguration()

        self.model_parameters = ModelParameters()
        self.sim_parameters = None
        # compression_methods is a list of (full_key, options) where full_key is
        # the final compressor name used everywhere, e.g. 'optimal_score_filtered_block'
        self.compression_methods = []

        self.dataset_parameters = None

        self.sbi_method = None
        self.pipeline_type = None
        self.sbi_noise_model = None
        self.sbi_seed = None

        self.test_job_simulations = None
        self.real_event_jobs = []
        self.test_noise_models = []
        self.plotting_options = None

        self._parsing_callables = {'job_options': self.parse_main_options,
                                    'parameters': self.parse_parameters,
                                    'simulations': self.parse_simulations_options,
                                    'seismic_context' : self.parse_seismic_context,
                                    'compression': self.parse_compression_options,
                                    'inference': self.parse_sbi_config,
                                    'jobs': self.parse_jobs_config}

    @classmethod
    def known_parameter_types(cls):
        """Every parameter a configuration may name: :attr:`parameter_types` and every nuisance key
        with a registered effect."""
        return cls.parameter_types + [key for key in EFFECT_REGISTRY if key not in cls.parameter_types]

    @classmethod
    def from_file(cls, config_file, *, output_directory=None, database_path=None):
        """Parse ``config_file`` into a configuration object.

        ``output_directory`` replaces the configured ``output_directory`` and ``database_path`` the
        Instaseis database path (``seismic_context.syngine_address``); None keeps the file's value.
        """
        configuration = cls()
        configuration.parse_config_file(config_file)
        if output_directory is not None:
            configuration.pipeline_parameters = configuration.pipeline_parameters._replace(
                output_directory=str(output_directory))
        if database_path is not None:
            configuration.sim_parameters = configuration.sim_parameters._replace(
                syngine_address=str(database_path))
        return configuration

    def parse_config_file(self, config_file):
        """Read ``config_file`` (YAML) and parse every block."""
        # read yaml config file
        with open(config_file, 'r', encoding = 'utf-8') as stream:
            config = yaml.safe_load(stream)

        self.process_configuration_data(config)

    def process_configuration_data(self, config):
        self.raw_config = config
        self.training = TrainingConfiguration.from_yaml_block(config)
        self.ml_scaler = ScalerConfiguration.from_yaml_block(config)
        for name, parsing_callable in self._parsing_callables.items():
            if name == 'job_options':
                subconfig = {key: value for key, value in config.items() if not(isinstance(value, dict) or isinstance(value, list))}
            elif name == 'compression':
                subconfig = config.get(name) or {}
            else:
                subconfig = config[name]
            parsing_callable(subconfig)
    
    def parse_main_options(self, config):
        # Filter to PipelineParameters' known fields so that top-level scalar keys belonging
        # to another part of the configuration (e.g. `ml_architecture`) do not break it.
        known = {k: v for k, v in config.items() if k in PipelineParameters._fields}
        self.pipeline_parameters = PipelineParameters(**known)

    def parse_parameters(self, config):

        parameters_config = config["inference"]
        for parameter_type in parameters_config.keys():
            if parameter_type in SBI_Configuration.known_parameter_types():
                parameter_values =parameters_config[parameter_type] 
                self._unpack_parameter_values(parameter_type, parameter_values )
                self._add_parameter_information(parameter_type, parameter_values['bounds'])
            else:
                allowed_types = ', '.join(SBI_Configuration.known_parameter_types())
                raise InvalidConfiguration(f"Invalid parameter type {parameter_type}. Only [ {allowed_types} ] allowed")
        
        nuisance_config = config["nuisance"]
        for parameter_type in nuisance_config.keys():
            if parameter_type in SBI_Configuration.known_parameter_types():
                parameter_values = nuisance_config[parameter_type]
                if parameter_type == 'velocity_model':
                    self.model_parameters.nuisance[parameter_type] = load_velocity_model(parameter_values["fiducial"])
                else:
                    self.model_parameters.nuisance[parameter_type] = parameter_values["fiducial"]
                self.model_parameters.bounds[parameter_type] = parameter_values['bounds']

                stage = SBI_Configuration._nuisance_stage(parameter_type, parameter_values)
                self.model_parameters.nuisance_stage[parameter_type] = stage

                # Any YAML key beyond the standard keys is effect-level
                # configuration forwarded to the SeismogramEffect constructor.
                effect_cfg = {
                    k: v for k, v in parameter_values.items()
                    if k not in SBI_Configuration._STANDARD_NUISANCE_KEYS
                }
                if effect_cfg:
                    self.model_parameters.nuisance_effect_config[parameter_type] = effect_cfg
            else:
                allowed_types = ', '.join(SBI_Configuration.known_parameter_types())
                raise InvalidConfiguration(f"Invalid parameter type {parameter_type}. Only [ {allowed_types} ] allowed")
    
    @staticmethod
    def _nuisance_stage(parameter_type, parameter_values):
        """The ``stage`` a nuisance block asks for (``"simulation"`` when absent), checked against
        the stages its key allows: those of its registered effect, the training augmentation for a
        conditioning-vector nuisance, and the simulation for any other nuisance.
        """
        stage = parameter_values.get("stage", "simulation")
        if parameter_type in EFFECT_REGISTRY:
            allowed = EFFECT_REGISTRY[parameter_type].stages
        elif parameter_type in CONDITIONING_AUGMENTABLE_KEYS:
            allowed = ("training_augmentation",)
        else:
            allowed = ("simulation",)
        if stage not in allowed:
            raise InvalidConfiguration(
                f"Nuisance {parameter_type} cannot use stage {stage!r}; it can use "
                f"[ {', '.join(allowed)} ].")
        return stage

    def _unpack_parameter_values(self, parameter_type, parameter_values):

        self.model_parameters.names[parameter_type] = SBI_Configuration.param_names_map[parameter_type]
        
        self.model_parameters.theta_fiducial[parameter_type] = parameter_values["fiducial"]
        self.model_parameters.stencil_deltas[parameter_type] = parameter_values['stencil_deltas']
        self.model_parameters.bounds[parameter_type] = parameter_values['bounds']

    def parse_simulations_options(self, config):
        simulations_config = dict(config)
        if "iterative_least_squares" in simulations_config:
            simulations_config["iterative_least_squares"] = IterativeLeastSquaresParameters(**simulations_config["iterative_least_squares"])
        if "sampling_method" in simulations_config:
            simulations_config["sampling_method"] = self._normalise_sampling_method(
                simulations_config["sampling_method"]
            )
        self.dataset_parameters = DatasetGenerationParameters(**simulations_config)

    @staticmethod
    def _normalise_sampling_method(sampling_method):
        """Resolve dict-form ``sampling_method`` entries into built samplers.

        String entries pass through unchanged (looked up in
        ``seismo_sbi.priors.parameter_sampler.SAMPLERS`` later). A dict entry selects a catalogue-driven prior: its ``type`` names a factory in
        :data:`SAMPLER_FACTORIES`, any ``catalogue`` path is loaded into an ``EventCatalogue``, and
        the factory builds the ``(args, num_samples)`` sampler.
        """
        resolved = {}
        for key, value in sampling_method.items():
            if isinstance(value, str):
                resolved[key] = value
            elif isinstance(value, dict):
                cfg = dict(value)
                sampler_type = cfg.pop("type", None)
                if sampler_type not in SAMPLER_FACTORIES:
                    allowed = ', '.join(sorted(SAMPLER_FACTORIES))
                    raise InvalidConfiguration(
                        f"Unknown sampler type {sampler_type!r} for parameter "
                        f"{key!r}. Allowed dict-form types: [ {allowed} ]."
                    )
                if "catalogue" in cfg:
                    cfg["catalogue"] = load_catalogue(
                        cfg["catalogue"],
                        magnitude_type=cfg.get("magnitude_type"),
                    )
                resolved[key] = SAMPLER_FACTORIES[sampler_type](**cfg)
            else:
                raise InvalidConfiguration(
                    f"sampling_method[{key!r}] must be a string or a dict; "
                    f"got {type(value).__name__}."
                )
        return resolved
        
    
    def parse_seismic_context(self, config):
        seismic_context_config = copy(config)
        receivers_details = seismic_context_config.pop("stations_path")
        receiver_component_details = seismic_context_config.pop("station_components_path")
        receiver_time_shifts_details = seismic_context_config.pop("station_time_shifts_path", None)
        seismic_context_config["receivers"] = Receivers(receivers_details, receiver_component_details, receiver_time_shifts_details)
        processing = seismic_context_config["processing"]
        if "filter_sampling_rate" not in processing:
            raise InvalidConfiguration(
                "seismic_context.processing.filter_sampling_rate is required: the rate (Hz) the "
                "observed data are bandpassed at, which the synthetics are filtered at too.")
        processing["filter_sampling_rate"] = float(processing["filter_sampling_rate"])

        self.sim_parameters = SimulationParameters(**seismic_context_config)

    def parse_compression_options(self, config):

        """Parse compression section into fully-qualified compressor keys.

        Example YAML::

            compression:
              - optimal_score:
                  filtered_block: '/path/to/noise'
              - optimal_score:
                  empirical_diagonal: '/path/to/noise'

        becomes::

            self.compression_methods = [
                ("optimal_score_filtered_block", OptimalScoreOptions("filtered_block", "/path/to/noise")),
                ("optimal_score_empirical_diagonal", OptimalScoreOptions("empirical_diagonal", "/path/to/noise")),
            ]
        """
        compression_config = config
        self.compression_methods = []

        # Normalise YAML into a list of (raw_type, raw_options)
        if isinstance(compression_config, dict):
            compression_list = list(compression_config.items())
        else:
            compression_list = []
            for full_dict in compression_config:
                name = list(full_dict.keys())[0]
                options = full_dict[name]
                compression_list.append((name, options))

        self.compression_methods = [compressor_options(raw_type, raw_options)
                                    for raw_type, raw_options in compression_list]

    def parse_sbi_config(self, config):
        inference_config = config
        self.sbi_method = inference_config["sbi"]["method"]
        self.pipeline_type = inference_config["sbi"].get("pipeline", "single_event")
        if self.pipeline_type not in PIPELINE_TYPES:
            raise InvalidConfiguration(
                f"inference.sbi.pipeline must be one of {', '.join(PIPELINE_TYPES)}, not {self.pipeline_type!r}.")
        self.sbi_noise_model = NoiseModelConfiguration.from_yaml_block(inference_config["sbi"]["noise_model"])
        self.sbi_seed = inference_config["sbi"].get("seed")
        self.likelihood_config = LikelihoodConfiguration.from_yaml_block(inference_config["likelihood"])

    def parse_jobs_config(self, config):
        jobs_config = config
        self.test_noise_models = []

        test_simulations_config = jobs_config["simulations"]
        self.test_job_simulations = TestJobs(**test_simulations_config)

        self.plotting_options = PlotsConfiguration.from_yaml_block(jobs_config.get("plots"))

        for noise_model, options in jobs_config["noise_models"].items():
            if noise_model not in SBI_Configuration.test_noise_models:
                allowed_types = ', '.join(SBI_Configuration.test_noise_models)
                raise InvalidConfiguration(f"Invalid noise model {noise_model}. Only [ {allowed_types} ] allowed")
            else:
                self._append_to_noise_methods(noise_model, options)

        for event_name, event_job in jobs_config["real_events"].items():
            if isinstance(event_job, dict) and "priors" in event_job:
                raise InvalidConfiguration(
                    f"real_events.{event_name}.priors: a per-event prior is not supported; the prior is "
                    "the parameters' bounds with simulations.sampling_method.")
        self.real_event_jobs = jobs_config["real_events"]


    
    def _append_to_noise_methods(self, noise_model, noise_options):
        if noise_model == "gaussian_noises":
            for noise_level in noise_options:
                self.test_noise_models.append((noise_model, noise_level))
        else:
            self.test_noise_models.append((noise_model, noise_options))
    
    def _add_parameter_information(self, parameter_type, bounds):

        if parameter_type == "source_location":
            self.model_parameters.information[parameter_type]= [
                    ParameterInformation("$y$", "$km$", DegreeKMConverter(bounds[0][0], DegreeType.LATITUDE)),
                    ParameterInformation("$x$", "$km$", DegreeKMConverter(bounds[0][1], DegreeType.LONGITUDE)),
                    ParameterInformation("$z$", "$km$"),
                    ParameterInformation("$\Delta t$", "$s$")
                ]
        elif parameter_type == "moment_tensor":
            try:
                scale = bounds[0][0]
            except TypeError:
                scale = bounds[1]
            scale = nearest_power_of_ten(scale)
            moment_tensor_scaler = partial(generic_scaler_callable, scale)
            moment_tensor_components = ["rr", "\\theta \\theta", "\\phi \\phi", "r \\theta", "r \\phi", "\\theta \\phi"]
            self.model_parameters.information[parameter_type] = [
                    ParameterInformation(f"$M_{{{mt_component}}}$", "", moment_tensor_scaler)
                        for mt_component in moment_tensor_components
            ]
        elif parameter_type == "earthquake_magnitude":
            self.model_parameters.information[parameter_type] = [
                    ParameterInformation("magnitude", "Nm")
            ]
    
def generic_scaler_callable(scale, x):
    return 10*x/scale


def nearest_power_of_ten(number):
    import math
    # Calculate the exponent of the number in base 10
    exponent = math.floor(math.log10(abs(number)))
    
    # Calculate the nearest power of ten
    nearest_power = 10 ** exponent
    
    return nearest_power