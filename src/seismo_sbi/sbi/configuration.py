"""Parse a pipeline YAML file into typed parameter records.

:class:`SBI_Configuration` reads the main options, the parameters with their priors and fiducial
values, the simulation and sampling options, the compression and SBI settings, and the test
jobs, into the records of :mod:`seismo_sbi.sbi.types.parameters`.
"""

import yaml
from functools import partial
from copy import copy

from seismo_sbi.sbi.types.parameter_labels import ParameterInformation, DegreeKMConverter, DegreeType
from seismo_sbi.simulators.receivers import Receivers
from seismo_sbi.sbi.types.parameters import ModelParameters, PipelineParameters, \
    SimulationParameters, DatasetGenerationParameters, TestJobs, IterativeLeastSquaresParameters
from seismo_sbi.simulators.cps.compatibility import load_velocity_model
from seismo_sbi.nuisance_effects.post_processing import (
    AUGMENTABLE_EFFECT_KEYS,
    POST_NOISE_EFFECT_KEYS,
    CONDITIONING_AUGMENTABLE_KEYS,
)
from seismo_sbi.priors.catalogue import load_catalogue
from seismo_sbi.priors.samplers import (
    make_catalogue_location_sampler,
    make_gutenberg_richter_mt_sampler,
)
from seismo_sbi.sbi.training_configuration import TrainingConfiguration
from seismo_sbi.utils.errors import InvalidConfiguration

#: Catalogue-driven sampler factories selectable via a dict-form
#: ``simulations.sampling_method`` entry (``type: <name>`` + factory kwargs).
SAMPLER_FACTORIES = {
    "catalogue_kde": make_catalogue_location_sampler,
    "gutenberg_richter": make_gutenberg_richter_mt_sampler,
}


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

    #: Valid values for a nuisance block's optional ``stage`` key.
    _NUISANCE_STAGES = frozenset({
        "simulation", "training_augmentation", "training_augmentation_post_noise",
    })

    compression_types = ["optimal_score", "theory_optimal_score", "second_order_score", "multi_optimal_score", "ml_compressor"]
    test_noise_models = ['gaussian_noises', 'real_noise', 'empirical_gaussian', 'gaussian_filtered']

    
    def __init__(self) -> None:

        self.pipeline_parameters = None
        self.training = TrainingConfiguration()

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
    def from_file(cls, config_file):
        """Parse ``config_file`` into a configuration object."""
        configuration = cls()
        configuration.parse_config_file(config_file)
        return configuration

    def parse_config_file(self, config_file):
        """Read ``config_file`` (YAML) and parse every block."""
        # read yaml config file
        with open(config_file, 'r', encoding = 'utf-8') as stream:
            config = yaml.safe_load(stream)

        self.process_configuration_data(config)

    def process_configuration_data(self, config):
        # `raw_config` is kept for the parameter scaler, which must be rebuilt from the same
        # file at inference time (see build_flexible_scaler).
        self.raw_config = config
        self.training = TrainingConfiguration.from_yaml_block(config)
        for name, parsing_callable in self._parsing_callables.items():
            if name == 'job_options':
                subconfig = {key: value for key, value in config.items() if not(isinstance(value, dict) or isinstance(value, list))}
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
            if parameter_type in SBI_Configuration.parameter_types:
                parameter_values =parameters_config[parameter_type] 
                self._unpack_parameter_values(parameter_type, parameter_values )
                self._add_parameter_information(parameter_type, parameter_values['bounds'])
            else:
                allowed_types = ', '.join(SBI_Configuration.parameter_types)
                raise InvalidConfiguration(f"Invalid parameter type {parameter_type}. Only [ {allowed_types} ] allowed")
        
        nuisance_config = config["nuisance"]
        for parameter_type in nuisance_config.keys():
            if parameter_type in SBI_Configuration.parameter_types:
                parameter_values = nuisance_config[parameter_type]
                if parameter_type == 'velocity_model':
                    self.model_parameters.nuisance[parameter_type] = load_velocity_model(parameter_values["fiducial"])
                else:
                    self.model_parameters.nuisance[parameter_type] = parameter_values["fiducial"]
                self.model_parameters.bounds[parameter_type] = parameter_values['bounds']

                # ``stage``: "simulation" (default, baked into the dataset) or "training_augmentation"
                # (applied per batch in the dataloader).
                stage = parameter_values.get("stage", "simulation")
                if stage not in SBI_Configuration._NUISANCE_STAGES:
                    allowed = ', '.join(sorted(SBI_Configuration._NUISANCE_STAGES))
                    raise InvalidConfiguration(
                        f"Invalid stage {stage!r} for nuisance {parameter_type}. "
                        f"Only [ {allowed} ] allowed"
                    )
                if (stage == "training_augmentation"
                        and parameter_type not in AUGMENTABLE_EFFECT_KEYS
                        and parameter_type not in CONDITIONING_AUGMENTABLE_KEYS):
                    allowed = ', '.join(AUGMENTABLE_EFFECT_KEYS + CONDITIONING_AUGMENTABLE_KEYS)
                    raise InvalidConfiguration(
                        f"Nuisance {parameter_type} cannot use stage 'training_augmentation' "
                        f"(only Category-2 post-processing effects + conditioning-vector "
                        f"augmentations [ {allowed} ] are augmentation-eligible; simulator-level "
                        f"nuisances must be baked in)."
                    )
                if stage == "training_augmentation_post_noise" and parameter_type not in POST_NOISE_EFFECT_KEYS:
                    allowed = ', '.join(POST_NOISE_EFFECT_KEYS)
                    raise InvalidConfiguration(
                        f"Nuisance {parameter_type} cannot use stage "
                        f"'training_augmentation_post_noise' (only post-noise effects "
                        f"[ {allowed} ] are eligible)."
                    )
                # Component dropout must leave exact zeros, so it is valid only after noise is added.
                if parameter_type in POST_NOISE_EFFECT_KEYS and stage != "training_augmentation_post_noise":
                    raise InvalidConfiguration(
                        f"Nuisance {parameter_type} must use stage "
                        f"'training_augmentation_post_noise' (it zeros channels after noise so "
                        f"they are exactly zero); got stage {stage!r}."
                    )
                # Conditioning augmentations perturb the dataloader's conditioning vector, which the simulator
                # cannot bake, so they are valid only as training_augmentation.
                if parameter_type in CONDITIONING_AUGMENTABLE_KEYS and stage != "training_augmentation":
                    raise InvalidConfiguration(
                        f"Nuisance {parameter_type} must use stage 'training_augmentation' (it "
                        f"perturbs the source-location conditioning vector in the ML dataloader); "
                        f"got stage {stage!r}."
                    )
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
                allowed_types = ', '.join(SBI_Configuration.parameter_types)
                raise InvalidConfiguration(f"Invalid parameter type {parameter_type}. Only [ {allowed_types} ] allowed")
    
    def _unpack_parameter_values(self, parameter_type, parameter_values):

        self.model_parameters.names[parameter_type] = SBI_Configuration.param_names_map[parameter_type]
        
        self.model_parameters.theta_fiducial[parameter_type] = parameter_values["fiducial"]
        self.model_parameters.stencil_deltas[parameter_type] = parameter_values['stencil_deltas']
        self.model_parameters.bounds[parameter_type] = parameter_values['bounds']

    def parse_simulations_options(self, config):
        simulations_config = config
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

        String entries pass through unchanged (looked up in ``DatasetGenerator.sampler_lookup_map``
        later). A dict entry selects a catalogue-driven prior: its ``type`` names a factory in
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
                ("optimal_score_filtered_block", {"type": "optimal_score", "covariance": "filtered_block", "path": "/path/to/noise"}),
                ("optimal_score_empirical_diagonal", {"type": "optimal_score", "covariance": "empirical_diagonal", "path": "/path/to/noise"}),
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

        for raw_type, raw_options in compression_list:
            if raw_type not in SBI_Configuration.compression_types:
                allowed_types = ', '.join(SBI_Configuration.compression_types)
                raise InvalidConfiguration(f"Invalid compression type {raw_type}. Only [ {allowed_types} ] allowed")

            # For covariance-based compressors (e.g. optimal_score) we expect a
            # single-entry dict giving the covariance option and its path.
            if raw_type == "optimal_score":
                if not isinstance(raw_options, dict) or len(raw_options) != 1:
                    raise InvalidConfiguration(
                        "optimal_score entries must be of the form:\n"
                        "  - optimal_score:\n      <covariance_option>: <path>"
                    )
                cov_name, cov_path = list(raw_options.items())[0]
                full_key = f"{raw_type}_{cov_name}"
                options = {"type": raw_type, "covariance": cov_name, "path": cov_path}
                self.compression_methods.append((full_key, options))
            else:
                # Non-covariance compressors keep their raw options and use the
                # raw type as the full key.
                full_key = raw_type
                options = {"type": raw_type, **(raw_options or {})}
                self.compression_methods.append((full_key, options))

    def parse_sbi_config(self, config):
        inference_config = config
        self.sbi_method = inference_config["sbi"]["method"]
        self.pipeline_type = inference_config["sbi"].get("pipeline", "single_event")
        self.sbi_noise_model = inference_config["sbi"]["noise_model"]
        self.sbi_seed = inference_config["sbi"].get("seed")
        self.likelihood_config = inference_config["likelihood"]

    def parse_jobs_config(self, config):
        jobs_config = config
        self.test_noise_models = []

        test_simulations_config = jobs_config["simulations"]
        self.test_job_simulations = TestJobs(**test_simulations_config)

        self.plotting_options = jobs_config["plots"]

        for noise_model, options in jobs_config["noise_models"].items():
            if noise_model not in SBI_Configuration.test_noise_models:
                allowed_types = ', '.join(SBI_Configuration.test_noise_models)
                raise InvalidConfiguration(f"Invalid noise model {noise_model}. Only [ {allowed_types} ] allowed")
            else:
                self._append_to_noise_methods(noise_model, options)

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