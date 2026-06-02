""" Configuration parser for sbi_pipeline.
"""

import yaml
from math import log10
from functools import partial
from copy import copy

from seismo_sbi.plotting.parameters import ParameterInformation, DegreeKMConverter, DegreeType
from seismo_sbi.instaseis_simulator.receivers import Receivers
from seismo_sbi.sbi.types.parameters import ModelParameters, PipelineParameters, \
    SimulationParameters, DatasetGenerationParameters, TestJobs, IterativeLeastSquaresParameters
from seismo_sbi.cps_simulator.compatibility import load_velocity_model
from seismo_sbi.instaseis_simulator.post_processing import (
    AUGMENTABLE_EFFECT_KEYS,
    POST_NOISE_EFFECT_KEYS,
)
from seismo_sbi.priors.catalogue import load_catalogue
from seismo_sbi.priors.samplers import (
    make_catalogue_location_sampler,
    make_gutenberg_richter_mt_sampler,
)

#: Catalogue-driven sampler factories selectable via a dict-form
#: ``simulations.sampling_method`` entry (``type: <name>`` + factory kwargs).
SAMPLER_FACTORIES = {
    "catalogue_kde": make_catalogue_location_sampler,
    "gutenberg_richter": make_gutenberg_richter_mt_sampler,
}


class InvalidConfiguration(Exception):
    pass

class SBI_Configuration:

    parameter_types = [
        # Core source parameters
        "source_location", "earthquake_magnitude", "moment_tensor", "velocity_model",
        # Simulator-level nuisance (Category 1 — modify forward model inputs)
        "stf_duration",
        # Post-processing nuisance (Category 2 — modify synthetic seismograms)
        "amplitude_error", "instrument_dropout", "scattering_coda", "time_shift_error",
        # Post-noise augmentation (applied after sensor noise; see ComponentDropoutEffect)
        "component_dropout",
    ]

    param_names_map = {
        "source_location": ["latitude", "longitude", "depth", "time_shift"],
        "moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
        "earthquake_magnitude": ["earthquake_magnitude"],
        "velocity_model": ["velocity_model"],
        # New nuisance types — single scalar per entry
        "stf_duration": ["stf_duration"],
        "amplitude_error": ["amplitude_error"],
        "instrument_dropout": ["instrument_dropout"],
        "scattering_coda": ["scattering_coda"],
        "time_shift_error": ["time_shift_error"],
        "component_dropout": ["component_dropout"],
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

        self.model_parameters = ModelParameters()
        self.sim_parameters = None
        # compression_methods is a list of (full_key, options) where full_key is
        # the final compressor name used everywhere, e.g. 'optimal_score_filtered_block'
        self.compression_methods = []

        self.dataset_parameters = None

        self.sbi_method = None
        self.pipeline_type = None
        self.sbi_noise_model = None

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

    def parse_config_file(self, config_file):
        # read yaml config file
        with open(config_file, 'r', encoding = 'utf-8') as stream:
            config = yaml.safe_load(stream)

        self.process_configuration_data(config)

    def process_configuration_data(self, config):
        # Retain the raw parsed YAML so downstream tooling can read top-level blocks
        # (e.g. `ml_scaler`, `ml_architecture`) without re-opening the file. Keeps the
        # training-time and inference-time scaler choice in sync via build_flexible_scaler.
        self.raw_config = config
        for name, parsing_callable in self._parsing_callables.items():
            if name == 'job_options':
                subconfig = {key: value for key, value in config.items() if not(isinstance(value, dict) or isinstance(value, list))}
            else:
                subconfig = config[name]
            parsing_callable(subconfig)
    
    def parse_main_options(self, config):
        # parse top level options. Filter to PipelineParameters' known fields so that
        # extra top-level scalar keys (e.g. `ml_architecture`, read separately from the
        # raw YAML by train_NPE.py) do not break construction.
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

                # Optional `stage` key selects where the nuisance is injected:
                # "simulation" (default — baked into the simulation dataset) or
                # "training_augmentation" (folded in per-batch in the ML dataloader).
                stage = parameter_values.get("stage", "simulation")
                if stage not in SBI_Configuration._NUISANCE_STAGES:
                    allowed = ', '.join(sorted(SBI_Configuration._NUISANCE_STAGES))
                    raise InvalidConfiguration(
                        f"Invalid stage {stage!r} for nuisance {parameter_type}. "
                        f"Only [ {allowed} ] allowed"
                    )
                if stage == "training_augmentation" and parameter_type not in AUGMENTABLE_EFFECT_KEYS:
                    allowed = ', '.join(AUGMENTABLE_EFFECT_KEYS)
                    raise InvalidConfiguration(
                        f"Nuisance {parameter_type} cannot use stage 'training_augmentation' "
                        f"(only Category-2 post-processing effects [ {allowed} ] are "
                        f"augmentation-eligible; simulator-level nuisances must be baked in)."
                    )
                if stage == "training_augmentation_post_noise" and parameter_type not in POST_NOISE_EFFECT_KEYS:
                    allowed = ', '.join(POST_NOISE_EFFECT_KEYS)
                    raise InvalidConfiguration(
                        f"Nuisance {parameter_type} cannot use stage "
                        f"'training_augmentation_post_noise' (only post-noise effects "
                        f"[ {allowed} ] are eligible)."
                    )
                # component_dropout zeros channels to mimic genuinely-absent components, which
                # must be EXACTLY zero — a pre-noise/simulation stage would leave `0 + noise`.
                # So it is only valid post-noise.
                if parameter_type in POST_NOISE_EFFECT_KEYS and stage != "training_augmentation_post_noise":
                    raise InvalidConfiguration(
                        f"Nuisance {parameter_type} must use stage "
                        f"'training_augmentation_post_noise' (it zeros channels after noise so "
                        f"they are exactly zero); got stage {stage!r}."
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

        String entries pass through unchanged (looked up in
        ``DatasetGenerator.sampler_lookup_map`` later). A dict entry selects a
        catalogue-driven prior: its ``type`` names a factory in
        :data:`SAMPLER_FACTORIES`, any ``catalogue`` path is loaded once into an
        ``EventCatalogue``, and the factory is called to build the
        ``(args, num_samples)`` closure (so the heavy I/O — catalogue load and
        b-value fit — happens a single time, at parse time).
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

        self.sim_parameters = SimulationParameters(**seismic_context_config)

    def parse_compression_options(self, config):

        """Parse compression section into fully-qualified compressor keys.

        Example YAML:

        compression:
          - optimal_score:
              filtered_block: '/path/to/noise'
          - optimal_score:
              empirical_diagonal: '/path/to/noise'

        becomes

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
            scale_string = str(round(log10(scale) - 1))
            moment_tensor_components = ["rr", "\\theta \\theta", "\\phi \\phi", "r \\theta", "r \\phi", "\\theta \\phi"]
            self.model_parameters.information[parameter_type] = [
                    # ParameterInformation(f"$m_{{{mt_component}}}$", f"$\\times 10^{{{scale_string}}} Nm$", moment_tensor_scaler)
                    ParameterInformation(f"$M_{{{mt_component}}}$", f"", moment_tensor_scaler)
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