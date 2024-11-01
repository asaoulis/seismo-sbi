""" Configuration parser for sbi_pipeline.
"""

import yaml
from math import log10
from functools import partial
from copy import copy

from seismo_sbi.plotting.parameters import ParameterInformation, DegreeKMConverter, DegreeType
from seismo_sbi.instaseis_simulator.receivers import Receivers
from seismo_sbi.sbi.parameters import ModelParameters, PipelineParameters, \
    SimulationParameters, DatasetGenerationParameters, TestJobs

class InvalidConfiguration(Exception):
    pass

class SBI_Configuration:

    parameter_types = ["source_location", "earthquake_magnitude", "moment_tensor"]

    param_names_map = {"source_location": ["latitude", "longitude", "depth", "time_shift"],
                        "moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
                        "earthquake_magnitude": ["earthquake_magnitude"]}

    compression_types = ["optimal_score", "second_order_score", "multi_optimal_score", "ml_compressor"]
    test_noise_models = ['gaussian_noises', 'real_noise', 'empirical_gaussian']

    
    def __init__(self) -> None:

        self.pipeline_parameters = None

        self.model_parameters = ModelParameters()
        self.sim_parameters = None
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
        for name, parsing_callable in self._parsing_callables:
            if name == 'job_options':
                subconfig = {key: value for key, value in config.items() if not isinstance(value, dict)}
            else:
                subconfig = config[name]
            parsing_callable(subconfig)
    
    def parse_main_options(self, config):
        # parse top level options 

        self.pipeline_parameters = PipelineParameters(**config)

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
                self.model_parameters.nuisance[parameter_type] = parameter_values["fiducial"]
                self.model_parameters.bounds[parameter_type] = parameter_values['bounds']
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
        self.dataset_parameters = DatasetGenerationParameters(**simulations_config)
        
    
    def parse_seismic_context(self, config):
        seismic_context_config = copy(config)
        receivers_details = seismic_context_config.pop("stations_path")
        receiver_component_details = seismic_context_config.pop("station_components_path")
        seismic_context_config["receivers"] = Receivers(receivers_details, receiver_component_details)

        self.sim_parameters = SimulationParameters(**seismic_context_config)

    def parse_compression_options(self, config):

        compression_config = config
        self.compression_methods = []
        for compression_type, options in compression_config.items():
            if compression_type in SBI_Configuration.compression_types:
                self.compression_methods.append((compression_type, options))
            else:
                allowed_types = ', '.join(SBI_Configuration.compression_types)
                raise InvalidConfiguration(f"Invalid compression type {compression_type}. Only [ {allowed_types} ] allowed")
                
    
    
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