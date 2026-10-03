"""Parameter records for the pipeline.

The ``NamedTuple`` records hold the pipeline, simulation, dataset and least-squares options;
:class:`ModelParameters` holds the inferred and nuisance parameters with their fiducial values,
bounds and priors, and converts between parameter dictionaries and vectors.
"""

import numpy as np
from typing import NamedTuple, List, Optional
from copy import deepcopy

from seismo_sbi.simulators.receivers import Receivers

#: The values of ``inference.sbi.pipeline``, each naming one pipeline class.
PIPELINE_TYPES = ("single_event", "multi_event", "vary_dataset_size", "mle_estimate")

class PipelineParameters(NamedTuple):
    run_name : str
    output_directory: str
    job_name: str
    generate_dataset : bool
    num_jobs : int

class TestJobs(NamedTuple):

    random_events : int
    fixed_events : List = []
    custom_events : List = []

class SimulationParameters(NamedTuple):

    receivers : Receivers
    components : str
    seismogram_duration : float
    syngine_address : str
    sampling_rate: float
    processing : dict
    simulation_type : str = "instaseis"
    cps_path: str = None
    cps_GFs_path : str = None
    cps_GFs_fiducial_path : str = None
    cps_multi_models_path: Optional[str] = None
    syngine_fiducial_address: Optional[str] = None
    #: Instaseis ensembles for 'instaseis_multi_ensemble', each
    #: ``{ensemble_dir, fiducial_dir, receivers: [station_name, ...]}``.
    instaseis_multi_models: Optional[list] = None
    #: Depth offset (km, positive down) added only at the Green's-function call, for a model
    #: whose free surface is not the catalogue's datum; 0.0 means they coincide.
    source_depth_offset_km: float = 0.0
    #: ``"peak"`` or ``"onset"``: whether the source time is the centroid or the start of the
    #: moment-rate function; None is the forward model's own convention.
    stf_alignment: Optional[str] = None
    #: Each station draws its own 1-D ensemble member per event, instead of one shared member.
    resample_member_per_station: bool = False
    #: Ensemble member sampling: None or 'per_event' (one member for all stations), 'per_station',
    #: or 'sector' (Poisson(sector_lambda) azimuthal sectors, one member each).
    member_sampling: Optional[str] = None
    sector_lambda: Optional[float] = None
    #: Cap on the open Instaseis database handles each worker keeps; None allows one per member.
    querier_cache_maxsize: Optional[int] = None

class IterativeLeastSquaresParameters(NamedTuple):

    max_iterations : int
    damping_factor : float
    dynamic_damping : bool = True
    mcmc_chain_for_mle : int = 0
    use_best_model : bool = True  # Use the lowest-chi^2 model at the end
    #: Stop once an accepted step lowers chi^2 by less than this fraction.
    chi2_tolerance : float = 1e-4

class DatasetGenerationParameters(NamedTuple):

    num_simulations : int
    #: Per-parameter sampler: a built-in sampler name, or a catalogue-prior closure built at parse time.
    sampling_method : dict
    use_fisher_to_constrain_bounds : int = 5
    iterative_least_squares : IterativeLeastSquaresParameters = IterativeLeastSquaresParameters(10, 0.01)


class ModelParameters:


    def __init__(self) -> None:

        self.theta_fiducial = {}
        self.stencil_deltas = {}
        self.bounds = {}
        self.nuisance = {}

        self.names = {}
        self.information = {}

        self.nuisance_effect_config: dict = {}

        # Nuisance key -> "simulation" or "training_augmentation"; not part of the theta vector,
        # so not in ``_parameter_names``.
        self.nuisance_stage: dict = {}

        self._parameter_names = [
            "theta_fiducial",
            "stencil_deltas",
            "bounds",
            "nuisance",
            "names",
            "information",
        ]

    @property
    def _parameters_register(self):
        """Always return up-to-date mapping of parameter types → current attribute values."""
        return {name: getattr(self, name) for name in self._parameter_names}

    def parameter_to_vector(self, parameter_type, only_theta_fiducial=False):
        """Flatten one register (e.g. ``'theta_fiducial'``) into a vector in parameter order."""

        if only_theta_fiducial:
            flattened_parameters = [item for param_name, sublist in self._parameters_register[parameter_type].items() for item in sublist 
                                        if param_name in self._parameters_register["theta_fiducial"].keys()]
        else:
            flattened_parameters = [item for sublist in self._parameters_register[parameter_type].values() for item in sublist]
        # np.concatenate converts namedtuple to np.array, so can't use it here
        if parameter_type != "information":
            try:
                flattened_parameters = np.array(flattened_parameters)
            except ValueError:
                # Parameters of different lengths (a 6-component tensor beside a 4-component location) form
                # a ragged object array.
                flattened_parameters = np.array(flattened_parameters, dtype=object)
        return flattened_parameters

    def get_parameter_values(self, param_name):
        """Return values for a specific parameter name, searching across dicts."""
        for container_name in ("theta_fiducial", "nuisance"):
            container = getattr(self, container_name)
            if param_name in container:
                return container[param_name]
        raise KeyError(f"Parameter '{param_name}' not found in theta_fiducial or nuisance")

    def vector_to_parameters(self, vector, parameter_type):
        i = 0
        copied_map = deepcopy(self._parameters_register[parameter_type])
        for param, parameter_value in copied_map.items():
            for j in range(len(parameter_value)):
                copied_map[param][j] = vector[i]
                i +=1
        return copied_map

    def vector_to_simulation_inputs(self, vector, only_theta_fiducial=False):
        i = 0
        inputs = {}
        copied_map = deepcopy(self._parameters_register['theta_fiducial'])
        for param, parameter_value in copied_map.items():
            inputs[param] = np.zeros(len(parameter_value))
            for j in range(len(parameter_value)):
                value = vector[i]
                inputs[param][j] = value
                i +=1
        if only_theta_fiducial:
            return inputs
        inputs.update(self.vector_to_nuisance_inputs(vector[i:]))
        return inputs

    def vector_to_nuisance_inputs(self, vector):
        """Read one value per nuisance from a vector holding ONLY the nuisance slots.

        Each nuisance consumes ``len(fiducial)`` slots in the order the configuration declares
        them; a scalar (0-D) fiducial and one whose fiducial is 2-D (a velocity model) each
        consume a single slot, the latter holding the object itself.
        """
        i = 0
        inputs = {}
        copied_map = deepcopy(self._parameters_register['nuisance'])
        for param, parameter_value in copied_map.items():
            parameter_value = np.asarray(parameter_value)
            if parameter_value.ndim == 0:
                inputs[param] = np.asarray(vector[i], dtype=parameter_value.dtype)
                i +=1
            elif parameter_value.ndim == 1:
                inputs[param] = np.zeros_like(parameter_value)
                for j in range(len(parameter_value)):
                    inputs[param][j] = vector[i]
                    i +=1
            else:
                inputs[param] = vector[i]
                i +=1
        return inputs

    def iterate_over_parameters(self):
        for param, parameter_value in self.theta_fiducial.items():
            for i in range(len(parameter_value)):
                yield self.theta_fiducial[param][i], self.stencil_deltas[param][i]
    
    def change_parameter(self, value, index):
        i = 0
        for param, parameter_value in self.theta_fiducial.items():
            for param_index in range(len(parameter_value)):
                if i == index:
                    self.theta_fiducial[param][param_index] = value
                    return
                i +=1
