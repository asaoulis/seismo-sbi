""" This module contains the parameters for the sbi - pipeline.
"""

import numpy as np
from typing import NamedTuple, List
from copy import deepcopy

from seismo_sbi.instaseis_simulator.receivers import Receivers

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

class IterativeLeastSquaresParameters(NamedTuple):

    max_iterations : int
    damping_factor : float

class DatasetGenerationParameters(NamedTuple):

    num_simulations : int
    sampling_method : str
    use_fisher_to_constrain_bounds : int = 5
    iterative_least_squares : IterativeLeastSquaresParameters = IterativeLeastSquaresParameters(10, 0.01)

import hashlib

class ModelParameters:


    def __init__(self) -> None:

        self.theta_fiducial = {}
        self.stencil_deltas = {}
        self.bounds = {}
        self.nuisance = {}

        self.names = {}
        self.information = {}

        self._parameters_register = {"theta_fiducial": self.theta_fiducial,
                                    "stencil_deltas": self.stencil_deltas,
                                    "bounds": self.bounds,
                                    "nuisance": self.nuisance,
                                    "names": self.names,
                                    "information": self.information}

    def parameter_to_vector(self, parameter_type, only_theta_fiducial=False):

        if only_theta_fiducial:
            flattened_parameters = [item for param_name, sublist in self._parameters_register[parameter_type].items() for item in sublist 
                                        if param_name in self._parameters_register["theta_fiducial"].keys()]
        else:
            flattened_parameters = [item for sublist in self._parameters_register[parameter_type].values() for item in sublist]
        # np.concatenate converts namedtuple to np.array, so can't use it here
        if parameter_type != "information":
            flattened_parameters = np.array(flattened_parameters)
        return flattened_parameters

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
                inputs[param][j] = vector[i]
                i +=1
        if only_theta_fiducial:
            return inputs
        copied_map = deepcopy(self._parameters_register['nuisance'])
        for param, parameter_value in copied_map.items():
            inputs[param] = np.zeros(len(parameter_value))
            for j in range(len(parameter_value)):
                inputs[param][j] = vector[i]
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


    def theta_fiducial_hash(self):
        # Serialize the contents of self.theta_fiducial as a string
        serialized_theta = str(self.theta_fiducial)

        # Hash the serialized string using md5 algorithm
        hash_object = hashlib.md5(serialized_theta.encode())

        # Get the hexadecimal representation of the hash digest
        hash_string = hash_object.hexdigest()

        # Take the first 8 characters of the hash string
        shortened_hash = hash_string[:8]

        return shortened_hash