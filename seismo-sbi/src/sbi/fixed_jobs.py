import numpy as np
from copy import deepcopy

from src.sbi.parameters import ModelParameters 

class FixedEventJobs:

    mechanism_moment_tensors = {
        "normal": np.array([1, -1, 0, 0, 0, 0]),
        "strike_slip": np.array([0, -1, 1, 0, 0, 0]),
        "thrust": np.array([0, 1, -1, 0, 0, 0]),
        "pure_isotropic": np.array([1, 1, 1, 0, 0, 0]),
        "pure_DC": np.array([1, 0, -1, 0, 0, 0]),
        "pure_CLVD": np.array([1, 1, -2, 0, 0, 0]),
        "rotated_strike": np.array([0, 0, 0, 1, 0, 0]),
    }

    def __init__(self, parameters : ModelParameters, M_0, output_path):

        self.parameters = parameters
        self.M_0 = M_0
        self.mag_string = f"{(np.log10(M_0) - 9.1)/1.5:.2f}"
        self.output_path = output_path


    def create_simulation_inputs(self, jobs):
        simulation_inputs = []
        for job in jobs:
            simulation_path = f"{self.output_path}/{job}_{self.mag_string}M.h5"
            mechanism_matrix = self.mechanism_moment_tensors[job]
            matrix_norm = (1/np.sqrt(2)) * np.sum(mechanism_matrix**2)**(1/2)
            moment_tensor = self.M_0 * mechanism_matrix/matrix_norm
            new_parameters = deepcopy(self.parameters)
            new_parameters.theta_fiducial['moment_tensor'] = moment_tensor
            inputs = {**new_parameters.theta_fiducial, **self.parameters.nuisance}
            simulation_inputs.append((inputs, simulation_path))
        
        return simulation_inputs