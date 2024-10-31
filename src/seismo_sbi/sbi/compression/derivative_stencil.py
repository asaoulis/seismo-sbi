import numpy as np
import joblib
from copy import deepcopy

from pathlib import Path
from itertools import product

from .gaussian import ScoreCompressionData
from ..configuration import ModelParameters


class DerivativeStencil:

    five_point_stencil_offsets = np.array([-2, -1, 1, 2])

    def __init__(self, parameters : ModelParameters, output_folder: Path):

        self.num_params = len(list(parameters.iterate_over_parameters()))
        self.parameters = parameters
        self.output_folder = output_folder

        self.simulation_output_paths = np.empty((self.num_params, 4), dtype=object)

        for i in range(self.num_params):
            for j in range(4):
                sim_name = str(
                    (output_folder / f"stencil_sim_{i}{j}.h5").resolve())
                self.simulation_output_paths[i, j] = sim_name

    def calculate_score_compression_data(self, simulator, data_loader, num_jobs=1) -> ScoreCompressionData:

        self.run_stencil_simulations(simulator, num_jobs)

        stencil_results = self.load_simulation_results(data_loader)
        data_fiducial = data_loader(str((self.output_folder / "theta_fiducial.h5").resolve()))

        gradients = self.compute_gradients_from_stencil(stencil_results)
        second_order_gradients = self.compute_second_order_gradients_from_stencil(stencil_results, data_fiducial)

        return ScoreCompressionData(self.parameters.parameter_to_vector('theta_fiducial'), data_fiducial, gradients, second_order_gradients)

    def run_stencil_simulations(self, simulator, num_jobs=1):

        five_point_stencil_offsets = np.array([-2, -1, 1, 2])
        simulation_job_args_list = []
        base_parameters = deepcopy(self.parameters)

        simulation_job_args_list.append(({**base_parameters.theta_fiducial, **self.parameters.nuisance}, 
            str((self.output_folder / "theta_fiducial.h5").resolve())))

        for i, (param_value, delta) in enumerate(self.parameters.iterate_over_parameters()):
            for j, offset in enumerate(five_point_stencil_offsets):

                new_parameters = deepcopy(self.parameters)
                new_parameters.change_parameter(param_value + offset * delta, i)

                sim_name = self.simulation_output_paths[i, j]
                simulation_job_args_list.append(({**new_parameters.theta_fiducial, **self.parameters.nuisance}, str(sim_name)))

        self.run_parallel_simulations(
            simulator, simulation_job_args_list, num_jobs)

    def run_parallel_simulations(self, simulator, simulation_job_args_list, num_parallel_jobs):

        if num_parallel_jobs > 1:
            with joblib.parallel_backend('loky', n_jobs=num_parallel_jobs):
                joblib.Parallel()(
                    joblib.delayed(simulator)(*simulation_job_args) for simulation_job_args in simulation_job_args_list
                )
        else: 
            for simulation_job_args in simulation_job_args_list:
                simulator(*simulation_job_args)

    def load_simulation_results(self, simulation_D_loader, stencil_simulations=None):

        if stencil_simulations is None:
            stencil_simulations = self.simulation_output_paths

        D_loader_array = np.frompyfunc(simulation_D_loader, 1, 1)

        stencil_results = D_loader_array(stencil_simulations)

        return stencil_results

    def compute_gradients_from_stencil(self, stencil_results):

        dD_dtheta_gradients = np.empty(
            (self.num_params, stencil_results[0, 0].shape[0]))
        delta_array = self.parameters.parameter_to_vector("stencil_deltas")
        for i in range(self.num_params):
            dD_dtheta_gradients[i, :] = (-stencil_results[i, 3] + 8 * stencil_results[i, 2]
                                         - 8 * stencil_results[i, 1] + stencil_results[i, 0]) \
                / (12*delta_array[i])

        return dD_dtheta_gradients


    def compute_second_order_gradients_from_stencil(self, stencil_results, D_central):

        dD_dtheta_2_gradients = np.empty(
            (self.num_params, stencil_results[0, 0].shape[0]))
        delta_array = self.parameters.parameter_to_vector("stencil_deltas")
        for i in range(self.num_params):
            dD_dtheta_2_gradients[i, :] = (-stencil_results[i, 3] + 16 * stencil_results[i, 2] - 30 * D_central \
                                         + 16 * stencil_results[i, 1] - stencil_results[i, 0]) \
                / (12*delta_array[i]**2)

        return dD_dtheta_2_gradients


class HessianDerivativeStencil:


    def __init__(self, parameters : ModelParameters, output_folder: Path):

        self.num_params = len(list(parameters.iterate_over_parameters()))
        self.parameters = parameters
        self.output_folder = output_folder

        self.simulation_output_paths = np.empty((self.num_params, self.num_params, 4), dtype=object)

        for i in range(self.num_params):
            for j in range(self.num_params):
                if i > j:
                    for k in range(4):
                        sim_name = str(
                            (output_folder / f"stencil_sim_{i}{j}{k}.h5").resolve())
                        self.simulation_output_paths[i, j, k] = sim_name

    def calculate_score_compression_data(self, simulator, data_loader, num_jobs=1) -> ScoreCompressionData:

        self.run_stencil_simulations(simulator, num_jobs)

        stencil_results = self.load_simulation_results(data_loader)
        data_fiducial = data_loader(str((self.output_folder / "theta_fiducial.h5").resolve()))

        gradients = self.compute_gradients_from_stencil(stencil_results)

        return ScoreCompressionData(self.parameters.parameter_to_vector('theta_fiducial'), data_fiducial, gradients, np.array([]))

    def run_stencil_simulations(self, simulator, num_jobs=1):

        offset_pairs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        simulation_job_args_list = []
        base_parameters = deepcopy(self.parameters)

        simulation_job_args_list.append(({**base_parameters.theta_fiducial, **self.parameters.nuisance}, 
            str((self.output_folder / "theta_fiducial.h5").resolve())))

        for i, (param_value_i, delta_i) in enumerate(self.parameters.iterate_over_parameters()):
            for j, (param_value_j, delta_j) in enumerate(self.parameters.iterate_over_parameters()):
                if i > j:
                    for k, (offset_i, offset_j) in enumerate(offset_pairs):

                        new_parameters = deepcopy(self.parameters)
                        new_parameters.change_parameter(param_value_i + offset_i * delta_i, i)
                        new_parameters.change_parameter(param_value_j + offset_j * delta_j, j)

                        sim_name = self.simulation_output_paths[i, j, k]

                        simulation_job_args_list.append(({**new_parameters.theta_fiducial, **self.parameters.nuisance}, str(sim_name)))

        self.run_parallel_simulations(
            simulator, simulation_job_args_list, num_jobs)

    def run_parallel_simulations(self, simulator, simulation_job_args_list, num_parallel_jobs):

        if num_parallel_jobs > 1:
            with joblib.parallel_backend('loky', n_jobs=num_parallel_jobs):
                joblib.Parallel()(
                    joblib.delayed(simulator)(*simulation_job_args) for simulation_job_args in simulation_job_args_list
                )
        else: 
            for simulation_job_args in simulation_job_args_list:
                simulator(*simulation_job_args)

    def load_simulation_results(self, simulation_D_loader, downsampled_length=None, stencil_simulations=None):

        if stencil_simulations is None:
            stencil_simulations = self.simulation_output_paths

        D_loader_array = np.frompyfunc(simulation_D_loader, 2, 1)

        stencil_results = D_loader_array(
            stencil_simulations, downsampled_length)

        return stencil_results

    def compute_gradients_from_stencil(self, stencil_results):

        dD_dtheta_gradients = np.empty(
            (self.num_params, self.num_params, stencil_results[1,0,0].shape[0]))
        delta_array = self.parameters.parameter_to_vector("stencil_deltas")
        for i in range(self.num_params):
            for j in range(self.num_params):
                if i > j:
                    dD_dtheta_gradients[i, j, :] = (stencil_results[i, j, 3] - stencil_results[i, j, 2]
                                                - stencil_results[i, j, 1] + stencil_results[i, j, 0]) \
                        / (4*delta_array[i]*delta_array[j])

        return dD_dtheta_gradients
