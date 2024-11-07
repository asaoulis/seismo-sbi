
import shutil
from pathlib import Path
from copy import deepcopy
import tempfile

from tqdm import tqdm
import numpy as np
from pprint import pprint

from ..compression.derivative_stencil import DerivativeStencil, ScoreCompressionData
from ...plotting.distributions import compute_scalar_moment
from ...utils.errors import error_handling_wrapper
from ..parameters import IterativeLeastSquaresParameters

class IterativeLeastSquaresSolver:

    def __init__(self, sim_parameters, model_parameters, dataloader, simulator, least_squares_configuration : IterativeLeastSquaresParameters, num_parallel_jobs = 1):
        self.sim_parameters = sim_parameters
        self.model_parameters = model_parameters

        self.data_loader = dataloader
        self.simulator = simulator
        self.least_squares_configuration = least_squares_configuration

        self.num_parallel_jobs = num_parallel_jobs

    @error_handling_wrapper(num_attempts=3)
    def solve_least_squares(self, observation, compressor, single_step = True):
        iterations = self.least_squares_configuration.max_iterations if not single_step else 1
        damping = self.least_squares_configuration.damping_factor if not single_step else 0

        misfit = np.inf
        new_parameters = deepcopy(self.model_parameters)
        model_params = new_parameters.parameter_to_vector('theta_fiducial', True)
        true_priors = deepcopy(compressor.prior_mean), deepcopy(compressor.prior_covariance)
        
        for _ in tqdm(range(iterations), "Performing iterative least squares for MLE fiducial", total=iterations):
            # Step 1: Compute gradients

            score_compression_data = self._compute_gradients(new_parameters)
            scaling_factors = self._create_scaling_vector(new_parameters)

            # Step 2: Compute the synthetic vector
            parameter_map = {**self.model_parameters.vector_to_parameters(model_params, 'theta_fiducial'),
                             **self.model_parameters.nuisance}
            synthetic = self.data_loader.convert_sim_data_to_array(
                    {"outputs": self.simulator.run_simulation(parameter_map)[1]}
                ).flatten()
            scaled_gradients = score_compression_data.data_parameter_gradients / scaling_factors[:, np.newaxis] 
            score_compression_data = score_compression_data._replace(data_parameter_gradients = scaled_gradients)
            score_compression_data = score_compression_data._replace(data_fiducial = synthetic)
            score_compression_data = score_compression_data._replace(theta_fiducial = model_params * scaling_factors)
            if true_priors[0] is not None:
                scaled_priors = (true_priors[0] * scaling_factors, true_priors[1] * scaling_factors**2)
                compressor.set_priors(scaled_priors)
            compressor.set_compression_variables(score_compression_data)

            misfit_new = compressor.compute_misfit(observation)
            if misfit_new > 0.99 * misfit:
                damping *= 1.2
            else:
                damping /= 1.5
            misfit = misfit_new
            print(f"chi^2: {misfit_new:.5f}, damping lambda: {damping:.3f}", flush=True)
            
            # Step 4: Compute the update step using the Gauss-Newton method
            theta_MLE = compressor.compute_theta_MLE(observation, damping=damping)
            
            # Step 5: Update the model parameters
            model_params = theta_MLE / scaling_factors
            theta_MLE_map =  {**new_parameters.vector_to_parameters(model_params, 'theta_fiducial')}
            update = 'New MLE:\n' + "\n".join(
                f"{k}: [{', '.join(f'{v_i:.3e}' if abs(v_i) < 1e-3 or abs(v_i) >= 1e3 else f'{v_i:.3f}' for v_i in v)}]" 
                for k, v in theta_MLE_map.items()
            )
            print(update, flush=True)
            new_parameters.theta_fiducial = new_parameters.vector_to_parameters(model_params, 'theta_fiducial')

        final_score_compression_data = self._compute_gradients(new_parameters)
        theta_MLE = model_params
        theta_MLE_map =  {**new_parameters.vector_to_parameters(model_params, 'theta_fiducial'),
                          **self.model_parameters.nuisance}
        D_MLE = self.data_loader.convert_sim_data_to_array(
                    {"outputs": self.simulator.run_simulation(theta_MLE_map)[1]}
                ).flatten()
        final_score_compression_data = final_score_compression_data._replace(data_fiducial = D_MLE)
        final_score_compression_data = final_score_compression_data._replace(theta_fiducial = theta_MLE)

        return final_score_compression_data

    @error_handling_wrapper(num_attempts=3)
    def _compute_gradients(self, model_parameters):

        with tempfile.TemporaryDirectory() as stencil_output_folder:
            stencil_output_folder_path = Path(stencil_output_folder)
            
            derivative_stencil = DerivativeStencil(model_parameters, stencil_output_folder_path)
            
            score_compression_data = derivative_stencil.calculate_score_compression_data(
                                        self.simulator.execute_sim_and_save_outputs,
                                        self.data_loader.load_flattened_simulation_vector,
                                        self.num_parallel_jobs)


        return score_compression_data #score_compression_data.data_parameter_gradients, score_compression_data.second_order_gradients
    

    def _create_scaling_vector(self, model_parameters):
        scaling_factors = []
        for param, value in model_parameters.theta_fiducial.items():
            if param == 'moment_tensor':
                moment_tensor_components = value
                _, M_0 = compute_scalar_moment(moment_tensor_components)
                scaling_factors.append( 1/ M_0 * np.ones(6))
            elif param == 'source_location':
                scaling_factors.append(np.array([0.1, 0.1, 0.5, 1]))
        
        return np.hstack(scaling_factors)