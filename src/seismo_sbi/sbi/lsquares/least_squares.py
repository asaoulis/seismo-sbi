from pathlib import Path
from copy import deepcopy
import tempfile

# Optional progress wrapper: use tqdm if available, else no-op
try:
    from tqdm import tqdm as _tqdm
    def iter_progress(iterable, desc=None, total=None):
        return _tqdm(iterable, desc=desc, total=total)
except Exception:
    def iter_progress(iterable, desc=None, total=None):
        return iterable

import numpy as np

from ..compression.derivative_stencil import DerivativeStencil
from ...plotting.distributions import compute_scalar_moment
from ...utils.errors import error_handling_wrapper
from ..types.parameters import IterativeLeastSquaresParameters

class IterativeLeastSquaresSolver:

    def __init__(self, sim_parameters, compression_methods, model_parameters, data_manager, simulator_wrapper, least_squares_configuration : IterativeLeastSquaresParameters, num_parallel_jobs = 1):
        self.sim_parameters = sim_parameters
        self.model_parameters = model_parameters
        self.compression_methods = compression_methods

        self.data_manager = data_manager
        self.data_loader = data_manager.data_loader
        self.stencil_args = (compression_methods, simulator_wrapper, self.sim_parameters)
        self.simulator = simulator_wrapper.simulator
        self.least_squares_configuration = least_squares_configuration

        self.num_parallel_jobs = num_parallel_jobs

    @error_handling_wrapper(num_attempts=3)
    def solve_least_squares(self, observation, compressor, single_step = True, return_history = False):
        iterations = self.least_squares_configuration.max_iterations if not single_step else 2
        damping = self.least_squares_configuration.damping_factor if not single_step else 0
        adaptive = self.least_squares_configuration.dynamic_damping

        misfit = np.inf
        new_parameters = deepcopy(self.model_parameters)
        model_params = new_parameters.parameter_to_vector('theta_fiducial', True)
        true_priors = deepcopy(compressor.prior_mean), deepcopy(compressor.prior_covariance)

        # Track the best (lowest chi^2) model encountered during iterations
        best_chi2 = np.inf
        best_params_vec = None
        best_iter = -1
        all_steps = [model_params]
        
        for it in iter_progress(range(iterations), "Performing iterative least squares for MLE fiducial", total=iterations):
            # Step 1: Compute gradients

            score_compression_data, extra_gradients = self.data_manager.compute_required_compression_data(new_parameters, *self.stencil_args)
            scaling_factors = self._create_scaling_vector(new_parameters)
            scaling_factors = np.ones_like(scaling_factors)

            if true_priors[0] is not None:
                scaled_priors = (true_priors[0] * scaling_factors, true_priors[1] * scaling_factors**2)
                compressor.set_priors(scaled_priors)
            if extra_gradients is not None:
                compressor.C.set_covariance(extra_gradients)
            compressor.set_compression_variables(score_compression_data)
            
            misfit_new = compressor.compute_misfit(observation)
            # Keep track of best model before applying update
            if misfit_new < best_chi2:
                best_chi2 = misfit_new
                best_params_vec = np.copy(model_params)
                best_iter = it
            
            if adaptive and not single_step:
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
            all_steps.append(deepcopy(model_params))

        # Optionally use the best (lowest chi^2) model found during the iterations
        if self.least_squares_configuration.use_best_model and best_params_vec is not None:
            new_parameters.theta_fiducial = new_parameters.vector_to_parameters(best_params_vec, 'theta_fiducial')
            print(f"Using best chi^2 model from iteration {best_iter}: chi^2={best_chi2:.5f}", flush=True)
        final_score_compression_data, extra_gradients = self.data_manager.compute_required_compression_data(new_parameters, *self.stencil_args)
        if not return_history:
            return final_score_compression_data, extra_gradients
        else:
            return final_score_compression_data, extra_gradients, all_steps

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