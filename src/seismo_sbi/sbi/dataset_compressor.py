import numpy as np
import joblib
import os

from .compression.derivative_stencil import DerivativeStencil, HessianDerivativeStencil
from .compression.gaussian import Compressor, ScoreCompressionData
from seismo_sbi.instaseis_simulator.dataset_generator import tqdm_joblib
from tqdm import tqdm


from ..instaseis_simulator.dataloader import SimulationDataLoader


class DatasetCompressor:

    def __init__(self, data_loader : SimulationDataLoader, simulator,
                        num_parallel_jobs = 1, downsampled_length = None):

        self.data_loader = data_loader
        self.simulator = simulator
        self.num_parallel_jobs = num_parallel_jobs
        self.downsampled_length = downsampled_length

        self.compressor = None
        self.synthetic_noise_model_sampler = None

    def load_compressor_and_noise_model(self, compressor : Compressor, synthetic_noise_model_sampler = None):

        self.compressor = compressor
        if synthetic_noise_model_sampler is None:
            synthetic_noise_model_sampler = self.compressor.create_covariance_matrix_sampler()

        self.synthetic_noise_model_sampler = synthetic_noise_model_sampler


    def run_derivative_stencil_for_compression_data(self, parameters,
                                                    stencil_output_folder, score_compression_data_path):

        derivative_stencil = DerivativeStencil(parameters, stencil_output_folder)
        
        score_compression_data = derivative_stencil.calculate_score_compression_data(
                                    self.simulator,
                                    self.data_loader.load_flattened_simulation_vector,
                                    self.num_parallel_jobs)

        np.save(score_compression_data_path, score_compression_data._asdict())

        return score_compression_data

    def run_hessian_stencil(self, parameters, score_compression_data : ScoreCompressionData, stencil_output_folder):

        hessian_derivative_stencil = HessianDerivativeStencil(parameters, stencil_output_folder)
        hessian_derivative_stencil.run_stencil_simulations(self.simulator, self.num_parallel_jobs)

        nonetype_safe_loader = lambda x, dummy: self.data_loader.load_flattened_simulation_vector(x) if x is not None else 0
        hessian_stencil_results = hessian_derivative_stencil.load_simulation_results(nonetype_safe_loader)

        first_order_gradients = score_compression_data.data_parameter_gradients
        diagonal_2nd_order = score_compression_data.second_order_gradients

        num_dim = first_order_gradients.shape[0]
        expanded_diagonal = np.zeros((num_dim, num_dim, first_order_gradients.shape[1]))
        for i in range(num_dim):
            expanded_diagonal[i,i] = diagonal_2nd_order[i]

        hessian_gradients = hessian_derivative_stencil.compute_gradients_from_stencil(hessian_stencil_results)
        hessian_gradients = hessian_gradients + hessian_gradients.transpose(1, 0, 2) + expanded_diagonal

        return hessian_gradients
    
    def compress_dataset(self, simulation_data_paths, param_names):
        cov = self.compressor.C
        matmul_callable = cov.create_matmul_inverse_covariance(cov.C_inverse, cov.data_vector_length)
        if self.num_parallel_jobs not in [0,1]:
            with tqdm_joblib(tqdm(desc="Compressing dataset: ", total=len(simulation_data_paths))) as progress_bar:

                with joblib.parallel_backend('loky', n_jobs=self.num_parallel_jobs):
                    results = joblib.Parallel()(
                        joblib.delayed(self._load_and_compress_sim)(sim_path, param_names, matmul_callable)
                                for sim_path in simulation_data_paths
                    )
        else:
            results = []
            for sim_path in simulation_data_paths:
                    results.append(self._load_and_compress_sim(sim_path, param_names))

        return np.stack(results)

    def _load_and_compress_sim(self, sim_path, param_names, matmul_callable):
        inputs, D = self.load_sim(sim_path, param_names)
        noise = self.synthetic_noise_model_sampler()
        if isinstance(noise, tuple):
            noise, _ = noise
        compressed_representation = self.compressor.compress_data_vector(D + noise, matmul_callable=matmul_callable)
        return np.concatenate([inputs, compressed_representation])

    def load_sim(self, sim_path, parameter_name_map):
        if len(parameter_name_map) > 0:
            inputs = self.data_loader.load_input_data(sim_path)
            fixed_keys = dict((param_type, param_names) if param_names != ["earthquake_magnitude"] else ("moment_tensor",["earthquake_magnitude"])\
                          for param_type, param_names in parameter_name_map.items())
            inputs = np.concatenate([[inputs[param_type][param_name] for param_name in param_names] for param_type, param_names in fixed_keys.items()])
        else:
            inputs = np.array([])
        D = self.data_loader.load_flattened_simulation_vector(sim_path)
        return inputs,D