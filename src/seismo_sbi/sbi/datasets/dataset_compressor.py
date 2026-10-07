"""Compress a folder of simulations into a training set.

:class:`DatasetCompressor` adds a noise draw to each simulation, compresses it with the loaded
compressor, and runs the stencil simulations the score compression needs.
"""

import numpy as np
import joblib

from ..compression.derivative_stencil import DerivativeStencil, HessianDerivativeStencil
from ..compression.gaussian import Compressor, ScoreCompressionData
from seismo_sbi.utils.parallel import tqdm_joblib, worker_seeds
from tqdm import tqdm

from seismo_sbi.simulators.simulation_io import SimulationDataLoader

class DatasetCompressor:

    def __init__(self, data_loader : SimulationDataLoader, simulator,
                        num_parallel_jobs = 1, downsampled_length = None):

        self.data_loader = data_loader
        self.simulator = simulator
        self.num_parallel_jobs = num_parallel_jobs
        self.downsampled_length = downsampled_length

        self.compressor = None
        self.synthetic_noise_model_sampler = None

    def load_compressor_and_noise_model(self, compressor : Compressor, synthetic_noise_model_sampler):

        self.compressor = compressor
        self.synthetic_noise_model_sampler = synthetic_noise_model_sampler


    def run_derivative_stencil_for_compression_data(self, parameters,
                                                    stencil_output_folder, use_fiducial=True, simulator=None, **kwargs):
        """The score-compression data from a derivative stencil run with ``simulator`` (a callable
        that simulates and saves), or with this compressor's own simulator when None."""
        derivative_stencil = DerivativeStencil(parameters, stencil_output_folder, use_fiducial=use_fiducial)
        
        score_compression_data = derivative_stencil.calculate_score_compression_data(
                                    self.simulator if simulator is None else simulator,
                                    self.data_loader.load_simulation_data_array,
                                    self.num_parallel_jobs,
                                    **kwargs)

        return score_compression_data

    def run_hessian_stencil(self, parameters, score_compression_data : ScoreCompressionData, stencil_output_folder):

        hessian_derivative_stencil = HessianDerivativeStencil(parameters, stencil_output_folder)
        hessian_derivative_stencil.run_stencil_simulations(self.simulator, self.num_parallel_jobs)

        nonetype_safe_loader = lambda x, dummy: self.data_loader.load_simulation_data_array(x) if x is not None else 0
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
    
    def compress_dataset(self, simulation_data_paths, param_names, seed=None):
        """One row ``[theta, compressed(D + noise)]`` per simulation; a ``seed`` gives each
        simulation its own reproducible noise draw, in any worker.
        """
        sim_seeds = worker_seeds(seed, len(simulation_data_paths), "training noise")
        cov = self.compressor.C
        matmul_callable = cov.create_matmul_inverse_covariance(cov.inverse_metadata, cov.data_vector_length)
        if self.num_parallel_jobs not in [0,1]:
            try:
                with tqdm_joblib(tqdm(desc="Compressing dataset: ", total=len(simulation_data_paths))):

                    with joblib.parallel_backend('loky', n_jobs=self.num_parallel_jobs):
                        results = joblib.Parallel()(
                            joblib.delayed(self._load_and_compress_sim)(sim_path, param_names, matmul_callable, sim_seed)
                                    for sim_path, sim_seed in zip(simulation_data_paths, sim_seeds)
                        )
            except Exception as e:
                print("Error during parallel compression:", e)
                raise e
            finally:
                from joblib.externals.loky import get_reusable_executor
                # reuse=True kills the pool Parallel used; with default arguments loky would first
                # restart that pool gracefully, which can hang on a worker that never exits.
                get_reusable_executor(reuse=True).shutdown(wait=True, kill_workers=True)
            
        else:
            results = []
            for sim_path, sim_seed in zip(simulation_data_paths, sim_seeds):
                    results.append(self._load_and_compress_sim(sim_path, param_names, matmul_callable, sim_seed))

        return np.stack(results)

    def _load_and_compress_sim(self, sim_path, param_names, matmul_callable, sim_seed=None):
        inputs, D = self.load_sim(sim_path, param_names)
        if sim_seed is not None:
            np.random.seed(sim_seed)
        noise = self.synthetic_noise_model_sampler.draw().noise
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
        D = self.data_loader.load_simulation_data_array(sim_path)
        return inputs,D