"""A seed in ``inference.sbi`` makes the draws the SBI leg makes in worker processes reproducible."""
import numpy as np

from seismo_sbi.sbi import likelihood
from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.dataset_compressor import DatasetCompressor
from seismo_sbi.sbi.dataset_generator import DatasetGenerator
from seismo_sbi.utils.parallel import worker_seeds

DATA_VECTOR_LENGTH = 8


def standard_normal_log_probability(theta):
    return -0.5 * np.sum(theta ** 2)


def mcmc_chains(seed):
    return likelihood.generate_samples(standard_normal_log_probability, False, 2, 20, 2, burn_in=10,
                                       num_processes=2, move_size=0.1, mle_start=np.zeros(2), seed=seed)


class IdentityCovariance:
    inverse_metadata = None
    data_vector_length = DATA_VECTOR_LENGTH

    def create_matmul_inverse_covariance(self, metadata, length):
        return lambda vector: vector


class FirstTwoSamples:
    C = IdentityCovariance()

    def compress_data_vector(self, data_vector, matmul_callable):
        return data_vector[:2]


def compressed_dataset(seed, num_parallel_jobs):
    compressor = DatasetCompressor(data_loader=None, simulator=None, num_parallel_jobs=num_parallel_jobs)
    compressor.load_compressor_and_noise_model(FirstTwoSamples(),
                                               lambda: np.random.normal(size=DATA_VECTOR_LENGTH))
    compressor.load_sim = lambda sim_path, param_names: (np.array([float(sim_path)]),
                                                         np.zeros(DATA_VECTOR_LENGTH))
    return compressor.compress_dataset(["1", "2", "3", "4"], {}, seed=seed)


def test_seeded_mcmc_chains_reproduce_across_worker_processes():
    np.testing.assert_array_equal(mcmc_chains(seed=3), mcmc_chains(seed=3))
    assert not np.array_equal(mcmc_chains(seed=3), mcmc_chains(seed=4))


def test_seeded_training_noise_reproduces_across_worker_processes():
    parallel = compressed_dataset(seed=5, num_parallel_jobs=2)
    np.testing.assert_array_equal(parallel, compressed_dataset(seed=5, num_parallel_jobs=2))
    np.testing.assert_array_equal(parallel, compressed_dataset(seed=5, num_parallel_jobs=1))
    assert len(np.unique(parallel[:, 1])) == 4


def test_sbi_seed_is_read_from_the_inference_block():
    configuration = SBI_Configuration()
    sbi_block = {"method": "posterior", "noise_model": {"type": "gaussian_noises"}}
    configuration.parse_sbi_config({"sbi": dict(sbi_block, seed=7), "likelihood": {"run": False}})
    assert configuration.sbi_seed == 7
    configuration.parse_sbi_config({"sbi": sbi_block, "likelihood": {"run": False}})
    assert configuration.sbi_seed is None


def simulation_job_args(seed):
    generator = DatasetGenerator(simulator=None, output_base_path="unused", seed=seed)
    captured = []
    generator.run_parallel_simulations = captured.extend
    generator._create_sampler_generator_dict = lambda parameters, details, priors: {
        "moment_tensor": lambda args, num_samples: (np.zeros(6) for _ in range(num_samples))}
    generator._sampler_args = lambda parameters, samplers: {"moment_tensor": None}
    generator._create_sampler_transformer = lambda parameters: lambda sample: {"source_location": [0, 0, 1, 0]}
    generator.run_and_save_simulations(None, {}, 3, sample_namer=lambda n: (f"sim_{i}" for i in range(n)))
    return captured


def test_seeded_training_simulations_each_carry_their_own_member_seed():
    seeds = [inputs["seed"] for inputs, _ in simulation_job_args(seed=2)]
    assert seeds == worker_seeds(2, 3, "training members") and len(set(seeds)) == 3


def test_unseeded_training_simulations_carry_no_seed():
    assert all("seed" not in inputs for inputs, _ in simulation_job_args(seed=None))


def test_worker_seed_streams_are_unrelated():
    assert set(worker_seeds(1, 50, "training noise")).isdisjoint(worker_seeds(1, 50, "mcmc chains"))
