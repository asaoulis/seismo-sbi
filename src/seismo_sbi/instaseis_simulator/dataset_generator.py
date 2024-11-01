from typing import List
from abc import ABC, abstractmethod
import joblib
import traceback

from functools import partial

import numpy as np

from seismo_sbi.sbi.configuration import InvalidConfiguration, ModelParameters
from seismo_sbi.sbi.configuration import SBI_Configuration

import contextlib
from tqdm import tqdm

# Monkey-patch of joblib to report into tqdm progress bar,
# solution taken from https://stackoverflow.com/a/61689175
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

class ParallelSimulationRunner(ABC):

    def __init__(self, simulator, num_parallel_jobs):

        self.simulator = self._error_handling_wrapper(simulator)
        self.num_parallel_jobs = num_parallel_jobs

    def _error_handling_wrapper(self, simulation_callable, num_attempts = 3):

        def _error_handled_simulation_callable(*args, **kwargs):

            for attempt_number in range(num_attempts):
                try:
                    simulation_callable(*args, **kwargs)
                    return None # tidy this up
                except Exception as exc:
                    # Error handling for remote instaseis simulations
                    # to prevent hanging on single connection failure
                    print(f"Simulation terminated with exception {attempt_number + 1} times:")
                    print(traceback.format_exception())
                    print("Retrying simulation...")

            print("Simulations failed. Exiting.")
            raise exc
            
        
        return _error_handled_simulation_callable
    
    @abstractmethod
    def run_and_save_simulations(self, input_generator, num_parallel_jobs=1):
        pass

    def run_parallel_simulations(self, simulation_job_args_list):

        if self.num_parallel_jobs not in [0, 1]:
            with tqdm_joblib(tqdm(desc="Running simulations: ", total=len(simulation_job_args_list))) as progress_bar:
                with joblib.parallel_backend('loky', n_jobs=self.num_parallel_jobs):
                    joblib.Parallel()(
                        joblib.delayed(self.simulator)(*simulation_job_args) for
                            simulation_job_args in simulation_job_args_list
                    )
        else:
            for simulation_job_args in simulation_job_args_list:
                self.simulator(*simulation_job_args)
        
    def _create_sampler_transformer(self, parameters : ModelParameters):
        return lambda sampled_vector: parameters.vector_to_simulation_inputs(sampled_vector)


from scipy.stats.qmc import LatinHypercube

## TODO: convert all of these into torch.Distributions,
## so that they can be used as priors in sbi

def constant_sampler(value, num_samples):
    for _ in range(num_samples):
        yield value

def latin_hypercube_sampler(bounds, num_samples):
    sampling_engine = LatinHypercube(len(bounds[0]))
    samples = sampling_engine.random(n = num_samples)
    delta = bounds[1] - bounds[0]
    for sample in samples:
        transformed_values = bounds[0] + np.multiply(sample, delta)
        yield transformed_values

def uniform_sampler(bounds, num_samples):
    for _ in range(num_samples):
        yield np.random.uniform(bounds[0], bounds[1])

def gaussian_sampler(bounds, num_samples):
    for _ in range(num_samples):
        yield np.random.multivariate_normal(bounds[0], np.diag(bounds[1]))

def transform_sampling_func(sampling_func, transform_func):
    def wrapper(*args, **kwargs):
        for value in sampling_func(*args, **kwargs):
            value = [np.array([v]) if np.isscalar(v) else v for v in value]
            yield transform_func(np.concatenate(value))
    return wrapper

class MomentTensorLogScaleHomogeneous:

    @staticmethod
    def _generate_sample(bounds):

        # following `Fully probabilistic seismic source inversion – Part 1: Efficient parameterisation.`
        # Stahler-Sigloch (2014), 2.2: Parametrisation of the moment tensor
        x = [np.random.uniform(0, 1) for _ in range(5)]
        Y3 = 1
        Y2 = np.sqrt(x[1])
        Y1 = Y2*x[0]

        # log prior on M_0
        M0 = np.exp(np.random.uniform(np.log(bounds[0]), np.log(bounds[1])))

        M_xx = np.sqrt(Y1) * np.cos(2*np.pi*x[2]) * np.sqrt(2) * M0
        M_yy = np.sqrt(Y1) * np.sin(2*np.pi*x[2]) * np.sqrt(2) * M0
        M_zz = np.sqrt(Y2 - Y1) * np.cos(2*np.pi*x[3]) * np.sqrt(2) * M0
        M_xy = np.sqrt(Y2 - Y1) * np.sin(2*np.pi*x[3]) * M0
        M_yz = np.sqrt(Y3 - Y2) * np.cos(2*np.pi*x[4]) * M0
        M_xz = np.sqrt(Y3 - Y2) * np.sin(2*np.pi*x[4]) * M0

        # Now convert to (r, t, p) coordinates - see
        # Aki & Richards (2002) p. 113
        M = [M_zz, M_xx, M_yy, M_xz, -M_yz, -M_xy]

        return np.array(M)

    @staticmethod
    def sampler(bounds, num_samples):
        for _ in range(num_samples):
            yield MomentTensorLogScaleHomogeneous._generate_sample(bounds)

from itertools import chain

class RejectionSamplingWrapper:

    def __init__(self, sampler, prior_bounds, num_samples):
        self.sampler = sampler
        self.sample_in_prior = self.create_prior(prior_bounds)
        self.num_samples = num_samples

    def create_prior(self, bounds):
        def prior(sample):
            for key in bounds.keys():
                if key == 'source_location':
                    if not np.all(np.logical_and(sample[key] > bounds[key][0], sample[key] < bounds[key][1])):
                        return False
            return True
        return prior
    
    def __iter__(self):
        counter = 0
        #tqdm pbar
        pbar = tqdm(total=self.num_samples, desc="Rejection Sampling prior")
        while counter < self.num_samples:
            sample_generator = self.sampler(100)
            for sample in sample_generator:
                if self.sample_in_prior(sample):
                    print(sample)
                    yield sample
                    counter+=1
                    pbar.update(1)

                    break

from scipy.stats import truncnorm
class TruncatedGaussianSampler:

    def __init__(self, mean, cov, lower, upper):
        std = np.sqrt(cov)
        self.truncated_normal = truncnorm(
            (lower - mean) / std,
            (upper - mean) / std,
            loc=mean,
            scale=std
        )
        self.num_dims = len(mean)

    def sampler(self, num_samples):
        samples= self.truncated_normal.rvs(size=(num_samples, self.num_dims))
        for sample in samples:
            yield sample

def truncated_gaussian_sampler(bounds, num_samples):
    sampler = TruncatedGaussianSampler(*bounds)
    for sample in sampler.sampler(num_samples):
        yield sample


class DatasetGenerator(ParallelSimulationRunner):

    sampler_lookup_map = {"latin hypercube" : latin_hypercube_sampler,
                          "uniform"         : uniform_sampler,
                          "uniform known"         : uniform_sampler,
                          "gaussian"        : gaussian_sampler,                     
                          "moment tensor log prior"   : MomentTensorLogScaleHomogeneous.sampler,
                          "constant": constant_sampler,
                          "truncated gaussian": truncated_gaussian_sampler}

    def __init__(self, simulator, output_base_path, num_parallel_jobs=1):
        super().__init__(simulator, num_parallel_jobs)

        self.output_base_path = output_base_path
        self.num_parallel_jobs = num_parallel_jobs


    def run_and_save_simulations(self, parameters : ModelParameters, sampler_details, indices, sample_namer = None, priors= (None, None)):
        if sample_namer is None:
            sample_namer = self._sample_namer
        try:
            num_samples = indices[1] - indices[0]
        except TypeError:
            num_samples = indices

        samplers = self._create_sampler_generator_dict(parameters, sampler_details, priors=priors)
        if priors[0] is None:
            sampler_args = parameters.bounds
        else:
            sampler_args = {key : (parameters.vector_to_simulation_inputs(priors[0])[key], 
                                   parameters.vector_to_simulation_inputs(priors[1])[key],
                                   parameters.bounds[key][0],
                                   parameters.bounds[key][1]) 
                                   for key in parameters.names.keys()}

        sampler_callable = lambda num_samples: zip(*[sampler(sampler_args[key], num_samples) for key, sampler in samplers.items()])
        
        sampler_transformer = transform_sampling_func(sampler_callable, self._create_sampler_transformer(parameters))

        input_generator = zip(sampler_transformer(num_samples),
                                sample_namer(indices))


        simulation_job_args_list = [input_config for input_config in input_generator]

        self.run_parallel_simulations(simulation_job_args_list)
    
    def run_predefined_batch(self, thetas, indices, parameters : ModelParameters):

        simulation_parameters = [parameters.vector_to_simulation_inputs(theta) for theta in thetas]
        input_generator = zip(simulation_parameters, self._sample_namer(indices))
        simulation_job_args_list = [input_config for input_config in input_generator]
        self.run_parallel_simulations(simulation_job_args_list)
    
    @staticmethod
    def _create_sampler_generator_dict(parameters : ModelParameters, sampler_details, priors = (None, None)):
        if priors[0] is None:
            samplers = {key : DatasetGenerator.sampler_lookup_map[sampler_details[key]] for key in chain(parameters.names.keys(), parameters.nuisance.keys())}
        else:
            samplers = {key : DatasetGenerator.sampler_lookup_map['truncated gaussian'] for key in chain(parameters.names.keys(), parameters.nuisance.keys())}

        return samplers

    @staticmethod
    def create_samplers(parameters : ModelParameters, sampler_details, priors = (None, None)):
        sampler_generators = DatasetGenerator._create_sampler_generator_dict(parameters, sampler_details, priors)
        # if priors[0] is None:
        sampler_args = parameters.bounds
        # else:
        #     sampler_args = {key : (parameters.vector_to_simulation_inputs(priors[0])[key], 
        #                            parameters.vector_to_simulation_inputs(priors[1])[key]) 
        #                            for key in parameters.names.keys()}
        samplers = {key : partial(sampler, sampler_args[key]) for key, sampler in sampler_generators.items()}

        return samplers



    def _sample_namer(self, indices):
        for i in range(indices[0], indices[1]):
            yield self.output_base_path + f"/sim_{i}.h5"

    def clear_all_outputs(self):
        # just delete the output folder
        import shutil
        shutil.rmtree(self.output_base_path)