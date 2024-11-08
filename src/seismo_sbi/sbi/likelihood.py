
import numpy as np
import emcee
from emcee.moves import GaussianMove
from multiprocessing import Pool
# from multiprocess import Pool
# from concurrent.futures import ProcessPoolExecutor as Pool
import os
from tqdm import tqdm
from functools import partial
import joblib
from ..instaseis_simulator.dataset_generator import tqdm_joblib

class GaussianLikelihoodEvaluator:

    def __init__(self, data, simulation_callable, scaler, loss_callable, ensemble = False, priors= (None, None)):
        self.data = data
        self.simulation_callable = simulation_callable
        self.scaler = scaler
        self.loss_callable = loss_callable
        self.ensemble = ensemble
        if priors[0] is None:
            self.log_prior = self.default_log_prior
        else:
            mean, covariances = priors
            self.log_prior = partial(self.gaussian_prior, mean=mean, covariances=covariances, scaler=scaler)

    def default_log_prior(self, scaled_theta):
        if np.any(scaled_theta < 0) or np.any(scaled_theta > 1):
            return -np.inf
        return 0

    def gaussian_prior(self, scaled_theta, mean, covariances, scaler):
        if np.any(scaled_theta < 0) or np.any(scaled_theta > 1):
                return -np.inf
        theta = scaler.inverse_transform(scaled_theta.reshape(1,-1)).flatten()
        return -0.5 * np.dot((theta - mean).T, np.dot(np.linalg.inv(np.diag(covariances)), (theta - mean)))

    def log_likelihood(self, scaled_source_parameters):

        source_parameters = self.scaler.inverse_transform(scaled_source_parameters.reshape(1,-1))
        synthetic_waveform = self.simulation_callable(source_parameters.flatten())
        diff = synthetic_waveform - self.data
        return self.loss_callable(diff)

    def log_probability(self, scaled_source_parameters):
        log_prior_value = self.log_prior(scaled_source_parameters)
        if not np.isfinite(log_prior_value):
            return -np.inf
        return self.log_likelihood(scaled_source_parameters) + log_prior_value

def run_embarrassingly_parallel_simulations(num_parameters, log_probability, burn_in, nsamples_per_walker, theta0, move_size, thin=5, return_sampler=False):
    if isinstance(move_size, list):
         first_size, second_size = tuple(move_size)
    else:
        first_size = move_size
        second_size = move_size
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    sampler = emcee.EnsembleSampler(1, num_parameters, log_probability, moves = GaussianMove(first_size))
    initial  =  np.random.rand(1, num_parameters)
    state = sampler.run_mcmc(initial, burn_in, skip_initial_state_check=True, progress=True, progress_kwargs=dict(position=0, leave=True))
    state = sampler.get_chain()[-1]
    sampler.reset()

    sampler = emcee.EnsembleSampler(1, num_parameters, log_probability, moves = GaussianMove(second_size))

    sampler.run_mcmc(state, nsamples_per_walker, skip_initial_state_check=True, progress=True, progress_kwargs=dict(position=0, leave=True))
    if return_sampler:
        return sampler
    return sampler.get_chain(flat=True, thin=1)


def generate_samples(log_probability, ensemble, num_parameters, nsamples_per_walker, nwalkers, burn_in=1000, num_processes=1, theta0=None, move_size=None):


    initial_samples = np.random.rand(nwalkers, num_parameters)
    if ensemble:
        with Pool(processes=num_processes) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, num_parameters, log_probability, pool=pool)

            # burn in
            print("Starting burn in...", flush=True)
            state = sampler.run_mcmc(initial_samples, burn_in, progress=True)
            sampler.reset()

            print("Starting posterior sampling..", flush=True)
            sampler.run_mcmc(state, nsamples_per_walker, progress=True)
            
        samples = sampler.get_chain(flat=True)
    else:
        with tqdm_joblib(tqdm(desc="Running MCMC chains: ", total=num_processes, position=0, leave=True)) as progress_bar:
                        with joblib.parallel_backend('loky', n_jobs=num_processes):
                            samples = joblib.Parallel()(
                                joblib.delayed(run_embarrassingly_parallel_simulations)(num_parameters, log_probability, burn_in, nsamples_per_walker, theta0, move_size) for
                                    _ in range(num_processes)
                            )
        samples = np.stack(samples).reshape(num_processes, -1, num_parameters).transpose(1,0,2).reshape(-1, num_parameters)
    return samples
