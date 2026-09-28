"""Draw parameter sets from the prior and run the forward model over them in parallel.

:class:`DatasetGenerator` writes one HDF5 simulation per sample; the sampler functions below
turn a configuration block into the callables that draw each parameter. A sampler takes bounds
and a sample count and returns an array of draws.
"""

import logging
from abc import ABC, abstractmethod
import joblib
import traceback

from functools import partial

import numpy as np

from seismo_sbi.sbi.configuration import ModelParameters
from seismo_sbi.simulators.cps.compatibility import load_velocity_model
from seismo_sbi.utils.parallel import tqdm_joblib, worker_seeds

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ParallelSimulationRunner(ABC):

    def __init__(self, simulator, num_parallel_jobs):

        self.simulator = self._error_handling_wrapper(simulator)
        self.num_parallel_jobs = num_parallel_jobs

    def _error_handling_wrapper(self, simulation_callable, num_attempts = 3):

        def _error_handled_simulation_callable(*args, **kwargs):

            # Bound outside the except block, whose target Python deletes on exit, so the
            # real worker error survives to be re-raised.
            last_exc = None
            for attempt_number in range(num_attempts):
                try:
                    simulation_callable(*args, **kwargs)
                    return True
                except Exception as exc:
                    last_exc = exc
                    # Printed, not logged: this runs in joblib worker processes, which carry no
                    # logging handlers of their own.
                    print(f"Simulation terminated with exception {attempt_number + 1} times:")
                    print(traceback.format_exc())
                    print("Retrying simulation...")

            # A sampled source can fall outside the forward model's valid domain, which no
            # retry recovers; skip it and let run_parallel_simulations catch a large fraction.
            print(f"Simulation FAILED after {num_attempts} attempts; SKIPPING sample. "
                  f"Last error: {type(last_exc).__name__}: {last_exc}")
            return False
            
        
        return _error_handled_simulation_callable
    
    @abstractmethod
    def run_and_save_simulations(self, input_generator, num_parallel_jobs=1):
        pass

    def run_parallel_simulations(self, simulation_job_args_list):

        if self.num_parallel_jobs not in [0, 1]:
            try:
                with tqdm_joblib(tqdm(desc="Running simulations: ", total=len(simulation_job_args_list))):
                    with joblib.parallel_backend('loky', n_jobs=self.num_parallel_jobs):
                        results = joblib.Parallel()(
                            joblib.delayed(self.simulator)(*simulation_job_args) for
                                simulation_job_args in simulation_job_args_list
                        )
            except Exception as exc:
                logger.warning("Parallel simulations failed. Exiting.")
                raise exc
            finally:
                from joblib.externals.loky import get_reusable_executor
                # reuse=True kills the pool Parallel used; with default arguments loky would first
                # restart that pool gracefully, which can hang on a worker that never exits.
                get_reusable_executor(reuse=True).shutdown(wait=True, kill_workers=True)
        else:
            results = [self.simulator(*simulation_job_args)
                       for simulation_job_args in simulation_job_args_list]

        self._guard_against_excessive_skips(results)

    @staticmethod
    def _guard_against_excessive_skips(results, max_skip_fraction=0.2):
        """Report the skipped simulations, and raise if too large a fraction failed.

        A few out-of-domain sources are expected; a large fraction means a systemic problem
        that would otherwise produce a near-empty dataset in silence.
        """
        total = len(results)
        n_skipped = sum(1 for r in results if r is False)
        if not n_skipped:
            return
        fraction = n_skipped / total if total else 0.0
        logger.warning(f"[dataset_generator] {n_skipped}/{total} simulations skipped "
                       f"({fraction:.2%}) after exhausting retries.")
        if fraction > max_skip_fraction:
            raise RuntimeError(
                f"Aborting dataset generation: {fraction:.1%} of simulations failed "
                f"(> {max_skip_fraction:.0%} threshold). This indicates a systemic "
                "problem (bad DB path / config / broken forward model), not rare "
                "out-of-domain source draws."
            )
        
    def _create_sampler_transformer(self, parameters : ModelParameters):
        return lambda sampled_vector: parameters.vector_to_simulation_inputs(sampled_vector)


from scipy.stats.qmc import LatinHypercube


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

def flatten_sample(values):
    """One draw per parameter -> the flat sample vector ``vector_to_simulation_inputs`` reads.

    A scalar draw takes one slot, a 1-D draw takes one slot per element, and anything else
    (a velocity model) takes one slot holding the object itself.
    """
    flat = []
    for v in values:
        if np.isscalar(v):
            flat.append(v)
        elif isinstance(v, (list, tuple)):
            flat.extend(v)
        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                flat.extend(v)
            else:
                flat.append(v)
        else:
            flat.append(v)
    return np.array(flat, dtype=object)

def transform_sampling_func(sampling_func, transform_func):
    def wrapper(*args, **kwargs):
        for value in sampling_func(*args, **kwargs):
            yield transform_func(flatten_sample(value))
    return wrapper

class MomentTensorLogScaleHomogeneous:

    @staticmethod
    def _generate_sample(bounds):

        # Parametrisation of Stahler and Sigloch (2014), section 2.2.
        x = [np.random.uniform(0, 1) for _ in range(5)]
        Y3 = 1
        Y2 = np.sqrt(x[1])
        Y1 = Y2*x[0]

        M0 = np.exp(np.random.uniform(np.log(bounds[0]), np.log(bounds[1])))

        M_xx = np.sqrt(Y1) * np.cos(2*np.pi*x[2]) * np.sqrt(2) * M0
        M_yy = np.sqrt(Y1) * np.sin(2*np.pi*x[2]) * np.sqrt(2) * M0
        M_zz = np.sqrt(Y2 - Y1) * np.cos(2*np.pi*x[3]) * np.sqrt(2) * M0
        M_xy = np.sqrt(Y2 - Y1) * np.sin(2*np.pi*x[3]) * M0
        M_yz = np.sqrt(Y3 - Y2) * np.cos(2*np.pi*x[4]) * M0
        M_xz = np.sqrt(Y3 - Y2) * np.sin(2*np.pi*x[4]) * M0

        # Into the (r, theta, phi) order, after Aki and Richards (2002) p. 113.
        M = [M_zz, M_xx, M_yy, M_xz, -M_yz, -M_xy]

        return np.array(M)

    @staticmethod
    def sampler(bounds, num_samples):
        for _ in range(num_samples):
            yield MomentTensorLogScaleHomogeneous._generate_sample(bounds)

from itertools import chain

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

from seismo_sbi.simulators.cps.CPS import perturb_model
from seismo_sbi.simulators.cps.smooth_perturbations import perturb_cps_model

class VelocityModelSampler:
    perturbation_methods = {
        "default": perturb_model,
        "smooth": perturb_cps_model
    }
    
    def __init__(self, velocity_model, kappa, num_samples, *args):
        self.velocity_model = velocity_model
        self.kappa = kappa
        self.num_samples = num_samples
        self.kwargs = {}
        if len(args) > 0 and args[0] == "smooth":
            self.perturbation_function = self.perturbation_methods["smooth"]
            self.kwargs = {'corr_length_km': 5.0,
                           'std_vp': kappa/100,
                           'std_vs': kappa/100,}
            logger.info(f"Using smooth perturbations with kappa={kappa}")
        else:
            self.perturbation_function = self.perturbation_methods["default"]
            self.kwargs['kappa'] = kappa

    def __iter__(self):
        for _ in range(self.num_samples):
            yield self.perturbation_function(self.velocity_model, **self.kwargs)

def velocity_model_sampler(velocity_model_args, num_samples):
    """A generator of perturbed velocity models."""
    velocity_model_path, kappa, *options = velocity_model_args
    velocity_model = load_velocity_model(velocity_model_path)
    sampler = iter(VelocityModelSampler(velocity_model, kappa, num_samples, *options))
    return sampler


class DatasetGenerator(ParallelSimulationRunner):

    sampler_lookup_map = {"latin hypercube" : latin_hypercube_sampler,
                          "uniform"         : uniform_sampler,
                          "uniform known"         : uniform_sampler,
                          "gaussian"        : gaussian_sampler,                     
                          "moment tensor log prior"   : MomentTensorLogScaleHomogeneous.sampler,
                          "constant": constant_sampler,
                          "truncated gaussian": truncated_gaussian_sampler,
                          "velocity model": velocity_model_sampler}

    def __init__(self, simulator, output_base_path, num_parallel_jobs=1, seed=None):
        super().__init__(simulator, num_parallel_jobs)

        self.output_base_path = output_base_path
        self.num_parallel_jobs = num_parallel_jobs
        #: Seeds each simulation's ensemble-member draw when set; None leaves the draws unseeded.
        self.seed = seed


    def run_and_save_simulations(self, parameters : ModelParameters, sampler_details, indices, sample_namer = None, priors= (None, None)):
        if sample_namer is None:
            sample_namer = self._sample_namer
        try:
            num_samples = indices[1] - indices[0]
        except TypeError:
            num_samples = indices

        samplers = self._create_sampler_generator_dict(parameters, sampler_details, priors=priors)
        if priors[0] is None:
            sampler_args = self._sampler_args(parameters, samplers)
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
        if self.seed is not None:
            member_seeds = worker_seeds(self.seed, len(simulation_job_args_list), "training members")
            simulation_job_args_list = [({**inputs, "seed": member_seed}, path) for (inputs, path), member_seed
                                        in zip(simulation_job_args_list, member_seeds)]
        self.run_parallel_simulations(simulation_job_args_list)
    
    def run_predefined_batch(self, thetas, indices, parameters : ModelParameters):

        simulation_parameters = [parameters.vector_to_simulation_inputs(theta) for theta in thetas]
        input_generator = zip(simulation_parameters, self._sample_namer(indices))
        simulation_job_args_list = [input_config for input_config in input_generator]
        self.run_parallel_simulations(simulation_job_args_list)
    
    @staticmethod
    def _resolve_sampler(entry):
        """The ``(args, num_samples)`` sampler a ``sampling_method`` entry names.

        The entry is either the name of a built-in sampler or an already-built closure the
        configuration produced.
        """
        if callable(entry):
            return entry
        return DatasetGenerator.sampler_lookup_map[entry]

    @staticmethod
    def _create_sampler_generator_dict(parameters : ModelParameters, sampler_details, priors = (None, None)):
        if priors[0] is None:
            samplers = {key : DatasetGenerator._resolve_sampler(sampler_details[key]) for key in chain(parameters.names.keys(), parameters.nuisance.keys())}
        else:
            samplers = {key : DatasetGenerator.sampler_lookup_map['truncated gaussian'] for key in chain(parameters.names.keys(), parameters.nuisance.keys())}

        return samplers

    @staticmethod
    def _sampler_args(parameters : ModelParameters, sampler_generators):
        """The first argument each sampler is bound to: bounds, except `constant`, which takes the fiducial.

        A constant parameter must contribute exactly ``len(fiducial)`` slots to the sample vector,
        because that is what ``ModelParameters.vector_to_simulation_inputs`` consumes; its bounds
        pair would contribute two and shift every parameter after it.
        """
        return {key: (parameters.get_parameter_values(key) if sampler is constant_sampler
                      else parameters.bounds[key])
                for key, sampler in sampler_generators.items()}

    @staticmethod
    def create_samplers(parameters : ModelParameters, sampler_details, priors = (None, None)):
        sampler_generators = DatasetGenerator._create_sampler_generator_dict(parameters, sampler_details, priors)
        sampler_args = DatasetGenerator._sampler_args(parameters, sampler_generators)
        samplers = {key : partial(sampler, sampler_args[key]) for key, sampler in sampler_generators.items()}

        return samplers



    def _sample_namer(self, indices):
        for i in range(indices[0], indices[1]):
            yield self.output_base_path + f"/sim_{i}.h5"

    def clear_all_outputs(self):
        pass
