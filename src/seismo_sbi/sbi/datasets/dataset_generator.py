"""Run the forward model over drawn parameter sets in parallel, one HDF5 simulation per sample.

:class:`DatasetGenerator` takes the simulation inputs (from
:meth:`~seismo_sbi.priors.parameter_sampler.ParameterSampler.draw_simulation_inputs`) and the path
each simulation is written to.
"""

import logging
import joblib
import traceback

from seismo_sbi.utils.parallel import tqdm_joblib, worker_seeds

from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetGenerator:

    def __init__(self, simulator, num_parallel_jobs=1, seed=None):

        self.simulator = self._error_handling_wrapper(simulator)
        self.num_parallel_jobs = num_parallel_jobs
        #: Seeds each simulation's ensemble-member draw when set; None leaves the draws unseeded.
        self.seed = seed

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
    
    def run_and_save_simulations(self, simulation_inputs, output_paths):
        """Simulate each input map of ``simulation_inputs`` and write it to the matching path of
        ``output_paths``."""
        simulation_job_args_list = list(zip(simulation_inputs, output_paths))
        if self.seed is not None:
            member_seeds = worker_seeds(self.seed, len(simulation_job_args_list), "training members")
            simulation_job_args_list = [({**inputs, "seed": member_seed}, path) for (inputs, path), member_seed
                                        in zip(simulation_job_args_list, member_seeds)]
        self.run_parallel_simulations(simulation_job_args_list)

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

