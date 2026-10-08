"""Run the forward model over drawn parameter sets in parallel, one HDF5 simulation per sample.

:class:`DatasetGenerator` takes the simulation inputs (from
:meth:`~seismo_sbi.priors.parameter_sampler.ParameterSampler.draw_simulation_inputs`) and the path
each simulation is written to.
"""

import logging

from seismo_sbi.utils.errors import skip_after_retries
from seismo_sbi.utils.parallel import run_tasks, worker_seeds

logger = logging.getLogger(__name__)


class DatasetGenerator:

    def __init__(self, simulator, num_parallel_jobs=1, seed=None):

        self.simulator = skip_after_retries(simulator)
        self.num_parallel_jobs = num_parallel_jobs
        #: Seeds each simulation's ensemble-member draw when set; None leaves the draws unseeded.
        self.seed = seed

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

        results = run_tasks(self.simulator, simulation_job_args_list, self.num_parallel_jobs,
                            "Running simulations")

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

