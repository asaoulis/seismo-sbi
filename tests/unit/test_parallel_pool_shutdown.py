"""A parallel simulation run kills the worker pool it ran on instead of starting another."""
from joblib.externals.loky import reusable_executor

from seismo_sbi.sbi.dataset_generator import ParallelSimulationRunner


class _Runner(ParallelSimulationRunner):
    def run_and_save_simulations(self, input_generator, num_parallel_jobs=1):
        pass


def _square(value):
    return value * value


def test_parallel_simulations_kill_the_pool_they_ran_on():
    _Runner(_square, num_parallel_jobs=2).run_parallel_simulations([(1,), (2,), (3,)])

    executor = reusable_executor._executor
    assert executor._flags.shutdown
    assert executor._max_workers == 2
