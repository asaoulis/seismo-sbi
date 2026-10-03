"""Parallel stages kill the worker pool they ran on, and pause garbage collection while forking in a notebook."""
import gc
import sys

from joblib.externals.loky import reusable_executor

from seismo_sbi.sbi.datasets.dataset_generator import DatasetGenerator
from seismo_sbi.utils.parallel import gc_paused_in_notebooks


def _square(value):
    return value * value


def test_parallel_simulations_kill_the_pool_they_ran_on():
    DatasetGenerator(_square, num_parallel_jobs=2).run_parallel_simulations([(1,), (2,), (3,)])

    executor = reusable_executor._executor
    assert executor._flags.shutdown
    assert executor._max_workers == 2


def test_garbage_collection_pauses_while_forking_only_inside_a_jupyter_kernel(monkeypatch):
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    with gc_paused_in_notebooks():
        assert gc.isenabled()

    monkeypatch.setitem(sys.modules, "ipykernel", object())
    with gc_paused_in_notebooks():
        assert not gc.isenabled()
    assert gc.isenabled()
