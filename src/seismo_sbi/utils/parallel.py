"""joblib helpers: a mapped loop and progress reporting.

``parallel_execution`` maps a function over inputs, serially for one job. ``tqdm_joblib`` patches
joblib so a parallel loop advances a tqdm bar given to it, and restores it on exit. Inside a
Jupyter kernel both pause garbage collection while workers start (``gc_paused_in_notebooks``).
``worker_seeds`` gives each task its own seed, since workers start with fresh random state.
"""

import contextlib
import gc
import sys
import zlib

import joblib
import numpy as np


# After https://stackoverflow.com/a/61689175
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to advance ``tqdm_object`` as a parallel loop completes tasks."""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress
    try:
        with gc_paused_in_notebooks():
            yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def parallel_execution(inputs, func, num_jobs = 20):
    """``[func(x) for x in inputs]``, with ``num_jobs`` joblib workers unless it is 0, 1 or None."""
    if num_jobs in [None, 0 , 1]:
        return [func(block) for block in inputs]
    with gc_paused_in_notebooks():
        return joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(func)(block) for block in inputs)


@contextlib.contextmanager
def gc_paused_in_notebooks():
    """Inside a Jupyter kernel, pause garbage collection while worker processes are forked.

    ipykernel registers a collection callback that takes threading's global lock. A forked child
    resets its threads while holding that lock, so a collection at that moment deadlocks the child
    for ever. The child inherits the paused collector. Outside a kernel nothing changes.
    """
    if "ipykernel" not in sys.modules or not gc.isenabled():
        yield
        return
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def worker_seeds(seed, num_tasks, stream):
    """``num_tasks`` independent integer seeds drawn from ``seed`` for the named ``stream``, or
    ``num_tasks`` Nones when ``seed`` is None. Different streams give unrelated seeds.
    """
    if seed is None:
        return [None] * num_tasks
    entropy = [seed, zlib.crc32(stream.encode())]
    return [int(task_seed) for task_seed in np.random.SeedSequence(entropy).generate_state(num_tasks)]
