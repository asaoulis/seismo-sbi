"""joblib helpers: a mapped loop and progress reporting.

``run_tasks`` maps a function over argument tuples on loky workers with a progress bar, serially
for one job. ``tqdm_joblib`` patches joblib so a parallel loop advances a tqdm bar given to it, and
restores it on exit. Inside a Jupyter kernel it pauses garbage collection while workers start
(``gc_paused_in_notebooks``).
``worker_seeds`` gives each task its own seed, since workers start with fresh random state.
"""

import contextlib
import gc
import logging
import sys
import zlib

import joblib
import numpy as np
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

logger = logging.getLogger(__name__)


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


def run_tasks(func, args_list, num_jobs, description):
    """``[func(*args) for args in args_list]``, serially for ``num_jobs`` None, 0 or 1, else on
    ``num_jobs`` loky workers behind a ``description`` progress bar. The worker pool is killed
    afterwards."""
    if num_jobs in [None, 0, 1]:
        return [func(*args) for args in args_list]
    try:
        with tqdm_joblib(tqdm(desc=description, total=len(args_list))):
            with joblib.parallel_backend('loky', n_jobs=num_jobs):
                return joblib.Parallel()(joblib.delayed(func)(*args) for args in args_list)
    except Exception:
        logger.warning(f"{description} failed.")
        raise
    finally:
        # reuse=True kills the pool Parallel used. With default arguments loky would first
        # restart that pool gracefully, which can hang on a worker that never exits.
        get_reusable_executor(reuse=True).shutdown(wait=True, kill_workers=True)


@contextlib.contextmanager
def gc_paused_in_notebooks():
    """Inside a Jupyter kernel, pause garbage collection while worker processes are forked.

    Outside a kernel nothing changes.
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
