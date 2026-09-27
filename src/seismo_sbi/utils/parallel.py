"""joblib helpers: a mapped loop and progress reporting.

``parallel_execution`` maps a function over inputs, serially for one job. ``tqdm_joblib`` patches
joblib so a parallel loop advances a tqdm bar given to it, and restores it on exit. Inside a
Jupyter kernel both start their workers with ``spawn`` (``spawn_workers_in_notebooks``).
"""

import contextlib
import sys

import joblib


# After https://stackoverflow.com/a/61689175
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to advance ``tqdm_object`` as a parallel loop completes tasks."""
    spawn_workers_in_notebooks()

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


def parallel_execution(inputs, func, num_jobs = 20):
    """``[func(x) for x in inputs]``, with ``num_jobs`` joblib workers unless it is 0, 1 or None."""
    if num_jobs in [None, 0 , 1]:
        return [func(block) for block in inputs]
    spawn_workers_in_notebooks()
    return joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(func)(block) for block in inputs)


def spawn_workers_in_notebooks():
    """In a Jupyter kernel, start loky workers with ``spawn``; elsewhere change nothing.

    ipykernel wraps ``Thread.__init__`` in a hook that takes threading's global lock. loky's default
    start runs that hook in the forked child before its exec, and the child waits for ever when
    another thread held the lock at the fork; ``spawn`` execs straight after forking.
    """
    if "ipykernel" in sys.modules:
        from joblib.externals.loky.backend.context import set_start_method
        set_start_method("spawn", force=True)
