"""joblib helpers: a mapped loop and progress reporting.

``parallel_execution`` maps a function over inputs, serially for one job. ``tqdm_joblib`` is a
context manager that patches ``joblib.parallel.Parallel.print_progress`` so a parallel loop
advances the tqdm bar given to it, and restores the original on exit.
"""

import contextlib

import joblib


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
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def parallel_execution(inputs, func, num_jobs = 20):
    """``[func(x) for x in inputs]``, with ``num_jobs`` joblib workers unless it is 0, 1 or None."""
    if num_jobs in [None, 0 , 1]:
        return [func(block) for block in inputs]
    return joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(func)(block) for block in inputs)
