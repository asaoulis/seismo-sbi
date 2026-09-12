"""Progress reporting for joblib-parallel loops.

``tqdm_joblib`` is a context manager that patches ``joblib.parallel.Parallel.print_progress`` so
a parallel loop advances the tqdm bar given to it, and restores the original on exit.
"""

import contextlib

import joblib


# Monkey-patch of joblib to report into tqdm progress bar,
# solution taken from https://stackoverflow.com/a/61689175
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

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
