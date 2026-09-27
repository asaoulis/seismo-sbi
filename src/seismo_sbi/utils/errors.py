"""Shared error types and the retry wrapper.

:func:`error_handling_wrapper` retries a flaky forward-model call a fixed number of times.
"""

import traceback
from functools import wraps


class InvalidConfiguration(Exception):
    """A configuration file asks for something the pipeline cannot build."""


def error_handling_wrapper(num_attempts=3):
    def decorator(simulation_callable):
        @wraps(simulation_callable)
        def _error_handled_simulation_callable(*args, **kwargs):
            last_exc = None
            for attempt_number in range(num_attempts):
                try:
                    return simulation_callable(*args, **kwargs)
                except Exception as exc:
                    # Error handling with the function name printed
                    last_exc = exc
                    func_name = simulation_callable.__name__
                    print(f"{func_name} terminated with exception {attempt_number + 1} times:")
                    print(''.join(traceback.format_exception(None, exc, exc.__traceback__)))
                    print(f"Retrying {func_name}...")

            # Re-raise the last failure itself; the ``except ... as`` name is gone once the block exits.
            print(f"{simulation_callable.__name__} failed after multiple attempts. Exiting.")
            raise last_exc
        
        return _error_handled_simulation_callable
    return decorator