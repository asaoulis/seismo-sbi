"""Shared error types and the two retry wrappers.

:func:`error_handling_wrapper` retries a flaky forward-model call a fixed number of times and then
raises; :func:`skip_after_retries` retries a simulation and then skips it.
"""

import traceback
from functools import wraps


class InvalidConfiguration(Exception):
    """A configuration file asks for something the pipeline cannot build."""


class PipelineStateError(Exception):
    """A pipeline step runs before the step that prepares what it needs."""


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


def skip_after_retries(simulation_callable, num_attempts=3):
    """``simulation_callable`` returning True once a call succeeds within ``num_attempts``, else
    False: a sampled source outside the forward model's valid domain is skipped, not fatal."""

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

        print(f"Simulation FAILED after {num_attempts} attempts; SKIPPING sample. "
              f"Last error: {type(last_exc).__name__}: {last_exc}")
        return False

    return _error_handled_simulation_callable