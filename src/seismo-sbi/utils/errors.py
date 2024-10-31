import traceback
from functools import wraps

def error_handling_wrapper(num_attempts=3):
    def decorator(simulation_callable):
        @wraps(simulation_callable)
        def _error_handled_simulation_callable(*args, **kwargs):
            for attempt_number in range(num_attempts):
                try:
                    return simulation_callable(*args, **kwargs)
                except Exception as exc:
                    # Error handling with the function name printed
                    func_name = simulation_callable.__name__
                    print(f"{func_name} terminated with exception {attempt_number + 1} times:")
                    print(''.join(traceback.format_exception(None, exc, exc.__traceback__)))
                    print(f"Retrying {func_name}...")

            print(f"{simulation_callable.__name__} failed after multiple attempts. Exiting.")
            raise exc
        
        return _error_handled_simulation_callable
    return decorator