import asyncio
import time
from functools import wraps


def measure_time(func):
    """
    Decorator to measure and record execution time.
    """

    @wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = await func(self, *args, **kwargs)
        elapsed_time = time.time() - start_time
        if hasattr(self, "_timings"):
            self._timings[func.__name__] = (
                self._timings.get(func.__name__, 0) + elapsed_time
            )
        return result

    @wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        elapsed_time = time.time() - start_time
        if hasattr(self, "_timings"):
            self._timings[func.__name__] = (
                self._timings.get(func.__name__, 0) + elapsed_time
            )
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class TimingBase:
    """
    Base class for timing methods. Wraps methods after initialization.
    """

    def __init__(self):
        self._timings = {}

    def get_summary(self):
        """
        Return timing summary as a dictionary.
        """
        return {k: round(v, 3) for k, v in self._timings.items()}
