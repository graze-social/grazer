import asyncio
import time
from functools import wraps

from app.logger import logger
from app.redis import RedisClient
from typing import Optional


def record_progress_in_logs():
    pass


def record_timing(fn_prefix: Optional[str] = None):
    """
    Decorator to measure and record execution time to Redis.
    """

    def record_timing_decorator(func):
        def field(func) -> str:
            return f"{fn_prefix}:{func.__name__}" if fn_prefix else func.__name__

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = await func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"ELAPSED TIME: {field(func)} - {elapsed_time}")
            await RedisClient.set_process_timing(field(func), elapsed_time)
            return result

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"ELAPSED TIME: {field(func)} - {elapsed_time}")
            _ = RedisClient.set_process_timing(field(func), elapsed_time)
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return record_timing_decorator
