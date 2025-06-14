import asyncio
import time
from functools import wraps
from app.logger import logger
from app.redis import RedisClient
from typing import Optional
from app.utils.profilers.adapters import GrafanaAdapter


def elapsed_time(start_time: float) -> float:
    return time.time() - start_time


def record_timing(prefix: Optional[str] = None, annotate: bool = False):
    """
    Decorator to measure and record execution time to Grafana, Prometheus, Redis or other data sources
    """

    def record_timing_decorator(func):
        def field(func) -> str:
            return f"{prefix}:{func.__name__}" if prefix else func.__name__

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = await func(self, *args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            identifier = field(func)
            logger.info(f"ELAPSED TIME: {identifier} - {elapsed_time}")
            try:
                await RedisClient.set_process_timing(identifier, elapsed_time)
                if annotate:
                    logger.info(f"Annotating {identifier}")
                    await GrafanaAdapter(
                        tags=[f"{func.__name__}", f"{func.__class__}"]
                    ).annotate_session(
                        txt=f"{identifier}-{start_time}",
                        time=start_time,
                        time_end=end_time,
                    )
                return result
            except Exception as e:
                logger.warn(f"Cannot record timing for func: {identifier}")
                logger.warn(e)
                return result

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            identifier = field(func)
            logger.info(f"ELAPSED TIME: {field(func)} - {elapsed_time}")
            try:
                _ = RedisClient.set_process_timing(field(func), elapsed_time)
                if annotate:
                    logger.info(f"Annotating {identifier}")
                    _ = GrafanaAdapter(
                        tags=[f"{func.__name__}", f"{func.__class__}"]
                    ).annotate_session(
                        txt=identifier, time=start_time, time_end=end_time
                    )
                return result
            except Exception as e:
                logger.warn(f"Cannot record timing for func: {identifier}")
                logger.warn(e)
                return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return record_timing_decorator
