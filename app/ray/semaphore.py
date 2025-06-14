import ray
import asyncio
from app.ray.timing_base import TimingBase, measure_time


@ray.remote(num_cpus=0.5, num_gpus=0, max_restarts=-1)
class SemaphoreActor(TimingBase):
    def __init__(self, max_concurrent_requests=50):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        super().__init__()

    @measure_time
    async def acquire(self):
        """Acquire a semaphore slot."""
        await self.semaphore.acquire()

    @measure_time
    async def release(self):
        """Release a semaphore slot."""
        self.semaphore.release()
