import random
import asyncio
from app.logger import logger
from app.ray.utils import discover_named_actors, discover_named_actor
from app.omni.boot_settings import BootSettings
from app.timings import record_timing


class Dispatcher:
    def __init__(
        self,
        cache=None,
        bluesky_semaphore=None,
        graze_semaphore=None,
        network_workers=[],
        gpu_embedding_workers=[],
        gpu_classifier_workers=[],
        cpu_workers=[],
        actor_namespace=None,
    ):
        """
        Initialize the dispatcher with workers.
        Args:
            num_cpu_workers: Number of CPU workers.
            num_network_workers: Number of NetworkWorkers.
            gpu_worker: A reference to the GPUWorker actor.
            cache: A reference to the shared Cache actor.
        """
        namespace = actor_namespace if actor_namespace else BootSettings().namespace

        print("Looking for cache...")
        self.cache = cache or discover_named_actor("cache:", timeout=10)
        print("Looking for Bluesky Semaphore...")
        self.bluesky_semaphore = bluesky_semaphore or discover_named_actor(
            "semaphore:bluesky", timeout=10
        )
        print("Looking for Graze Semaphore...")
        self.graze_semaphore = graze_semaphore or discover_named_actor(
            "semaphore:graze", timeout=10
        )
        print("Looking for Network Workers...")
        self.network_workers = network_workers or discover_named_actors(
            "network:", timeout=10
        )
        print("Looking for GPU Worker...")
        self.gpu_embedding_workers = gpu_embedding_workers or discover_named_actors(
            f"gpu:{namespace}:embedders", timeout=10
        )
        self.gpu_classifier_workers = gpu_classifier_workers or discover_named_actors(
            f"gpu:{namespace}:classifiers", timeout=10
        )
        print("Looking for CPU Workers...")
        self.cpu_workers = cpu_workers or discover_named_actors("cpu:", timeout=10)
        super().__init__()

    async def generate_timing_report(self):
        """
        Generate a timing report for all actors managed by the dispatcher.
        Returns:
            A flat dictionary with timings sorted by seconds in descending order.
        """
        timing_report = {}
        # Collect timing data from SemaphoreActors
        semaphore_actors = [
            ("bluesky_semaphore", self.bluesky_semaphore),
            ("graze_semaphore", self.graze_semaphore),
        ]
        for name, actor in semaphore_actors:
            timings = await actor.get_summary.remote()
            for method, seconds in timings.items():
                timing_report[f"{name}.{method}"] = round(seconds, 3)
        # Collect timing data from NetworkWorkers
        for i, worker in enumerate(self.network_workers):
            timings = await worker.get_summary.remote()
            for method, seconds in timings.items():
                timing_report[f"network_worker_{i}.{method}"] = round(seconds, 3)
        # Collect timing data from GPUWorker
        for i, worker in enumerate(self.gpu_embedding_workers):
            timings = await worker.get_summary.remote()
            for method, seconds in timings.items():
                timing_report[f"gpu_embedding_worker_{i}.{method}"] = round(seconds, 3)
        for i, worker in enumerate(self.gpu_classifier_workers):
            timings = await worker.get_summary.remote()
            for method, seconds in timings.items():
                timing_report[f"gpu_classifier_worker_{i}.{method}"] = round(seconds, 3)
        # Collect timing data from CPUWorkers
        for i, worker in enumerate(self.cpu_workers):
            timings = await worker.get_summary.remote()
            for method, seconds in timings.items():
                timing_report[f"cpu_worker_{i}.{method}"] = round(seconds, 3)
        # Sort the report by timings in descending order
        sorted_timing_report = dict(
            sorted(timing_report.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_timing_report

    @record_timing(fn_prefix="Dispatcher")
    async def distribute_tasks(self, records, manifest_data, report_output=True):
        # [THEORY] 1.how long is this list and why does it take so long
        logger.warn(f"[warn debug] length of manifest data: {len(manifest_data)}")
        logger.warn(f"[warn debug] number of CPU Workers: {self.cpu_workers}")
        for manifest in manifest_data:
            worker = random.choice(self.cpu_workers)
            # max_concurrency = await worker.max_concurrency.remote()
            # logger.warn(f"[warn debug] max concurrency {max_concurrency}")
            # active_tasks = await worker.get_active_task_count.remote()
            # logger.warn(f"[warn debug] active tasks {active_tasks}")

            logger.warn(f"[warn debug] processing batch for {manifest[0]}")
            worker.process_batch.remote(records, [manifest], report_output)

            await asyncio.sleep(0.1)
