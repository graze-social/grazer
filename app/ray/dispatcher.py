import random
import asyncio
from app.ray.utils import discover_named_actors, discover_named_actor


class Dispatcher:
    def __init__(self):
        """
        Initialize the dispatcher with workers.
        Args:
            num_cpu_workers: Number of CPU workers.
            num_network_workers: Number of NetworkWorkers.
            gpu_worker: A reference to the GPUWorker actor.
            cache: A reference to the shared Cache actor.
        """
        print("Looking for cache...")
        self.cache = discover_named_actor("cache:", timeout=10)
        print("Looking for Bluesky Semaphore...")
        self.bluesky_semaphore = discover_named_actor("semaphore:bluesky", timeout=10)
        print("Looking for Graze Semaphore...")
        self.graze_semaphore = discover_named_actor("semaphore:graze", timeout=10)
        print("Looking for Network Workers...")
        self.network_workers = discover_named_actors("network:", timeout=10)
        print("Looking for GPU Worker...")
        self.gpu_embedding_workers = discover_named_actors("gpu:embedders", timeout=10)
        self.gpu_classifier_workers = discover_named_actors(
            "gpu:classifiers", timeout=10
        )
        print("Looking for CPU Workers...")
        self.cpu_workers = discover_named_actors("cpu:", timeout=10)
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

    async def distribute_tasks(self, records, manifest_data):
        for manifest in manifest_data:
            while True:
                worker = random.choice(self.cpu_workers)
                max_concurrency = await worker.max_concurrency.remote()
                active_tasks = await worker.get_active_task_count.remote()
                if active_tasks < max_concurrency:
                    worker.process_batch.remote(records, [manifest])
                    break
                else:
                    await asyncio.sleep(0.1)
