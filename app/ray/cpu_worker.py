import traceback
import asyncio
from asyncio import Semaphore
import uuid
from datetime import datetime
import ray
from app.algos.manager import AlgoManager
from app.helpers import create_exception_json
from app.logger import logger
from app.ray.timing_base import TimingBase, measure_time
from app.sentry import sentry_sdk

@ray.remote(max_concurrency=5)
class CPUWorker(TimingBase):
    def __init__(
        self,
        gpu_embedding_workers,
        gpu_classifier_workers,
        network_workers,
        cache,
        compute_environment=None,
    ):
        """
        Initialize the CPUWorker with references to a GPUWorker and NetworkWorkers.

        Args:
            gpu_worker: A reference to the GPUWorker actor.
            network_workers: A list of references to NetworkWorker actors.
        """
        self.gpu_embedding_workers = gpu_embedding_workers
        self.gpu_classifier_workers = gpu_classifier_workers
        self.network_workers = network_workers
        self.cache = cache
        self.compute_environment = compute_environment
        self.semaphore = Semaphore(30)
        self.active_tasks = 0
        self._max_concurrency = 5
        super().__init__()

    async def max_concurrency(self):
        return self._max_concurrency

    async def get_active_task_count(self):
        return self.active_tasks

    @measure_time
    async def process_manifest(self, algorithm_id, manifest, records):
        async with self.semaphore:
            count = 0
            self.active_tasks += 1
            print(f"Processing {algorithm_id}")
            timing = 0
            response = {
                "algorithm_id": algorithm_id,
                "compute_time": datetime.utcnow().isoformat(),
                "uuid": str(uuid.uuid4()),
            }
            try:
                algo_manager = await AlgoManager.initialize(
                    manifest,
                    self.gpu_embedding_workers,
                    self.gpu_classifier_workers,
                    self.network_workers,
                    self.cache,
                )
                gpu_accelerated = await algo_manager.is_gpu_accelerated()
                matched_records, _, timing = await algo_manager.matching_records(
                    records
                )
                response["compute_environment"] = "gpu" if gpu_accelerated else "cpu"
                response["compute_amount"] = timing
                response["matches"] = matched_records
                count = len(response["matches"])
            except Exception as e:
                sentry_sdk.capture_exception(e)
                logger.error(
                    f"Error while processing records with algorithm {algorithm_id}. "
                    f"Error: {e}"
                )
                error_dict = create_exception_json(e)
                error_dict["algorithm_id"] = algorithm_id
                response["error"] = error_dict
                response["compute_environment"] = "none"
                response["compute_amount"] = 0
                response["matches"] = []
                logger.error(traceback.format_exc())
            finally:
                print(f"Finished {algorithm_id}, took {timing}, {count} matches")
                self.cache.report_output.remote(response)
                self.active_tasks -= 1

    @measure_time
    async def process_batch(self, records, manifests):
        """
        Process a batch of records using the manifest.

        Args:
            records: A list of input data records.
            manifest: Metadata or configuration used to process the records.

        Returns:
            List of processed record results.
        """
        processes = []
        for algorithm_id, manifest in manifests:
            processes.append(self.process_manifest(algorithm_id, manifest, records))
        results = await asyncio.gather(*processes)
        return results
