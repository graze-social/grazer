import uuid
from datetime import datetime, timedelta
from app.runpod.base import RunpodBase
from app.jetstream import Jetstream
from app.algos.manager import AlgoManager
from app.logger import logger

class RunpodBackfiller(RunpodBase):
    @classmethod
    async def run(cls, dispatcher, algorithm_id, manifest, batch_size=2000, max_match_count=500):
        logger.info(f"{algorithm_id}, {manifest}")
        manager = await AlgoManager.initialize(
            manifest,
            dispatcher.gpu_embedding_workers,
            dispatcher.gpu_classifier_workers,
            dispatcher.network_workers,
            dispatcher.cache,
        )
        record_count = 0
        match_count = 0
        batch = []
        last_write_time = datetime.utcnow()

        async for record in Jetstream.yield_jetstream_reversed():
            batch.append(record)
            now = datetime.utcnow()
            
            # Check if we should write out the batch (batch size or time elapsed)
            if len(batch) >= batch_size or (batch and (now - last_write_time) >= timedelta(seconds=60)):
                record_count += len(batch)
                logger.info("Processing batch...")
                matched_records, _, timing = await manager.matching_records(list(reversed(batch)))
                if matched_records:
                    match_count += len(matched_records)
                    response = {
                        "algorithm_id": algorithm_id,
                        "compute_time": now.isoformat(),
                        "uuid": str(uuid.uuid4()),
                        "matches": matched_records,
                        "compute_environment": await manager.is_gpu_accelerated(),
                        "compute_amount": timing
                    }
                    logger.info(response)
                    manager.cache.report_output.remote(response, True)
                batch.clear()
                last_write_time = now
                if match_count >= max_match_count:
                    return