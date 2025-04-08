from app.runpod.base import RunpodBase
from app.jetstream import Jetstream
from app.redis import RedisClient
from app.sentry import sentry_sdk

class RunpodBacktester(RunpodBase):
    @classmethod
    async def check_for_killswitch(cls, task_id):
        killed_key = f"{task_id}_killed"
        killed = await RedisClient.get(killed_key)
        if killed:
            await cls.publish_status(task_id, {"status": "Received killswitch..."})
            return True
        return False

    @classmethod
    async def startup_manager(cls, dispatcher, task_id, manifest):
        await cls.publish_status(
            task_id, {"status": "Job has been received by a live feed analyzer..."}
        )
        await cls.publish_status(task_id, {"status": "Loading feed analyzer..."})
        manager = await cls.initialize_algo(dispatcher, manifest, task_id)
        if not manager:
            return
        return manager

    @classmethod
    async def run_crawl(cls, task_id, manager, batch_size=1000, max_match_count=500):
        record_count = 0
        match_count = 0
        await cls.publish_status(
            task_id, {"status": "Starting crawl on historic data..."}
        )
        batch = []
        matched_records = []
        async for record in Jetstream.yield_jetstream_reversed():
            batch.append(record)
            if len(batch) >= batch_size:
                record_count += len(batch)
                print("Processing batch...")
                try:
                    matched_records = await manager.matching_records(
                        list(reversed(batch))
                    )
                except Exception as e:
                    sentry_sdk.capture_exception(e)
                    await cls.publish_status(
                        task_id,
                        {
                            "error": "Algorithm couldn't analyze these records - please check your logic and the posts you provided."
                        },
                    )
                if matched_records:
                    match_count += len(matched_records[0])
                print("Processed batch.")
                await cls.publish_status(
                    task_id,
                    {
                        "matched_records": matched_records,
                        "record_count": record_count,
                        "match_count": match_count,
                    },
                )
                batch.clear()
                if match_count >= max_match_count:
                    await cls.publish_status(
                        task_id,
                        {
                            "status": f"Finished running to max count of {max_match_count}, stopping..."
                        },
                    )
                    await cls.publish_status(task_id, {"finished": True})
                    return
                if await cls.check_for_killswitch(task_id):
                    return

        # If there are leftover records in the final partial batch, process them
        if batch:
            record_count += len(batch)
            matched_records = await manager.matching_records(list(reversed(batch)))
            await cls.publish_status(
                task_id,
                {
                    "matched_records": matched_records,
                    "record_count": record_count,
                    "match_count": match_count,
                },
            )

    @classmethod
    async def live_query(cls, dispatcher, task_id, manifest):
        """
        Pull data from Jetstream in reverse order, gather into batches, then
        process each batch with `match_records_for_manifest`. Otherwise, the
        logic remains the same as your existing `live_query`.
        """
        manager = await cls.startup_manager(dispatcher, task_id, manifest)
        if await cls.check_for_killswitch(task_id):
            return
        await cls.run_crawl(task_id, manager)
