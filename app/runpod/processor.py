import random
import asyncio

from app.logic_evaluator import LogicEvaluator
from app.logger import logger
from app.runpod.base import RunpodBase
from app.helpers import chunk
from app.redis import RedisClient
from app.settings import CURRENT_ALGORITHMS_KEY


class RunpodProcessor(RunpodBase):
    @classmethod
    async def ingest_feed(cls, transactions):
        records = []
        deletes = []
        for transaction in transactions:
            if transaction.get("commit", {}).get("operation") == "create":
                records.append(transaction)
            elif transaction.get("commit", {}).get("operation") == "delete":
                deletes.append(transaction)
        if deletes:
            await RedisClient.push_delete_transactions(deletes)
        return records

    @classmethod
    async def process_algos(cls, dispatcher, transactions):
        records = await cls.ingest_feed(transactions)
        algo_data = await RunpodProcessor.get_algorithm_operators()
        await RunpodProcessor.run_algos(
            dispatcher,
            records,
            algo_data["ALGORITHM_MANIFESTS"],
            algo_data["CONDITION_MAP"],
        )
        await dispatcher.cache.flush_to_egress.remote()

    @classmethod
    async def run_algos(cls, dispatcher, records, manifests, all_operators):
        manifests = list(manifests.items())
        random.shuffle(manifests)
        # await run_precache(dispatcher, records, [{}], all_operators)
        for manifest_chunk in chunk(manifests, 1000):
            hydrations = []
            algorithm_ids = []
            for algorithm_id, manifest in manifest_chunk:
                if manifest:
                    hydrations.append(
                        LogicEvaluator.rehydrate_single_manifest(
                            manifest, all_operators
                        )
                    )
                    algorithm_ids.append(algorithm_id)
        hydrated_manifests = await asyncio.gather(*hydrations)
        manifest_data = list(zip(algorithm_ids, hydrated_manifests))
        await dispatcher.distribute_tasks(records, manifest_data)
        while len(asyncio.all_tasks()) > 100:
            logger.info(f"Current Task Depth is {len(asyncio.all_tasks())}")
            asyncio.sleep(1)

    @classmethod
    async def get_algorithm_operators(cls):
        response = await RedisClient.get(CURRENT_ALGORITHMS_KEY)
        return {
            "ALGORITHM_OPERATOR_IDS": {int(e): v for e, v in response[0].items()},
            "CONDITION_MAP": {int(e): v for e, v in response[1].items()},
            "ALGORITHM_MANIFESTS": {int(e): v for e, v in response[2].items()},
        }
