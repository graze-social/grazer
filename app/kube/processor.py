import random
import asyncio

from app.ray.dispatcher import Dispatcher
from app.logic_evaluator import LogicEvaluator
from app.logger import logger
from app.kube.base import KubeBase
from app.helpers import chunk
from app.redis import RedisClient
from app.settings import CURRENT_ALGORITHMS_KEY, EgressSettings
from app.timings import record_timing


has_egress = EgressSettings().egress_enabled


class KubeProcessor(KubeBase):
    @classmethod
    @record_timing(fn_prefix="KubeProcessor")
    async def ingest_feed(cls, transactions):
        records = []
        deletes = []

        for transaction in transactions:
            if transaction.get("commit", {}).get("operation") == "create":
                records.append(transaction)
            elif transaction.get("commit", {}).get("operation") == "delete":
                deletes.append(transaction)
        if deletes:
            print(f"Delete length is {len(deletes)}")
            await RedisClient.push_delete_transactions(deletes)
        return records

    @classmethod
    @record_timing(fn_prefix="KubeProcessor")
    async def process_algos(cls, dispatcher, transactions):
        records = await cls.ingest_feed(transactions)
        algo_data = await KubeProcessor.get_algorithm_operators()
        await KubeProcessor.run_algos(
            dispatcher,
            records,
            algo_data["ALGORITHM_MANIFESTS"],
            algo_data["CONDITION_MAP"],
        )

    @classmethod
    @record_timing(fn_prefix="KubeProcessor")
    async def run_algos(cls, dispatcher: Dispatcher, records, manifests, all_operators):
        logger.warn(
            f"[warn debug] running records: {len(records)} manifests: {len(manifests.items())}"
        )
        manifests = list(manifests.items())
        random.shuffle(manifests)
        # await run_precache(dispatcher, records, [{}], all_operators)
        for manifest_chunk in chunk(manifests, 1000):
            hydrated_manifests = []
            algorithm_ids = []
            for algorithm_id, manifest in manifest_chunk:
                if manifest:
                    hydrated_manifests.append(
                        LogicEvaluator.rehydrate_single_manifest(
                            manifest, all_operators
                        )
                    )
                    algorithm_ids.append(algorithm_id)
            manifest_data = list(zip(algorithm_ids, hydrated_manifests))
            await dispatcher.distribute_tasks(records, manifest_data, has_egress)
            logger.warn("[warn debug] All tasks distributed ")
            logger.warn(f"[warn debug] Task count: {len(asyncio.all_tasks())}")

            # [THEORY] 2. why this number, 100?
            while len(asyncio.all_tasks()) > 100:
                logger.info(f"Current Task Depth is {len(asyncio.all_tasks())}")
                asyncio.sleep(1)

        logger.warn("[warn debug] run algos function exiting")

    @classmethod
    async def get_algorithm_operators(cls):
        response = await RedisClient.get(CURRENT_ALGORITHMS_KEY)
        return {
            "ALGORITHM_OPERATOR_IDS": {int(e): v for e, v in response[0].items()},
            "CONDITION_MAP": {int(e): v for e, v in response[1].items()},
            "ALGORITHM_MANIFESTS": {int(e): v for e, v in response[2].items()},
        }
