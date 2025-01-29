from app.redis import RedisClient
from app.algos.manager import AlgoManager
from app.logger import logger


class RunpodBase:
    @classmethod
    async def publish_status(cls, task_id, message):
        logger.info(f"Publishing message to {task_id} of {message}")
        await RedisClient.publish(task_id, message)

    @classmethod
    async def initialize_algo(cls, dispatcher, manifest, task_id):
        try:
            return await AlgoManager.initialize(
                manifest,
                dispatcher.gpu_embedding_workers,
                dispatcher.gpu_classifier_workers,
                dispatcher.network_workers,
                dispatcher.cache,
            )
        except Exception:
            await cls.publish_status(
                task_id,
                {
                    "error": "Algorithm can't compile - please check your logic, something is preventing it from properly loading."
                },
            )
