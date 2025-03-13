import json
from redis import asyncio as aioredis
from app.settings import REDIS_URL, REDIS_DELETE_POST_QUEUE


class RedisClient:
    try:
        REDIS_CLIENT = aioredis.from_url(REDIS_URL, decode_responses=True)
    except:
        REDIS_CLIENT = None

    @classmethod
    async def push_delete_transactions(cls, deletes):
        await cls.REDIS_CLIENT.rpush(REDIS_DELETE_POST_QUEUE, json.dumps(deletes))

    @classmethod
    async def send_pipeline(cls, data, key):
        pipeline = cls.REDIS_CLIENT.pipeline()
        for item in data:
            await pipeline.rpush(key, json.dumps(item))
        await pipeline.execute()

    @classmethod
    async def publish(cls, task_id, message):
        await cls.REDIS_CLIENT.publish(task_id, json.dumps(message))

    @classmethod
    async def blpop(cls, queue_name):
        value = await cls.REDIS_CLIENT.blpop(queue_name)
        return json.loads(value[1])

    @classmethod
    async def get(cls, keyname):
        value = await cls.REDIS_CLIENT.get(keyname)
        if value:
            return json.loads(value)
