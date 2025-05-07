import asyncio
import ray
from app.sqs_consumer import SQSConsumer


async def run_consumer():
    consumer = SQSConsumer.options(max_concurrency=10, lifetime="detached").remote()
    ray.get([consumer.run.remote()])  # type: ignore


asyncio.run(run_consumer())
