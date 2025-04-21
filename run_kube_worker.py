import asyncio
import ray
from app.sqs_consumer import SQSConsumer


consumer = SQSConsumer.options(max_concurrency=5)

async def run_consumer():
    konsumer = ray.get(consumer.remote())
    await konsumer.run()

asyncio.run(run_consumer())
