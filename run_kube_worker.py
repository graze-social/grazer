import asyncio
from app.sqs_consumer import SQSConsumer
consumer = SQSConsumer()
asyncio.run(consumer.receive_messages())