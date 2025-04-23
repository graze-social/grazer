import ray
import asyncio
import aioboto3
import json
from typing import Any
from app.logger import logger
from app.sentry import sentry_sdk
from app.kube.router import KubeRouter
from app.settings import StreamerSettings

settings = StreamerSettings()

@ray.remote(num_cpus=0.5)
class SQSConsumer:
    """Consume messages from an AWS SQS queue, parse JSON, and forward them to KubeRouter."""

    def __init__(self):
        self.queue_url = settings.sqs_queue_url
        self.aws_region = settings.aws_region
        self.polling_interval = settings.sqs_polling_interval
        self.session = aioboto3.Session()
        self.shutdown_event = asyncio.Event()

    async def receive_messages(self):
        """Continuously poll SQS for messages."""
        async with self.session.client("sqs", region_name=self.aws_region) as sqs:
            while not self.shutdown_event.is_set():
                try:
                    response = await sqs.receive_message(
                        QueueUrl=self.queue_url,
                        MaxNumberOfMessages=10,
                        WaitTimeSeconds=10,
                    )

                    messages = response.get("Messages", [])
                    if not messages:
                        continue

                    tasks = [self.process_message(sqs, msg) for msg in messages]
                    await asyncio.gather(*tasks)

                except Exception as e:
                    logger.error("Failed to receive messages from SQS: %s", e)
                    sentry_sdk.capture_exception(e)
                    await asyncio.sleep(self.polling_interval)

    async def process_message(self, sqs: Any, message: dict[str, Any]):
        """Parse JSON message and send it to KubeRouter."""
        receipt_handle = message["ReceiptHandle"]
        body = message.get("Body", "")

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error("JSON decoding error: %s", e)
            sentry_sdk.capture_exception(e)
            await self.delete_message(sqs, receipt_handle)
            return
        try:
            await KubeRouter.process_request(data, {}, settings.noop)
            await self.delete_message(sqs, receipt_handle)
        except Exception as e:
            logger.error("Failed to process message: %s", e)
            sentry_sdk.capture_exception(e)

    async def delete_message(self, sqs: Any, receipt_handle: str):
        """Delete message from the queue after successful processing."""
        try:
            await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
            logger.debug("Message deleted from SQS")
        except Exception as e:
            logger.error("Failed to delete message: %s", e)
            sentry_sdk.capture_exception(e)

    async def run(self):
        """Run the SQS consumer indefinitely."""
        logger.info("Starting SQS consumer for queue: %s", self.queue_url)
        try:
            await self.receive_messages()
        finally:
            self.shutdown_event.set()

    def stop(self):
        """Signal the consumer to shut down."""
        self.shutdown_event.set()
