import ray
import asyncio
import aioboto3
import json
from app.logger import logger
from app.sentry import sentry_sdk
from app.kube.router import KubeRouter
from app.settings import StreamerSettings
from app.ray.dispatcher import Dispatcher
from typing import Any
import traceback

# wrapper for strutured bsky data
# TODO: use bsky
from app.stream_data import StreamData

settings = StreamerSettings()


@ray.remote(num_cpus=0.5, max_task_retries=-1, max_restarts=-1)
class SQSConsumer:
    """Consume messages from an AWS SQS queue, parse JSON, and forward them to KubeRouter."""

    def __init__(self):
        self.queue_url = settings.sqs_queue_url
        self.aws_region = settings.aws_region
        self.polling_interval = settings.sqs_polling_interval
        self.session = aioboto3.Session()
        self.shutdown_event = asyncio.Event()
        self.dispatcher = Dispatcher()
        # debug
        self.gathered_tasks: int = 0

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
                    logger.warn(f"[warn debug] messages {len(messages)}")

                    tasks = [self.process_message(sqs, msg) for msg in messages]
                    self.gathered_tasks += len(tasks)
                    logger.warn("gathered tasks ")
                    await asyncio.gather(*tasks)

                except Exception as e:
                    logger.error("Failed to receive messages from SQS: %s", e)
                    sentry_sdk.capture_exception(e)
                    await asyncio.sleep(self.polling_interval)

    async def conform_message_data(
        self, sqs: Any, message: dict[str, Any]
    ) -> StreamData:
        receipt_handle = message["ReceiptHandle"]
        body = message.get("Body", "")
        try:
            body_data = json.loads(body)
            return StreamData(body_data)
        except json.JSONDecodeError as e:
            logger.error("JSON decoding error: %s", e)
            sentry_sdk.capture_exception(e)
            await self.delete_message(sqs, receipt_handle)
            return StreamData({})

    async def process_message(self, sqs: Any, message: dict[str, Any]):
        """Parse JSON message and send it to KubeRouter."""
        receipt_handle = message["ReceiptHandle"]
        body = message.get("Body", "")

        try:
            body_data = json.loads(body)
            stream_data = StreamData(body_data)
        except json.JSONDecodeError as e:
            logger.error("JSON decoding error: %s", e)
            sentry_sdk.capture_exception(e)
            await self.delete_message(sqs, receipt_handle)
            return
        try:
            await KubeRouter.process_request(
                self.dispatcher, stream_data, settings.noop
            )
            logger.warn(
                f"[warn debug] {len(stream_data.transactions())} messages processed, deleting messages"
            )

            logger.warn(f"[warn debug] deleting receipt: {receipt_handle}")
            await self.delete_message(sqs, receipt_handle)
        except Exception as e:
            logger.error("Failed to process message: %s", e)
            sentry_sdk.capture_exception(e)
            if settings.stream_debug:
                logger.error(traceback.print_exc())

    async def delete_message(self, sqs: Any, receipt_handle: str):
        """Delete message from the queue after successful processing."""
        try:
            await sqs.delete_message(
                QueueUrl=self.queue_url, ReceiptHandle=receipt_handle
            )
            logger.info(f"Message deleted from SQS {receipt_handle}")
        except Exception as e:
            logger.error("Failed to delete message: %s", e)
            sentry_sdk.capture_exception(e)

    def stop(self):
        """Signal the consumer to shut down."""
        self.shutdown_event.set()

    # this is a debug method for checking the heartbeat of the ray actor
    def num_gathered_tasks(self):
        logger.info(f"current tasks {self.gathered_tasks}")
        return self.gathered_tasks
