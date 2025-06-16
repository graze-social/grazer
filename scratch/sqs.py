import time
import json
from aioboto3 import Session
import asyncio
from types_aiobotocore_sqs.client import SQSClient

QUEUE = "https://sqs.us-east-1.amazonaws.com/715841359524/sky-feeder"
SQS = None
BATCH_FILE = "./scratch/batch2.json"




async def get_queue_length(sqs: SQSClient) -> int:
    qattrs = await sqs.get_queue_attributes(
        QueueUrl=QUEUE,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesDelayed",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    )
    total = 0
    for k, v in qattrs["Attributes"].items():
        print(v)
        total += int(v)
    return total


async def push_batch():
    global SQS
    session = Session()

    sqs = await session.client("sqs").__aenter__()
    sqs: SQSClient
    # purge the queue
    # PurgeQueueInProgress: An error occurred (AWS.SimpleQueueService.PurgeQueueInProgress) when calling the PurgeQueue operation: Only one PurgeQueue operation on sky-feeder is allowed every 60 seconds.
    # await sqs.purge_queue(QueueUrl=QUEUE)
    messages_count = 0
    with open(BATCH_FILE) as file:
        messages = json.loads(file.read())
        for msg in messages:
            await sqs.send_message(QueueUrl=QUEUE, MessageBody=json.dumps(msg["Body"]))
            messages_count += 1
            print(f"re-enqueued message: {msg['MessageId']}")
            await asyncio.sleep(0.1)
    await asyncio.sleep(5)
    start_time = time.time()
    sleep_tokens = 0
    while True:
        queue_length = await get_queue_length(sqs)
        if queue_length > 0:
            print("queue is active")
            print(f"current size: {queue_length} pushed messages: {messages_count}")
            sleep_tokens += 5
            await asyncio.sleep(10)
        else:
            break

    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time}")
    print(f"total sleep tokens: {sleep_tokens}")
    print(f"total elapsed time (seconds): {elapsed_time - sleep_tokens}")

    await sqs.close()


if __name__ == "__main__":
    asyncio.run(push_batch())
