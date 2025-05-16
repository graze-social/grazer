import os
import json
import asyncio
from datetime import datetime, timedelta, timezone

import websockets
import aioboto3
from redis import asyncio as aioredis

from app.logger import logger
from app.settings import JETSTREAM_URL as SETTINGS_JETSTREAM_URL
from app.sentry import sentry_sdk


class Jetstream:
    """Consume Bluesky Jetstream and (optionally) forward *app.bsky.feed.post*
    messages to an AWS SQS queue.

    The design implements two completely independent tasks:
    1. A dedicated *producer* task that reads the WebSocket continuously without any
       interruptions and drops messages into an ``asyncio.Queue``.
    2. A completely separate *flusher* task that drains the queue and sends batches
       to SQS based on size/interval thresholds.

    This ensures the websocket reader is never blocked by SQS operations.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    JETSTREAM_URL: str = os.getenv(
        "JETSTREAM_URL",
        f"{SETTINGS_JETSTREAM_URL}&wantedCollections=app.bsky.feed.post"
        if "wantedCollections" not in SETTINGS_JETSTREAM_URL
        else SETTINGS_JETSTREAM_URL,
    )

    # Determine the AWS region from the SQS queue URL
    SQS_QUEUE_URL: str | None = os.getenv("SQS_QUEUE_URL")
    AWS_REGION: str = os.getenv(
        "AWS_REGION", "us-east-1"
    )  # Default to us-east-1 based on the queue URL
    REDIS_URL: str | None = os.getenv("REDIS_URL")

    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 2_000))
    FLUSH_INTERVAL: int = int(os.getenv("FLUSH_INTERVAL", 30))  # seconds
    MAX_SQS_BATCH: int = 10  # AWS hard‑limit per SendMessageBatch
    QUEUE_MAX_SIZE: int = int(
        os.getenv("QUEUE_MAX_SIZE", 10_000)
    )  # Maximum size for the message queue

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk(seq: list[str], size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    @staticmethod
    def _validate_raw_message(raw: str) -> str | None:
        if len(raw) > 100_000:
            return None
        try:
            data = json.loads(raw)
            created_at = data.get("commit", {}).get("record", {}).get("createdAt")
            if created_at:
                for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
                    try:
                        ts = datetime.strptime(created_at, fmt).replace(
                            tzinfo=timezone.utc
                        )
                        break
                    except ValueError:
                        ts = None
                if not ts or ts > datetime.utcnow().replace(tzinfo=timezone.utc):
                    data["commit"]["record"]["createdAt"] = datetime.utcnow().strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    )
                    return json.dumps(data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("validation error: %s", exc)
            return None
        return raw

    # ------------------------------------------------------------------
    # SQS
    # ------------------------------------------------------------------
    SQS_CLIENT = None

    @classmethod
    async def _init_sqs_client(cls):
        """Initialize SQS client using the __aenter__ approach."""
        if cls.SQS_CLIENT is None:
            try:
                logger.info("Initializing SQS client in region %s", cls.AWS_REGION)
                session = aioboto3.Session()
                cls.SQS_CLIENT = await session.client(
                    "sqs", region_name=cls.AWS_REGION
                ).__aenter__()

                # Verify the queue exists
                logger.info("Verifying SQS queue exists: %s", cls.SQS_QUEUE_URL)
                try:
                    # Get queue attributes as a simple way to check if queue exists
                    await cls.SQS_CLIENT.get_queue_attributes(
                        QueueUrl=cls.SQS_QUEUE_URL, AttributeNames=["QueueArn"]
                    )
                    logger.info("SQS queue verified successfully")
                except Exception as e:
                    logger.error("SQS queue verification failed: %s", e)
                    logger.error(
                        "Please check if the queue URL is correct and the queue exists"
                    )
                    # Reset client so we don't keep using a bad configuration
                    await cls.SQS_CLIENT.close()
                    cls.SQS_CLIENT = None
                    raise
            except Exception as e:
                logger.error("Failed to initialize SQS client: %s", e)
                sentry_sdk.capture_exception(e)
                raise

    @classmethod
    async def _send_batch_to_sqs(cls, batch: list[str]):
        """Send a batch to SQS using an *aioboto3* session.

        Uses the __aenter__ approach to create and manage the SQS client.
        Completely isolated from the websocket operations.
        """
        if not batch:
            return

        try:
            # Ensure SQS client is initialized
            if cls.SQS_CLIENT is None:
                await cls._init_sqs_client()

            # Additional safety check
            if cls.SQS_CLIENT is None:
                logger.error("Cannot send to SQS: client initialization failed")
                return

            for chunk in cls._chunk(batch, cls.MAX_SQS_BATCH):
                entries = [
                    {"Id": str(i), "MessageBody": body} for i, body in enumerate(chunk)
                ]
                try:
                    logger.debug("Sending batch of %d messages to SQS", len(entries))
                    resp = await cls.SQS_CLIENT.send_message_batch(
                        QueueUrl=cls.SQS_QUEUE_URL, Entries=entries
                    )
                    if failed := resp.get("Failed"):
                        logger.error("%d SQS failures: %s", len(failed), failed)
                    else:
                        logger.debug(
                            "Successfully sent %d messages to SQS", len(entries)
                        )
                except Exception as e:
                    logger.error("SQS send_message_batch error: %s", e)
                    sentry_sdk.capture_exception(e)

                    # If we get a NonExistentQueue error, try to reinitialize the client
                    if "NonExistentQueue" in str(e):
                        logger.info(
                            "Attempting to reinitialize SQS client due to NonExistentQueue error"
                        )
                        if cls.SQS_CLIENT:
                            await cls.SQS_CLIENT.close()
                            cls.SQS_CLIENT = None
                        # Don't reinitialize immediately - let the next batch trigger it
                        break
        except Exception as e:
            logger.error("Error in send_batch_to_sqs: %s", e)
            sentry_sdk.capture_exception(e)

    # ------------------------------------------------------------------
    # Producer / flusher tasks
    # ------------------------------------------------------------------
    @classmethod
    async def _producer(
        cls, queue: asyncio.Queue, redis, shutdown_event: asyncio.Event
    ):
        """Continuously read websocket and enqueue validated messages.

        This task runs independently and is never blocked by SQS operations.
        If the queue fills up, we'll log a warning but continue processing the websocket
        (potentially dropping messages if the queue remains full).

        For SQS streaming, we use:
        1. The stored cursor from Redis if available
        2. Current time if no Redis cursor is found (start from now)
        """
        # For SQS streaming, use current time as default or stored cursor from Redis
        cursor = int(
            datetime.utcnow().timestamp() * 1_000_000
        )  # Current time in microseconds
        if redis and (stored := await redis.get("jetstream:last_cursor")):
            try:
                cursor = int(stored)
                logger.info(
                    "Retrieved cursor from Redis: %s (timestamp: %s)",
                    cursor,
                    datetime.fromtimestamp(cursor / 1_000_000, tz=timezone.utc),
                )
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse Redis cursor %r: %s", stored, e)

        ws_url = f"{cls.JETSTREAM_URL}&maxMessageSizeBytes=100000&cursor={cursor}"
        logger.info("Producer connecting with cursor for current time: %s", ws_url)

        try:
            async with websockets.connect(
                ws_url, ping_interval=20, ping_timeout=10
            ) as ws:
                while not shutdown_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        if (msg := cls._validate_raw_message(raw)) is not None:
                            # Try to put in queue with short timeout to avoid blocking
                            try:
                                await asyncio.wait_for(queue.put(msg), timeout=0.1)
                            except asyncio.TimeoutError:
                                # Queue is full, log warning but keep websocket reading
                                if not getattr(cls, "_queue_full_warned", False):
                                    logger.warning(
                                        "Message queue full! Some messages may be dropped."
                                    )
                                    cls._queue_full_warned = True
                                continue

                            # Reset warning flag if we successfully put a message
                            if getattr(cls, "_queue_full_warned", False):
                                cls._queue_full_warned = False
                    except asyncio.TimeoutError:
                        # Just a timeout on websocket read, continue
                        continue
                    except websockets.ConnectionClosed:
                        logger.warning("Websocket connection closed, reconnecting...")
                        break
                    except Exception as e:
                        logger.error("Error in producer task: %s", e)
                        sentry_sdk.capture_exception(e)
                        await asyncio.sleep(1)  # Brief pause before continuing
        except Exception as e:
            logger.error("Fatal error in producer: %s", e)
            sentry_sdk.capture_exception(e)
            # Signal flusher to also shut down
            shutdown_event.set()

    @classmethod
    async def _flusher(cls, queue: asyncio.Queue, redis, shutdown_event: asyncio.Event):
        """Drain *queue*; flush to SQS when size or time threshold met.

        This task runs completely independently from the websocket reader.
        """
        batch: list[str] = []
        last_flush = datetime.utcnow()

        while not shutdown_event.is_set():
            try:
                # Calculate remaining time until next scheduled flush
                timeout = (
                    cls.FLUSH_INTERVAL
                    - (datetime.utcnow() - last_flush).total_seconds()
                )
                timeout = max(
                    timeout, 0.1
                )  # Small minimum timeout to check shutdown_event

                try:
                    # Get message with timeout to allow periodic flushing
                    msg = await asyncio.wait_for(queue.get(), timeout=timeout)
                    batch.append(msg)
                    queue.task_done()
                except asyncio.TimeoutError:
                    # No new message, check if we should flush based on time
                    pass

                # Check if we should flush based on batch size or time elapsed
                should_flush = (
                    len(batch) >= cls.BATCH_SIZE
                    or (datetime.utcnow() - last_flush).total_seconds()
                    >= cls.FLUSH_INTERVAL
                )

                if should_flush and batch:
                    logger.info("Flushing %d records", len(batch))
                    # Send batch to SQS in a separate task to avoid blocking this loop
                    # This ensures we can keep processing the queue even if SQS has issues
                    asyncio.create_task(cls._send_batch_to_sqs(batch.copy()))

                    if redis:
                        try:
                            last_cursor = json.loads(batch[-1])["time_us"] - 1
                            await redis.set("jetstream:last_cursor", last_cursor)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to store cursor: %s", exc)

                    # Clear batch and reset timer
                    batch = []
                    last_flush = datetime.utcnow()

            except Exception as e:
                logger.error("Error in flusher task: %s", e)
                sentry_sdk.capture_exception(e)
                await asyncio.sleep(1)  # Brief pause to avoid tight error loops

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    async def stream_to_sqs(cls):
        if not cls.SQS_QUEUE_URL:
            raise RuntimeError("SQS_QUEUE_URL env‑var required to stream to SQS")

        # Log important configuration for debugging
        logger.info("Starting Jetstream with configuration:")
        logger.info("  AWS_REGION: %s", cls.AWS_REGION)
        logger.info("  SQS_QUEUE_URL: %s", cls.SQS_QUEUE_URL)
        logger.info("  BATCH_SIZE: %d", cls.BATCH_SIZE)
        logger.info("  FLUSH_INTERVAL: %d seconds", cls.FLUSH_INTERVAL)
        logger.info("  QUEUE_MAX_SIZE: %d", cls.QUEUE_MAX_SIZE)

        # Try to initialize SQS client, but continue even if it fails
        # (we'll retry in the _send_batch_to_sqs method)
        try:
            await cls._init_sqs_client()
            if cls.SQS_CLIENT is not None:
                logger.info("SQS client initialized successfully")
            else:
                logger.warning(
                    "Initial SQS client initialization failed, will retry later"
                )
        except Exception as e:
            logger.warning("Failed to initialize SQS client on startup: %s", e)
            logger.info("Will continue and retry SQS initialization later")

        redis = None
        if cls.REDIS_URL:
            try:
                redis = aioredis.from_url(cls.REDIS_URL, decode_responses=True)
                logger.info("Redis client initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize Redis client: %s", e)
                sentry_sdk.capture_exception(e)

        try:
            while True:  # outer reconnect loop
                # Create a bounded queue to prevent memory issues
                queue: asyncio.Queue = asyncio.Queue(maxsize=cls.QUEUE_MAX_SIZE)
                shutdown_event = asyncio.Event()

                try:
                    logger.info("Starting producer and flusher tasks")
                    # Create both tasks and run them concurrently, but independently
                    producer = asyncio.create_task(
                        cls._producer(queue, redis, shutdown_event)
                    )
                    flusher = asyncio.create_task(
                        cls._flusher(queue, redis, shutdown_event)
                    )

                    # Wait for either task to complete
                    done, pending = await asyncio.wait(
                        [producer, flusher], return_when=asyncio.FIRST_COMPLETED
                    )

                    # Signal shutdown and cancel remaining tasks
                    shutdown_event.set()
                    logger.info(
                        "One task completed, signaling shutdown to remaining tasks"
                    )
                    for task in pending:
                        task.cancel()

                    # Wait for remaining tasks to clean up (with timeout)
                    await asyncio.wait(pending, timeout=5)

                    # Check for exceptions in completed tasks
                    for task in done:
                        try:
                            await task
                        except Exception as e:
                            logger.error("Task failed with: %s", e)
                            sentry_sdk.capture_exception(e)

                except KeyboardInterrupt:
                    logger.info("Received KeyboardInterrupt, shutting down...")
                    shutdown_event.set()
                    break

                except Exception as exc:  # noqa: BLE001
                    logger.warning("stream_to_sqs error (%s); reconnecting in 5 s", exc)
                    sentry_sdk.capture_exception(exc)
                    await asyncio.sleep(5)

                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

        finally:
            # Properly close the SQS client when we're done
            if cls.SQS_CLIENT is not None:
                try:
                    logger.info("Closing SQS client")
                    await cls.SQS_CLIENT.close()
                    cls.SQS_CLIENT = None
                except Exception as e:
                    logger.error("Error closing SQS client: %s", e)
                    sentry_sdk.capture_exception(e)

            # Close Redis connection
            if redis is not None:
                try:
                    logger.info("Closing Redis connection")
                    await redis.close()
                except Exception as e:
                    logger.error("Error closing Redis connection: %s", e)
                    sentry_sdk.capture_exception(e)

    # ------------------------------------------------------------------
    # Slice‑based helpers (unchanged)
    # ------------------------------------------------------------------
    @classmethod
    async def graceful_close(cls, ws):
        try:
            await asyncio.wait_for(
                ws.close(code=1000, reason="End of slice"), timeout=1
            )
        except asyncio.TimeoutError:
            logger.warning("Graceful websocket close timed out; ignoring.")

    @classmethod
    async def fetch_minute_data(cls, start_us: int, end_us: int):
        url = f"{cls.JETSTREAM_URL}&maxMessageSizeBytes=100000&cursor={start_us}"
        logger.info("Connecting for slice [%s -> %s]", start_us, end_us)
        ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        try:
            while True:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=2)
                except asyncio.TimeoutError:
                    logger.info(
                        "No more data for slice %s->%s (2s silence).", start_us, end_us
                    )
                    break
                except websockets.ConnectionClosed:
                    logger.info("Connection closed for slice %s->%s", start_us, end_us)
                    break
                if len(raw_msg) >= 100000:
                    continue
                try:
                    data = json.loads(raw_msg)
                    if (msg_time := data.get("time_us", 0)) > end_us:
                        asyncio.create_task(cls.graceful_close(ws))
                        break
                    if data.get("commit", {}).get("operation") == "create":
                        yield data
                except Exception as exc:
                    logger.error("Error parsing message: %s", exc)
        finally:
            if not ws.closed:
                asyncio.create_task(cls.graceful_close(ws))

    @classmethod
    async def yield_jetstream_reversed(
        cls, end_cursor: int | None = None, start_cursor: int | None = None
    ):
        now_us = int(datetime.utcnow().timestamp() * 1_000_000)
        end_cursor = end_cursor or now_us - 60_000_000
        start_cursor = start_cursor or now_us - 24 * 3_600 * 1_000_000
        if start_cursor >= end_cursor:
            logger.warning("start_cursor >= end_cursor; no data to pull.")
            return
        current_end = end_cursor
        while current_end > start_cursor:
            one_min_ago = current_end - 60_000_000
            time_range_start = max(one_min_ago, start_cursor)
            utc_time = datetime.fromtimestamp(
                time_range_start / 1_000_000, tz=timezone.utc
            )
            logger.info("Reading slice from %s", utc_time)
            async for record in cls.fetch_minute_data(time_range_start, current_end):
                yield record
            current_end = time_range_start
