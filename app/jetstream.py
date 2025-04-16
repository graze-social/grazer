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
    messages to an AWS SQS queue.

    **Design refresh (2025‑04‑15)** — The forwarder now mirrors the original
    RunPod design: a dedicated *producer* task reads the WebSocket nonstop and
    drops messages into an ``asyncio.Queue``; an independent *flusher* task
    drains that queue and sends batches to SQS based on size/interval. This
    prevents back‑pressure on the websocket reader.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    JETSTREAM_URL: str = os.getenv(
        "JETSTREAM_URL",
        f"{SETTINGS_JETSTREAM_URL}&wantedCollections=app.bsky.feed.post"
        if "wantedCollections" not in SETTINGS_JETSTREAM_URL else SETTINGS_JETSTREAM_URL,
    )
    AWS_REGION: str = os.getenv("AWS_REGION", "us-west-2")
    SQS_QUEUE_URL: str | None = os.getenv("SQS_QUEUE_URL")
    REDIS_URL: str | None = os.getenv("REDIS_URL")

    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 2_000))
    FLUSH_INTERVAL: int = int(os.getenv("FLUSH_INTERVAL", 30))  # seconds
    MAX_SQS_BATCH: int = 10  # AWS hard‑limit per SendMessageBatch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk(seq: list[str], size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    @staticmethod
    def _validate_raw_message(raw: str) -> str | None:
        if "leiarcaica" in raw.lower() or len(raw) > 100_000:
            return None
        try:
            data = json.loads(raw)
            created_at = data.get("commit", {}).get("record", {}).get("createdAt")
            if created_at:
                for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
                    try:
                        ts = datetime.strptime(created_at, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        ts = None
                if not ts or ts > datetime.utcnow().replace(tzinfo=timezone.utc):
                    data["commit"]["record"]["createdAt"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                    return json.dumps(data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("validation error: %s", exc)
            return None
        return raw

    # ------------------------------------------------------------------
    # SQS
    # ------------------------------------------------------------------
    @classmethod
    async def _send_batch_to_sqs(cls, batch: list[str]):
        """Send a batch to SQS using an *aioboto3* session.

        Some Alpine / slim images ship an old aioboto3 build where the module‑level
        helper ``aioboto3.client`` is missing.  We therefore instantiate an
        explicit session (works on every version).
        """
        if not batch:
            return
        session = aioboto3.Session()
        async with session.client("sqs", region_name=cls.AWS_REGION) as sqs:
            for chunk in cls._chunk(batch, cls.MAX_SQS_BATCH):
                entries = [{"Id": str(i), "MessageBody": body} for i, body in enumerate(chunk)]
                try:
                    resp = await sqs.send_message_batch(QueueUrl=cls.SQS_QUEUE_URL, Entries=entries)
                except Exception as e:
                    import code;code.interact(local=dict(globals(), **locals())) 
                if failed := resp.get("Failed"):
                    logger.error("%d SQS failures: %s", len(failed), failed)

    # ------------------------------------------------------------------
    # Producer / flusher pair
    @classmethod
    async def _producer(cls, queue: asyncio.Queue, redis):
        """Continuously read websocket and enqueue validated messages."""
        cursor = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1_000_000)
        if redis and (stored := await redis.get("jetstream:last_cursor")):
            cursor = int(stored)

        ws_url = f"{cls.JETSTREAM_URL}&maxMessageSizeBytes=100000&cursor={cursor}"
        logger.info("Producer connecting %s", ws_url)

        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            async for raw in ws:
                if (msg := cls._validate_raw_message(raw)) is not None:
                    await queue.put(msg)

    @classmethod
    async def _flusher(cls, queue: asyncio.Queue, redis):
        """Drain *queue*; flush to SQS when size or time threshold met."""
        batch: list[str] = []
        last_flush = datetime.utcnow()

        while True:
            timeout = cls.FLUSH_INTERVAL - (datetime.utcnow() - last_flush).total_seconds()
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=max(timeout, 0.0))
                batch.append(msg)
            except asyncio.TimeoutError:
                pass  # interval elapsed

            should_flush = (
                len(batch) >= cls.BATCH_SIZE or
                (datetime.utcnow() - last_flush).total_seconds() >= cls.FLUSH_INTERVAL
            )
            if should_flush and batch:
                logger.info("Flushing %d records", len(batch))
                await cls._send_batch_to_sqs(batch)
                if redis:
                    try:
                        last_cursor = json.loads(batch[-1])["time_us"] - 1
                        await redis.set("jetstream:last_cursor", last_cursor)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to store cursor: %s", exc)
                batch = []
                last_flush = datetime.utcnow()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    async def stream_to_sqs(cls):
        if not cls.SQS_QUEUE_URL:
            raise RuntimeError("SQS_QUEUE_URL env‑var required to stream to SQS")

        redis = aioredis.from_url(cls.REDIS_URL, decode_responses=True) if cls.REDIS_URL else None

        while True:  # outer reconnect loop
            queue: asyncio.Queue = asyncio.Queue(maxsize=cls.BATCH_SIZE * 3)
            try:
                producer = asyncio.create_task(cls._producer(queue, redis))
                flusher  = asyncio.create_task(cls._flusher(queue, redis))
                await asyncio.gather(producer, flusher)
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("stream_to_sqs error (%s); reconnecting in 5 s", exc)
                sentry_sdk.capture_exception(exc)
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Slice‑based helpers (unchanged)
    # ------------------------------------------------------------------
    @classmethod
    async def graceful_close(cls, ws):
        try:
            await asyncio.wait_for(ws.close(code=1000, reason="End of slice"), timeout=1)
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
                    logger.info("No more data for slice %s->%s (2s silence).", start_us, end_us)
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
    async def yield_jetstream_reversed(cls, end_cursor: int | None = None, start_cursor: int | None = None):
        now_us = int(datetime.utcnow().timestamp() * 1_000_000)
        end_cursor   = end_cursor   or now_us - 60_000_000
        start_cursor = start_cursor or now_us - 24 * 3_600 * 1_000_000
        if start_cursor >= end_cursor:
            logger.warning("start_cursor >= end_cursor; no data to pull.")
            return
        current_end = end_cursor
        while current_end > start_cursor:
            one_min_ago = current_end - 60_000_000
            time_range_start = max(one_min_ago, start_cursor)
            utc_time = datetime.fromtimestamp(time_range_start / 1_000_000, tz=timezone.utc)
            logger.info("Reading slice from %s", utc_time)
            async for record in cls.fetch_minute_data(time_range_start, current_end):
                yield record
            current_end = time_range_start
