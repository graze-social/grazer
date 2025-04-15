import os
import json
import asyncio
from datetime import datetime, timedelta, timezone

import websockets
import aioboto3
from redis import asyncio as aioredis

from app.logger import logger
from app.settings import JETSTREAM_URL as SETTINGS_JETSTREAM_URL


class Jetstream:
    """Utility class for consuming Bluesky Jetstream and, optionally, forwarding
    *app.bsky.feed.post* messages to an AWS SQS queue.

    Existing slice‑based back‑fill helpers are preserved; new methods add a
    real‑time *WebSocket → SQS* forwarder.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    JETSTREAM_URL: str = os.getenv(
        "JETSTREAM_URL",
        f"{SETTINGS_JETSTREAM_URL}&wantedCollections=app.bsky.feed.post"  # falls back to settings
        if "wantedCollections" not in SETTINGS_JETSTREAM_URL
        else SETTINGS_JETSTREAM_URL,
    )
    AWS_REGION: str = os.getenv("AWS_REGION", "us-west-2")
    SQS_QUEUE_URL: str | None = os.getenv("SQS_QUEUE_URL")
    REDIS_URL: str | None = os.getenv("REDIS_URL")

    # batching / flow‑control
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 2_000))
    FLUSH_INTERVAL: int = int(os.getenv("FLUSH_INTERVAL", 30))  # seconds
    MAX_SQS_BATCH: int = 10  # AWS hard‑limit

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk(seq: list[str], size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    @staticmethod
    def _validate_raw_message(raw: str) -> str | None:
        """Skip spam / oversize messages and normalise future timestamps."""
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
    # SQS helpers
    # ------------------------------------------------------------------
    @classmethod
    async def _send_batch_to_sqs(cls, messages: list[str]):
        async with aioboto3.client("sqs", region_name=cls.AWS_REGION) as sqs:
            for chunk in cls._chunk(messages, cls.MAX_SQS_BATCH):
                entries = [{"Id": str(i), "MessageBody": body} for i, body in enumerate(chunk)]
                resp = await sqs.send_message_batch(QueueUrl=cls.SQS_QUEUE_URL, Entries=entries)
                if failed := resp.get("Failed"):
                    logger.error("%d SQS failures: %s", len(failed), failed)

    # ------------------------------------------------------------------
    # Real‑time forwarder internals
    # ------------------------------------------------------------------
    @classmethod
    async def _flush(cls, batch: list[str], redis):
        if not batch:
            return
        logger.info("Flushing %d records to SQS", len(batch))
        await asyncio.gather(*(cls._send_batch_to_sqs(c) for c in cls._chunk(batch, cls.MAX_SQS_BATCH)))
        if redis:
            try:
                last_cursor = json.loads(batch[-1])["time_us"] - 1
                await redis.set("jetstream:last_cursor", last_cursor)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to store cursor: %s", exc)

    @classmethod
    async def _consume_once(cls):
        if not cls.SQS_QUEUE_URL:
            raise RuntimeError("SQS_QUEUE_URL env‑var required to stream to SQS")

        redis = aioredis.from_url(cls.REDIS_URL, decode_responses=True) if cls.REDIS_URL else None
        cursor = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1_000_000)
        if redis:
            stored = await redis.get("jetstream:last_cursor")
            if stored:
                cursor = int(stored)

        ws_url = f"{cls.JETSTREAM_URL}&maxMessageSizeBytes=100000&cursor={cursor}"
        logger.info("Connecting to Jetstream %s", ws_url)

        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            batch: list[str] = []
            last_flush = datetime.utcnow()
            async for raw in ws:
                if (msg := cls._validate_raw_message(raw)) is None:
                    continue
                batch.append(msg)
                now = datetime.utcnow()
                if len(batch) >= cls.BATCH_SIZE or (now - last_flush).total_seconds() >= cls.FLUSH_INTERVAL:
                    await cls._flush(batch, redis)
                    batch = []
                    last_flush = now

    # ------------------------------------------------------------------
    # Public API for real‑time streaming to SQS
    # ------------------------------------------------------------------
    @classmethod
    async def stream_to_sqs(cls):
        """Run forever with automatic reconnect, forwarding to SQS."""
        while True:
            try:
                await cls._consume_once()
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Jetstream error (%s). Reconnecting in 5 s", exc)
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Existing slice‑based back‑fill helpers (unchanged)
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
        now_us = int((datetime.utcnow()).timestamp() * 1_000_000)
        end_cursor = end_cursor or now_us - 60_000_000
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

