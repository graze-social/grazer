from datetime import datetime, timedelta
import json
import asyncio
import websockets
from app.logger import logger
from app.settings import JETSTREAM_URL


class Jetstream:
    @classmethod
    async def fetch_minute_data(cls, start_us: int, end_us: int):
        """
        Connect at cursor 'start_us'. Yield messages ascending in time.
        The moment we see a message with time_us > end_us, we:
          - Fire off a background task to gracefully close the websocket,
          - break so we can move on to the next slice.
        If we go 2 seconds without receiving any data, we assume
        there's no more data here and move on.
        """
        url = f"{JETSTREAM_URL}&maxMessageSizeBytes=100000&cursor={start_us}"
        logger.info(f"Connecting for slice [{start_us} -> {end_us}]")

        ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)

        try:
            while True:
                try:
                    # If no message arrives within 2 seconds, we're done
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=2)
                except asyncio.TimeoutError:
                    logger.info(
                        f"No more data for slice {start_us}->{end_us} (2s of silence). Moving on."
                    )
                    break
                except websockets.ConnectionClosed:
                    logger.info(f"Connection closed for slice {start_us}->{end_us}")
                    break

                # Skip any unwanted or too-large message
                if len(raw_msg) >= 100000:
                    continue

                try:
                    data = json.loads(raw_msg)

                    msg_time = data.get("time_us", 0)

                    # If this message is newer than our slice's end, close + break.
                    if msg_time > end_us:
                        logger.info(
                            f"Got time_us={msg_time} > end_us={end_us}; closing in background."
                        )
                        asyncio.create_task(Jetstream.graceful_close(ws))
                        break

                    # If it's within this slice, and an actual 'create'
                    if data.get("commit", {}).get("operation") == "create":
                        yield data

                except Exception as exc:
                    logger.error(f"Error parsing message: {exc}")

        finally:
            # If we're exiting for any reason (timeout, break, etc.),
            # ensure the websocket is closed in the background if still open.
            if not ws.closed:
                asyncio.create_task(Jetstream.graceful_close(ws))

    @classmethod
    async def yield_jetstream_reversed(
        cls, end_cursor: int = None, start_cursor: int = None
    ):
        """
        Read data minute by minute (current_end -> current_end - 1 minute),
        forcibly closing each slice as soon as the data is out of range or no data arrives.
        """
        # Example: you could do int(datetime.utcnow().timestamp()*1e6)
        now_us = int((datetime.utcnow() - timedelta(days=0)).timestamp() * 1_000_000)
        if end_cursor is None:
            end_cursor = now_us - 60_000_000  # default: now minus 1 minute
        if start_cursor is None:
            start_cursor = now_us - 24 * 3600 * 1_000_000  # default: now minus 1 day

        if start_cursor >= end_cursor:
            logger.warning("start_cursor >= end_cursor; no data to pull.")
            return

        current_end = end_cursor

        while current_end > start_cursor:
            one_min_ago = current_end - 60_000_000
            time_range_start = max(one_min_ago, start_cursor)
            logger.info(f"Reading slice [{time_range_start} -> {current_end}]")

            # Fetch the data for this 1-minute slice, yielding as soon as we get it
            async for record in Jetstream.fetch_minute_data(
                time_range_start, current_end
            ):
                yield record

            # Move on to the previous minute
            current_end = time_range_start

    @classmethod
    async def graceful_close(cls, ws):
        """
        Try a graceful close; give it 1 second before giving up.
        """
        try:
            await asyncio.wait_for(
                ws.close(code=1000, reason="End of slice"), timeout=1
            )
        except asyncio.TimeoutError:
            logger.warning("Graceful websocket close timed out; ignoring.")
