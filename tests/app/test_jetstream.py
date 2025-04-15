import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import websockets
from websockets.frames import Close

from app.jetstream import Jetstream   # adjust if the import path differs


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_websocket():
    """Return a fully‑mocked websocket connection."""
    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws.closed = False

    async def _connect(*_a, **_kw):
        return mock_ws

    with patch("websockets.connect", new=_connect):
        yield mock_ws


# ---------------------------------------------------------------------------
# Original slice‑based tests (unchanged)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_minute_data_success(mock_websocket):
    start_us = 1_700_000_000_000_000
    end_us   = start_us + 5_000_000

    good = json.dumps({"time_us": start_us + 1_000_000,
                       "commit": {"operation": "create"}})

    mock_websocket.recv.side_effect = [
        good,
        websockets.exceptions.ConnectionClosed(Close(1000, "End"), None)
    ]

    records = [r async for r in Jetstream.fetch_minute_data(start_us, end_us)]
    assert records and records[0]["commit"]["operation"] == "create"


@pytest.mark.asyncio
async def test_fetch_minute_data_timeout(mock_websocket):
    start_us = 1_700_000_000_000_000
    end_us   = start_us + 5_000_000

    mock_websocket.recv.side_effect = asyncio.TimeoutError()
    assert [r async for r in Jetstream.fetch_minute_data(start_us, end_us)] == []


@pytest.mark.asyncio
async def test_fetch_minute_data_large_message(mock_websocket):
    start_us = 1_700_000_000_000_000
    end_us   = start_us + 5_000_000

    big   = "X" * 100_001
    good  = json.dumps({"time_us": start_us + 1_000_000,
                        "commit": {"operation": "create"}})

    mock_websocket.recv.side_effect = [
        big, good,
        websockets.exceptions.ConnectionClosed(Close(1000, "End"), None)
    ]

    records = [r async for r in Jetstream.fetch_minute_data(start_us, end_us)]
    assert len(records) == 1 and records[0]["commit"]["operation"] == "create"


@pytest.mark.asyncio
async def test_fetch_minute_data_exceeds_end(mock_websocket):
    start_us = 1_700_000_000_000_000
    end_us   = start_us + 5_000_000

    out_of_range = json.dumps({"time_us": end_us + 1_000_000})
    mock_websocket.recv.side_effect = [
        out_of_range,
        websockets.exceptions.ConnectionClosed(Close(1000, "End"), None)
    ]

    assert [r async for r in Jetstream.fetch_minute_data(start_us, end_us)] == []


@pytest.mark.asyncio
async def test_yield_jetstream_reversed(mocker):
    start_cursor = int((datetime.utcnow() - timedelta(minutes=5)).timestamp() * 1_000_000)
    end_cursor   = int(datetime.utcnow().timestamp() * 1_000_000)

    mock_slice = AsyncMock()
    mock_slice.__aiter__.return_value = [{"commit": {"operation": "create"}}] * 2
    mocker.patch("app.jetstream.Jetstream.fetch_minute_data", return_value=mock_slice)

    records = [r async for r in Jetstream.yield_jetstream_reversed(end_cursor=end_cursor,
                                                                   start_cursor=start_cursor)]
    assert records


@pytest.mark.asyncio
async def test_graceful_close(mock_websocket):
    await Jetstream.graceful_close(mock_websocket)
    mock_websocket.close.assert_awaited_once_with(code=1000, reason="End of slice")


# ---------------------------------------------------------------------------
# NEW tests — validate helper & flush path
# ---------------------------------------------------------------------------
def test_validate_raw_message_filters_and_fixes(monkeypatch):
    # spam filter
    assert Jetstream._validate_raw_message("leiarcaica") is None

    # size filter
    assert Jetstream._validate_raw_message("x" * 100_001) is None

    # future‑timestamp gets rewritten
    future = (datetime.utcnow() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    raw = json.dumps({"commit": {"record": {"createdAt": future}}})
    fixed = Jetstream._validate_raw_message(raw)
    assert fixed is not None
    parsed = json.loads(fixed)
    fixed_dt = datetime.strptime(parsed["commit"]["record"]["createdAt"],
                                 "%Y-%m-%dT%H:%M:%S.%fZ")
    assert fixed_dt <= datetime.utcnow()


def test_chunk_helper():
    data = list(range(23))
    chunks = list(Jetstream._chunk(data, 10))
    assert len(chunks) == 3 and chunks[0] == list(range(10))


@pytest.mark.asyncio
async def test_flush_calls_sqs_and_sets_cursor(monkeypatch):
    # patch the internal SQS sender
    send_mock = AsyncMock()
    monkeypatch.setattr(Jetstream, "_send_batch_to_sqs", send_mock)

    # dummy redis with async set
    class DummyRedis:
        def __init__(self): self.calls = []
        async def set(self, key, val): self.calls.append((key, val))

    redis = DummyRedis()

    batch = [json.dumps({"time_us": 123, "commit": {"operation": "create"}})]
    await Jetstream._flush(batch, redis)

    send_mock.assert_awaited()                           # SQS send happened
    assert redis.calls == [("jetstream:last_cursor", 122)]
