import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import websockets
from websockets.frames import Close
from datetime import datetime, timedelta, timezone
from app.jetstream import Jetstream  # Ensure correct import path



@pytest.fixture
def mock_websocket():
    """Fixture to create a properly mocked websocket connection."""
    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws.closed = False  # Default to open

    async def mock_connect(*args, **kwargs):
        return mock_ws  # Ensure `await websockets.connect(...)` returns an AsyncMock

    with patch("websockets.connect", new=mock_connect):
        yield mock_ws  # Yield the correctly configured mock

@pytest.mark.asyncio
async def test_fetch_minute_data_success(mock_websocket):
    """Test fetching a valid stream of messages within the time range."""
    start_us = 1_700_000_000_000_000
    end_us = start_us + 5_000_000

    valid_message = json.dumps({"time_us": start_us + 1_000_000, "commit": {"operation": "create"}})

    # Ensure correct side effects
    mock_websocket.recv.side_effect = [
        valid_message,
        websockets.exceptions.ConnectionClosed(Close(1000, "End"), None)
    ]

    results = [record async for record in Jetstream.fetch_minute_data(start_us, end_us)]
    
    assert results  # Ensure it processes correctly

@pytest.mark.asyncio
async def test_fetch_minute_data_timeout(mock_websocket):
    """Test that the function handles a timeout (2s silence) and exits correctly."""
    start_us = 1_700_000_000_000_000
    end_us = start_us + 5_000_000

    # Simulate timeout
    mock_websocket.recv.side_effect = asyncio.TimeoutError()

    results = [record async for record in Jetstream.fetch_minute_data(start_us, end_us)]

    assert results == []  # No messages received


@pytest.mark.asyncio
async def test_fetch_minute_data_large_message(mock_websocket):
    """Test that messages exceeding the size limit are skipped."""
    start_us = 1_700_000_000_000_000
    end_us = start_us + 5_000_000

    oversized_message = "X" * 100001
    valid_message = json.dumps({"time_us": start_us + 1_000_000, "commit": {"operation": "create"}})

    # Fix ConnectionClosed instantiation
    mock_websocket.recv.side_effect = [oversized_message, valid_message, websockets.exceptions.ConnectionClosed(Close(1000, "End"), None)]

    results = [record async for record in Jetstream.fetch_minute_data(start_us, end_us)]

    assert len(results) == 1  # The large message was skipped
    assert results[0]["commit"]["operation"] == "create"


@pytest.mark.asyncio
async def test_fetch_minute_data_exceeds_end(mock_websocket):
    """Test that messages past `end_us` trigger a graceful close."""
    start_us = 1_700_000_000_000_000
    end_us = start_us + 5_000_000

    # Message outside the valid range
    out_of_range_message = json.dumps({"time_us": end_us + 1_000_000})

    # Fix ConnectionClosed instantiation
    mock_websocket.recv.side_effect = [out_of_range_message, websockets.exceptions.ConnectionClosed(Close(1000, "End"), None)]

    results = [record async for record in Jetstream.fetch_minute_data(start_us, end_us)]

    assert len(results) == 0  # No valid messages within the range


@pytest.mark.asyncio
async def test_yield_jetstream_reversed(mocker):
    """Test `yield_jetstream_reversed` iterating over multiple slices."""
    start_cursor = int((datetime.utcnow() - timedelta(minutes=5)).timestamp() * 1_000_000)
    end_cursor = int(datetime.utcnow().timestamp() * 1_000_000)

    mock_fetch = AsyncMock()
    mock_fetch.__aiter__.return_value = [{"commit": {"operation": "create"}}] * 2  # Two messages per slice

    mocker.patch("app.jetstream.Jetstream.fetch_minute_data", return_value=mock_fetch)

    results = [record async for record in Jetstream.yield_jetstream_reversed(end_cursor=end_cursor, start_cursor=start_cursor)]

    assert len(results) > 0  # We should have received multiple messages


@pytest.mark.asyncio
async def test_graceful_close(mock_websocket):
    """Test `graceful_close` ensures websocket closes properly."""
    await Jetstream.graceful_close(mock_websocket)

    mock_websocket.close.assert_awaited_once_with(code=1000, reason="End of slice")
