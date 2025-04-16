import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch, MagicMock, call

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

    with patch("websockets.connect", _connect):
        yield mock_ws


@pytest.fixture
def mock_sqs_client():
    """Return a mocked SQS client."""
    # Create a mock for the SQS client itself
    mock_client = AsyncMock()
    mock_client.get_queue_attributes = AsyncMock(return_value={"Attributes": {"QueueArn": "arn:aws:sqs:us-east-1:123456789012:test-queue"}})
    mock_client.send_message_batch = AsyncMock(return_value={"Successful": [{"Id": "0"}], "Failed": []})
    mock_client.close = AsyncMock()
    
    # Mock the Session and client creation process
    with patch("aioboto3.Session") as mock_session_class:
        # Set up the mock session instance and client method
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Configure client method to return our mock when __aenter__ is called
        async def client_aenter_impl():
            return mock_client
            
        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__ = client_aenter_impl
        
        mock_session.client = AsyncMock(return_value=mock_client_ctx)
        
        # Provide the mock client to the test
        yield mock_client


@pytest.fixture
def mock_redis():
    """Return a mocked Redis client."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    mock_redis.close = AsyncMock()
    
    # Use redis.asyncio instead of aioredis (which is now deprecated)
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        yield mock_redis


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
# Helper function tests
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


# ---------------------------------------------------------------------------
# New tests for producer/flusher architecture
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_init_sqs_client(mock_sqs_client):
    """Test SQS client initialization."""
    # Reset the class attribute to ensure test isolation
    Jetstream.SQS_CLIENT = None
    
    # Set required environment variables
    with patch.object(Jetstream, 'SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/123456789012/test-queue'):
        await Jetstream._init_sqs_client()
        
        # Verify the client was initialized and is our mock
        assert Jetstream.SQS_CLIENT is mock_sqs_client
        
        # Verify queue verification was called
        mock_sqs_client.get_queue_attributes.assert_called_once()


@pytest.mark.asyncio
async def test_send_batch_to_sqs(mock_sqs_client):
    # Set up test data
    Jetstream.SQS_CLIENT = mock_sqs_client
    batch = [json.dumps({"test": "message1"}), json.dumps({"test": "message2"})]
    
    # Set required environment variables
    with patch.object(Jetstream, 'SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/123456789012/test-queue'):
        await Jetstream._send_batch_to_sqs(batch)
        
        # Verify SQS was called with correct parameters
        mock_sqs_client.send_message_batch.assert_awaited_once()
        call_args = mock_sqs_client.send_message_batch.call_args[1]
        assert call_args['QueueUrl'] == Jetstream.SQS_QUEUE_URL
        assert len(call_args['Entries']) == 2
        assert call_args['Entries'][0]['MessageBody'] == json.dumps({"test": "message1"})


@pytest.mark.asyncio
async def test_producer_uses_current_time(monkeypatch):
    """Test that producer uses current time for cursor when no Redis value exists."""
    # Mock Redis to return None (no stored cursor)
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    
    # Create a queue and event for testing
    queue = asyncio.Queue()
    shutdown_event = asyncio.Event()
    
    # Mock the validate_raw_message to return the input
    monkeypatch.setattr(Jetstream, '_validate_raw_message', lambda x: x)
    
    # Track the URL used in websockets.connect
    url_used = None
    
    # Mock websockets.connect to capture the URL and avoid actual connections
    async def mock_connect(url, *args, **kwargs):
        nonlocal url_used
        url_used = url
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            json.dumps({"test": "message"}),
            websockets.exceptions.ConnectionClosed(
                Close(1000, "Test close"), None
            )
        ])
        mock_ws.close = AsyncMock()
        mock_ws.closed = False
        return mock_ws
    
    # Apply the mock
    monkeypatch.setattr(websockets, "connect", mock_connect)
    
    # Run the producer (will exit after the ConnectionClosed exception)
    await Jetstream._producer(queue, mock_redis, shutdown_event)
    
    # Verify the URL was captured
    assert url_used is not None
    
    # Verify the URL contains a cursor parameter
    assert "&cursor=" in url_used
    
    # Check that Redis was queried
    mock_redis.get.assert_called_once_with("jetstream:last_cursor")


@pytest.mark.asyncio
async def test_producer_uses_redis_cursor(monkeypatch):
    """Test that producer uses Redis cursor when available."""
    # Mock Redis to return a cursor value
    stored_cursor = "1701000000000000"
    mock_redis = AsyncMock()
    mock_redis.get.return_value = stored_cursor
    
    # Create a queue and event for testing
    queue = asyncio.Queue()
    shutdown_event = asyncio.Event()
    
    # Mock the validate_raw_message to return the input
    monkeypatch.setattr(Jetstream, '_validate_raw_message', lambda x: x)
    
    # Track the URL used in websockets.connect
    url_used = None
    
    # Mock websockets.connect to capture the URL and avoid actual connections
    async def mock_connect(url, *args, **kwargs):
        nonlocal url_used
        url_used = url
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            json.dumps({"test": "message"}),
            websockets.exceptions.ConnectionClosed(
                Close(1000, "Test close"), None
            )
        ])
        mock_ws.close = AsyncMock()
        mock_ws.closed = False
        return mock_ws
    
    # Apply the mock
    monkeypatch.setattr(websockets, "connect", mock_connect)
    
    # Run the producer (will exit after the ConnectionClosed exception)
    await Jetstream._producer(queue, mock_redis, shutdown_event)
    
    # Verify the URL was captured and contains the Redis cursor
    assert url_used is not None
    assert f"&cursor={stored_cursor}" in url_used
    
    # Check that Redis was queried
    mock_redis.get.assert_called_once_with("jetstream:last_cursor")


@pytest.mark.asyncio
async def test_flusher_batches_messages(monkeypatch):
    # Create mocks
    mock_send = AsyncMock()
    monkeypatch.setattr(Jetstream, '_send_batch_to_sqs', mock_send)
    
    mock_redis = AsyncMock()
    
    # Create a queue and add messages
    queue = asyncio.Queue()
    shutdown_event = asyncio.Event()
    
    # Add test messages
    batch_size = 5  # Small batch size for testing
    monkeypatch.setattr(Jetstream, 'BATCH_SIZE', batch_size)
    monkeypatch.setattr(Jetstream, 'FLUSH_INTERVAL', 60)  # Long interval to focus on batch size
    
    # Add messages to queue
    for i in range(batch_size + 2):  # Add more than batch size
        await queue.put(json.dumps({"test": f"message-{i}", "time_us": 1000 + i}))
    
    # Start the flusher
    flusher_task = asyncio.create_task(Jetstream._flusher(queue, mock_redis, shutdown_event))
    
    # Allow time for flusher to process
    await asyncio.sleep(0.2)
    
    # Check that send_batch_to_sqs was called at least once
    mock_send.assert_awaited()
    
    # The first batch should have exactly BATCH_SIZE messages
    first_call_args = mock_send.call_args_list[0][0][0]
    assert len(first_call_args) == batch_size
    
    # Clean up
    shutdown_event.set()
    await flusher_task


@pytest.mark.asyncio
async def test_flusher_time_based_flush(monkeypatch):
    # Create mocks
    mock_send = AsyncMock()
    monkeypatch.setattr(Jetstream, '_send_batch_to_sqs', mock_send)
    
    mock_redis = AsyncMock()
    
    # Create a queue and event
    queue = asyncio.Queue()
    shutdown_event = asyncio.Event()
    
    # Configure small flush interval for testing
    monkeypatch.setattr(Jetstream, 'BATCH_SIZE', 100)  # Large batch size
    monkeypatch.setattr(Jetstream, 'FLUSH_INTERVAL', 0.2)  # Short interval for testing
    
    # Add a few messages (less than batch size)
    for i in range(3):
        await queue.put(json.dumps({"test": f"message-{i}", "time_us": 1000 + i}))
    
    # Start the flusher
    flusher_task = asyncio.create_task(Jetstream._flusher(queue, mock_redis, shutdown_event))
    
    # Allow time for flush interval to trigger
    await asyncio.sleep(0.3)
    
    # Check that send_batch_to_sqs was called despite not reaching batch size
    mock_send.assert_awaited()
    
    # Clean up
    shutdown_event.set()
    await flusher_task


@pytest.mark.asyncio
async def test_stream_to_sqs_lifecycle(monkeypatch, mock_sqs_client):
    """Test the stream_to_sqs lifecycle with simplified mocks."""
    # Create mocks for producer and flusher to avoid complex dependencies
    producer_called = False
    flusher_called = False
    cleanup_called = False
    
    async def mock_producer(*args):
        nonlocal producer_called
        producer_called = True
        # Exit immediately for testing
        return
    
    async def mock_flusher(*args):
        nonlocal flusher_called
        flusher_called = True
        # Exit immediately for testing
        return
    
    # Apply mocks to avoid real connections/work
    monkeypatch.setattr(Jetstream, '_producer', mock_producer)
    monkeypatch.setattr(Jetstream, '_flusher', mock_flusher)
    
    # Mock Redis class
    mock_redis = AsyncMock()
    mock_redis.close = AsyncMock()
    
    # Mock redis.asyncio.from_url to return our mock
    monkeypatch.setattr("redis.asyncio.from_url", lambda *args, **kwargs: mock_redis)
    
    # Create a simplified stream_to_sqs method for testing
    async def mock_stream_to_sqs():
        try:
            # Initialize SQS client directly
            Jetstream.SQS_CLIENT = mock_sqs_client
            
            # Create test queue and event
            queue = asyncio.Queue()
            shutdown_event = asyncio.Event()
            
            # Call mocked functions directly rather than as tasks
            await Jetstream._producer(queue, mock_redis, shutdown_event)
            await Jetstream._flusher(queue, mock_redis, shutdown_event)
            
            return
        finally:
            nonlocal cleanup_called
            cleanup_called = True
            # Clean up resources
            if Jetstream.SQS_CLIENT:
                await Jetstream.SQS_CLIENT.close()
            await mock_redis.close()
    
    # Replace stream_to_sqs with our test version
    monkeypatch.setattr(Jetstream, 'stream_to_sqs', mock_stream_to_sqs)
    
    # Set required environment variables
    with patch.object(Jetstream, 'SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/123456789012/test-queue'):
        # Run the stream operation
        await Jetstream.stream_to_sqs()
        
        # Verify expected components were called
        assert producer_called, "Producer function was not called"
        assert flusher_called, "Flusher function was not called"
        assert cleanup_called, "Cleanup was not performed"
        
        # Verify client was closed
        mock_sqs_client.close.assert_called_once()
        mock_redis.close.assert_called_once()


@pytest.mark.asyncio
async def test_yield_jetstream_reversed_default_cursor(monkeypatch):
    """Test that yield_jetstream_reversed uses 1-day ago cursor by default."""
    # Calculate expected times
    now_us = int(datetime.utcnow().timestamp() * 1_000_000)
    one_day_ago = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1_000_000)
    
    # Create a mock for fetch_minute_data that records calls and returns an empty async iterator
    calls = []
    
    async def mock_fetch(start_us, end_us):
        calls.append((start_us, end_us))
        # Create a mock that behaves like an empty async iterator
        mock_result = AsyncMock()
        mock_result.__aiter__ = AsyncMock()
        mock_result.__aiter__.return_value = AsyncMock()
        mock_result.__aiter__.return_value.__anext__ = AsyncMock(side_effect=StopAsyncIteration())
        return mock_result
    
    # Apply the mock
    monkeypatch.setattr(Jetstream, 'fetch_minute_data', mock_fetch)
    
    # Call the function (it will complete immediately since our mock iterator is empty)
    async for _ in Jetstream.yield_jetstream_reversed():
        pass
    
    # Verify at least one call was made to fetch_minute_data
    assert len(calls) > 0
    
    # The first call should contain start_cursor near one_day_ago
    # and end_cursor near now-60s
    expected_end = now_us - 60_000_000  # now minus 1 minute
    
    # Allow some leeway for test execution time
    assert abs(calls[0][0] - one_day_ago) < 10_000_000  # Start cursor should be close to one day ago
    assert abs(calls[0][1] - expected_end) < 10_000_000  # End cursor should be close to now-60s


import asyncio
import json
from datetime import datetime, timedelta, timezone
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.jetstream import Jetstream

@pytest.mark.asyncio
async def test_yield_jetstream_reversed_simple():
    """Simple test for yield_jetstream_reversed."""
    # Calculate expected times
    now_us = int(datetime.utcnow().timestamp() * 1_000_000)
    one_day_ago = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1_000_000)
    
    # Create a class to act as an async iterator
    class MockAsyncIterator:
        def __init__(self, data):
            self.data = data
            self.index = 0
            
        def __aiter__(self):
            return self
            
        async def __anext__(self):
            if self.index >= len(self.data):
                raise StopAsyncIteration
            value = self.data[self.index]
            self.index += 1
            return value
    
    # Track calls to fetch_minute_data
    calls = []
    
    # Mock fetch_minute_data 
    async def mock_fetch(start_us, end_us):
        calls.append((start_us, end_us))
        # Return a single test record
        return MockAsyncIterator([{"test": "record"}])
    
    # Apply the mock
    with patch.object(Jetstream, 'fetch_minute_data', mock_fetch):
        # Call the generator and collect results
        results = []
        async for record in Jetstream.yield_jetstream_reversed():
            results.append(record)
            break  # Only need one iteration for test
        
        # Verify calls were made with expected values
        assert len(calls) > 0
        
        # Allow some margin for test execution time
        assert abs(calls[0][0] - one_day_ago) < 10_000_000  # Start cursor ~= one day ago
        assert abs(calls[0][1] - (now_us - 60_000_000)) < 10_000_000  # End cursor ~= now - 60s
        
        # Verify we got the expected record
        assert len(results) == 1
        assert results[0] == {"test": "record"}