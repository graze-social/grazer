import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from io import BytesIO
from PIL import Image
import httpx
import aiohttp

# Patch `ray.remote` before importing NetworkWorker
with patch("ray.remote", lambda *args, **kwargs: lambda cls: cls):
    from app.ray.network_worker import NetworkWorker


@pytest.fixture(scope="function")
def network_worker_instance(mocker):
    """
    Provide a NetworkWorker instance with Ray removed.
    Mocks out dependencies on Cache, Semaphore, and Network Calls.
    """
    print("Setting up mock NetworkWorker instance...")

    # Mock dependencies
    mock_cache = AsyncMock()
    mock_bluesky_semaphore = AsyncMock()
    mock_graze_semaphore = AsyncMock()

    # Instantiate NetworkWorker with mocks
    instance = NetworkWorker(
        cache=mock_cache,
        bluesky_semaphore=mock_bluesky_semaphore,
        graze_semaphore=mock_graze_semaphore,
    )
    return instance


@pytest.mark.asyncio
async def test_fetch_asset_failure(network_worker_instance, mocker):
    """
    Test asset retrieval with HTTP failure.
    """
    print("Running test_fetch_asset_failure...")

    async def mock_post(*args, **kwargs):
        raise httpx.HTTPError("HTTP Error")

    mocker.patch("httpx.AsyncClient.post", side_effect=mock_post)

    with pytest.raises(httpx.HTTPError, match="HTTP Error"):
        await network_worker_instance.fetch_asset("test_type", "test_name", {})


@pytest.mark.asyncio
async def test_get_asset_fetch_and_cache(network_worker_instance, mocker):
    """
    Test fetching and caching an asset when not found in cache.
    """
    print("Running test_get_asset_fetch_and_cache...")

    network_worker_instance.cache.get_asset.remote.return_value = None
    mocker.patch.object(network_worker_instance, "fetch_asset", new=AsyncMock(return_value={"fetched": True}))

    await network_worker_instance.get_asset("test_type", {}, "template")

    expected_key = "test_type__test_type__"
    network_worker_instance.cache.cache_asset.remote.assert_called_once_with(expected_key, {"fetched": True})


@pytest.mark.asyncio
async def test_get_or_set_handle_did_cached(network_worker_instance):
    """
    Test retrieving an existing DID from cache.
    """
    print("Running test_get_or_set_handle_did_cached...")

    network_worker_instance.cache.get_did.remote.return_value = "did:plc:mock"

    result = await network_worker_instance.get_or_set_handle_did("test-handle")

    assert result == "did:plc:mock"



@pytest.mark.asyncio
async def test_get_or_set_handle_did_fetch(network_worker_instance, mocker):
    """
    Test resolving a handle when not in cache.
    """
    print("Running test_get_or_set_handle_did_fetch...")

    network_worker_instance.cache.get_did.remote.return_value = None
    mock_resolver = MagicMock()
    mock_resolver.handle.resolve.return_value = "did:plc:resolved"

    mocker.patch("app.ray.network_worker.IdResolver", return_value=mock_resolver)

    result = await network_worker_instance.get_or_set_handle_did("test-handle")

    assert result == "did:plc:resolved"
    network_worker_instance.cache.cache_did.remote.assert_called_once_with("test-handle", "did:plc:resolved")
