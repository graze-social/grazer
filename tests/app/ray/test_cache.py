import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import ray

# Fully mock `ray.remote` before importing `Cache`
with patch("ray.remote", lambda *args, **kwargs: lambda cls: cls):
    from app.ray.cache import Cache  # Re-import after patching

@pytest.fixture(scope="function")
def cache_instance(mocker):
    """
    Provide a Cache instance with Ray removed.
    Mocks out dependencies on Redis and Egress.
    """
    print("Setting up mock Cache instance...")
    
    # Patch dependencies
    mocker.patch('app.egress.Egress.send_results', new=AsyncMock(return_value=None))
    mocker.patch('app.redis.RedisClient.send_pipeline', new=AsyncMock(return_value=None))
    
    # Directly instantiate Cache (since ray.remote is now mocked)
    instance = Cache()  
    return instance

@pytest.mark.asyncio
async def test_cache_image(cache_instance):
    try:
        print("Running test_cache_image...")
        await cache_instance.cache_image('img1', b'data1')
        result = await cache_instance.get_image('img1')
        assert result == b'data1'
    except Exception as e:
        print(f"test_cache_image failed with error: {e}")
        assert False

@pytest.mark.asyncio
async def test_bulk_get_image(cache_instance):
    try:
        print("Running test_bulk_get_image...")
        images = {'img1': b'data1', 'img2': b'data2'}
        for k, v in images.items():
            await cache_instance.cache_image(k, v)
        results = await cache_instance.bulk_get_image(['img1', 'img2', 'img3'])
        assert results == [b'data1', b'data2', None]
    except Exception as e:
        print(f"test_bulk_get_image failed with error: {e}")
        assert False

@pytest.mark.asyncio
async def test_cache_asset_ttl(cache_instance):
    try:
        print("Running test_cache_asset_ttl...")
        await cache_instance.cache_asset('asset1', {'foo': 'bar'})
        result = await cache_instance.get_asset('asset1')
        assert result == {'foo': 'bar'}
    except Exception as e:
        print(f"test_cache_asset_ttl failed with error: {e}")
        assert False

@pytest.mark.asyncio
async def test_bulk_cache_prediction(cache_instance):
    try:
        print("Running test_bulk_cache_prediction...")
        ids = ['pred1', 'pred2']
        data = [0.9, 0.1]
        await cache_instance.bulk_cache_prediction(ids, data)
        results = await cache_instance.bulk_get_prediction(ids)
        assert results == data
    except Exception as e:
        print(f"test_bulk_cache_prediction failed with error: {e}")
        assert False

@pytest.mark.asyncio
async def test_report_output_and_flush(cache_instance):
    try:
        print("Running test_report_output_and_flush...")
        await cache_instance.report_output({'result': 1})
        await cache_instance.report_output({'foo': 'bar'}, force_write=True)
        await asyncio.sleep(0.5)
    except Exception as e:
        print(f"test_report_output_and_flush failed with error: {e}")
        assert False

@pytest.mark.asyncio
async def test_cache_embedding_bulk(cache_instance):
    try:
        print("Running test_cache_embedding_bulk...")
        keys = ['key1', 'key2']
        values = [[0.1, 0.2], [0.3, 0.4]]
        await cache_instance.bulk_cache_embedding(keys, values)
        results = await cache_instance.bulk_get_embedding(keys)
        assert results == values
    except Exception as e:
        print(f"test_cache_embedding_bulk failed with error: {e}")
        assert False

@pytest.mark.asyncio
async def test_cache_did(cache_instance):
    try:
        print("Running test_cache_did...")
        await cache_instance.cache_did('did1', {'data': '123'})
        result = await cache_instance.get_did('did1')
        assert result == {'data': '123'}
    except Exception as e:
        print(f"test_cache_did failed with error: {e}")
        assert False

@pytest.mark.asyncio
async def test_concurrent_cache_operations(cache_instance):
    try:
        print("Running test_concurrent_cache_operations...")
        tasks = [cache_instance.cache_image(f'img{i}', f'data{i}'.encode()) for i in range(20)]
        await asyncio.gather(*tasks)
        results = await cache_instance.bulk_get_image([f'img{i}' for i in range(20)])
        expected = [f'data{i}'.encode() for i in range(20)]
        assert results == expected
    except Exception as e:
        print(f"test_concurrent_cache_operations failed with error: {e}")
        assert False
