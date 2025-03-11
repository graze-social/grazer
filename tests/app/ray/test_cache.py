import pytest
import asyncio
from app.ray.cache import Cache

@pytest.fixture
async def cache_actor(ray_local):
    actor = Cache.remote(batch_size=2)
    yield actor
    ray.kill(actor)

@pytest.mark.asyncio
async def test_cache_image(cache_actor):
    await cache_actor.cache_image.remote('img1', b'data1')
    result = await cache_actor.get_image.remote('img1')
    assert result == b'data1'

@pytest.mark.asyncio
async def test_bulk_get_image(cache_actor):
    images = {'img1': b'data1', 'img2': b'data2'}
    for img_id, data in images.items():
        await cache_actor.cache_image.remote(img_id, data)

    results = await cache_actor.bulk_get_image.remote(['img1', 'img2', 'img3'])
    assert results == [b'data1', b'data1', None]

@pytest.mark.asyncio
async def test_cache_asset_ttl(cache_actor):
    await cache_actor.cache_asset.remote('asset1', {'foo': 'bar'})
    result = await cache_actor.get_asset.remote('asset1')
    assert result == {'foo': 'bar'}

@pytest.mark.asyncio
async def test_bulk_cache_prediction(cache_actor):
    ids = ['pred1', 'pred2']
    data = [0.9, 0.1]
    await cache_actor.bulk_cache_prediction.remote(ids, data)
    results = await cache_actor.bulk_get_prediction.remote(ids)
    assert results == data

@pytest.mark.asyncio
async def test_report_output_and_flush(mocker, cache_actor):
    mocked_egress = mocker.patch('app.ray.cache.Egress.send_results', return_value=None)

    # Batch size = 3 for testing
    test_actor = Cache.remote(batch_size=2)

    await cache_actor.report_output.remote({'result': 1})
    await cache_actor.report_output.remote({'foo': 'bar'})
    
    # should not flush yet (batch size = 100 by default)
    assert not mocked_egress.called

    # Force flush
    await cache_actor.report_output.remote({'force': 'flush'}, force_write=True)
    assert mocked_egress.called_once_with(
        [{'force': 'flush'}], 'ray_workers:output'
    )

@pytest.mark.asyncio
async def test_cache_embedding_bulk(cache_actor):
    keys = ['key1', 'key2']
    values = [[0.1, 0.2], [0.3, 0.4]]
    await cache_actor.bulk_cache_embedding.remote(keys, values)
    results = await cache_actor.bulk_get_embedding.remote(keys)
    assert results == values

@pytest.mark.asyncio
async def test_cache_did(cache_actor):
    await cache_actor.cache_did.remote('did1', {'data': '123'})
    result = await cache_actor.get_did.remote('did1')
    assert result == {'data': '123'}

@pytest.mark.asyncio
async def test_concurrent_cache_operations(cache_actor):
    # Test concurrent writes to verify RWLock integrity
    tasks = [
        cache_actor.cache_image.remote(f'img{i}', f'data{i}'.encode()) 
        for i in range(20)
    ]
    await asyncio.gather(*tasks)

    results = await cache_actor.bulk_get_image.remote([f'img{i}' for i in range(20)])
    expected = [b'data' + str(i).encode() for i in range(20)]
    assert results == expected
