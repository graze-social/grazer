import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.algos.base import BaseParser, ImageParser, get_cache_key

@pytest.mark.parametrize("model_name, url, category, expected", [
    ("modelA", "http://image.com/1.jpg", "cat1", get_cache_key("modelA", "http://image.com/1.jpg", "cat1")),
    ("modelB", "http://image.com/2.jpg", "cat2", get_cache_key("modelB", "http://image.com/2.jpg", "cat2")),
])
def test_get_cache_key(model_name, url, category, expected):
    assert get_cache_key(model_name, url, category) == expected

@pytest.mark.asyncio
async def test_base_parser_initialization():
    algo_manager = MagicMock()
    algo_manager.logic_evaluator = MagicMock()
    algo_manager.cache = MagicMock()
    algo_manager.gpu_embedding_workers = [MagicMock()]
    algo_manager.gpu_classifier_workers = [MagicMock()]
    algo_manager.network_workers = [MagicMock()]

    class TestParser(BaseParser):
        async def register_operations(self, logic_evaluator):
            pass
    
    parser = await TestParser.initialize(algo_manager)
    assert isinstance(parser, BaseParser)
    assert parser.cache == algo_manager.cache
    assert parser.gpu_embedding_workers == algo_manager.gpu_embedding_workers
    assert parser.gpu_classifier_workers == algo_manager.gpu_classifier_workers
    assert parser.network_workers == algo_manager.network_workers

@pytest.mark.asyncio
async def test_base_parser_no_workers():
    class TestParser(BaseParser):
        async def register_operations(self, logic_evaluator):
            pass
    
    parser = TestParser()
    parser.network_workers = []
    parser.gpu_embedding_workers = []
    parser.gpu_classifier_workers = []
    
    with pytest.raises(ValueError, match="No network_workers are available"):
        _ = parser.network_worker
    with pytest.raises(ValueError, match="No gpu_embedding_workers are available"):
        _ = parser.gpu_embedding_worker
    with pytest.raises(ValueError, match="No gpu_classifier_workers are available"):
        _ = parser.gpu_classifier_worker

@pytest.mark.asyncio
async def test_image_parser_fetch_image():
    parser = ImageParser()
    parser.network_workers = [MagicMock()]
    parser.network_workers[0].fetch_image.remote = AsyncMock(return_value="image_data")

    url = "http://example.com/image.jpg"
    result = await parser.fetch_image(url)
    assert result == "image_data"
    parser.network_workers[0].fetch_image.remote.assert_called_once_with(url)

@pytest.mark.asyncio
async def test_image_parser_fetch_images():
    parser = ImageParser()
    parser.network_workers = [MagicMock()]
    parser.fetch_image = AsyncMock(side_effect=["img1", "img2", Exception("Error fetching")])

    urls = ["url1", "url2", "url3"]
    results = await parser.fetch_images(urls)
    assert results[:2] == ["img1", "img2"]
    assert isinstance(results[2], Exception)

@pytest.mark.asyncio
async def test_image_parser_precache_predictions():
    parser = ImageParser()
    parser.cache = MagicMock()
    parser.cache.bulk_get_prediction.remote = AsyncMock(return_value=[None, [0.8, 0.2]])
    parser.MODEL_NAME = "test_model"

    urls = ["url1", "url2"]
    category = "cat"
    cached_probs, uncached_urls = await parser.precache_predictions(urls, category)
    assert cached_probs == {"url2": {"cat": 0.8, "not_cat": 0.2}}
    assert uncached_urls == ["url1"]

@pytest.mark.asyncio
async def test_image_parser_get_probabilities():
    parser = ImageParser()
    parser.precache_predictions = AsyncMock(return_value=({}, ["url1"]))
    parser.fetch_images = AsyncMock(return_value=["img_data"])
    parser.probability_function = AsyncMock(return_value=[[0.7, 0.3]])
    parser.process_probs = MagicMock(return_value={"cat": 0.7, "not_cat": 0.3})
    parser.MODEL_NAME = "test_model"

    urls = ["url1"]
    category = "cat"
    results = await parser.get_probabilities(urls, category)
    assert results == {"url1": 0.7}

@pytest.mark.asyncio
async def test_image_parser_get_ml_scores():
    parser = ImageParser()
    parser.get_all_image_urls = AsyncMock(return_value={"cid1": ["url1"]})
    parser.get_probabilities = AsyncMock(return_value={"url1": 0.8})

    records = [{"commit": {"cid": "cid1"}}]
    scores = await parser.get_ml_scores(records, "cat", ">=", 0.5)
    assert np.array_equal(scores, np.array([0.8]))

@pytest.mark.asyncio
async def test_image_parser_classifier_operator():
    parser = ImageParser()
    parser.get_all_image_urls = AsyncMock(return_value={"cid1": ["url1"]})
    parser.get_probabilities = AsyncMock(return_value={"url1": 0.8})
    parser.compare = AsyncMock(return_value=[True])

    records = [{"commit": {"cid": "cid1"}}]
    bools = await parser.classifier_operator(records, "cat", ">=", 0.5)
    assert np.array_equal(bools, np.array([True]))

@pytest.mark.asyncio
async def test_image_parser_precache_images():
    parser = ImageParser()
    parser.cache = MagicMock()
    parser.cache.bulk_get_image.remote = AsyncMock(return_value=[None, "cached_data"])
    parser.fetch_images = AsyncMock()

    urls = ["url1", "url2"]
    await parser.precache_images(urls)
    parser.fetch_images.assert_called_once_with(["url1"])
