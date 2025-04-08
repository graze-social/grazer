import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from app.algos.operators.attribute import AttributeParser, resolve_path_batch, EMBED_TYPES
from app.logic_evaluator import LogicEvaluator

@pytest.mark.parametrize("records, path, expected", [
    ([{"commit": {"record": {"langs": ["en", "fr"]}}}], "langs[0]", np.array(["en"])),
    ([{"commit": {"record": {"embed": {"images": [{"alt": "desc1"}, {"alt": "desc2"}]}}}}], "embed.images[*].alt", np.array([["desc1", "desc2"]], dtype=object)),
    ([{"commit": {"record": {"nested": {"key": "value"}}}}], "nested.key", np.array(["value"])),
    ([{"commit": {"record": {"missing": ""}}}], "nonexistent.key", np.array([None], dtype=object)),
])
def test_resolve_path_batch(records, path, expected):
    result = resolve_path_batch(records, path)
    assert np.array_equal(result, expected)

@pytest.mark.asyncio
async def test_attribute_compare():
    parser = AttributeParser()
    parser.attribute_compare = AsyncMock(return_value=np.array([True, False, True]))
    records = [{"commit": {"record": {"langs": ["en", "fr"]}}}] * 3
    field_selector = "langs[0]"
    operator = "=="
    target_value = "en"
    
    result = await parser.attribute_compare(records, field_selector, operator, target_value)
    assert np.array_equal(result, np.array([True, False, True]))

@pytest.mark.asyncio
async def test_post_type():
    parser = AttributeParser()
    parser.post_type = AsyncMock(return_value=np.array([True, False, True]))
    records = [{"commit": {"record": {"embed": {"$type": EMBED_TYPES["post"]}}}}] * 3
    operator = "in"
    post_names = ["quote"]
    
    result = await parser.post_type(records, operator, post_names)
    assert np.array_equal(result, np.array([True, False, True]))

@pytest.mark.asyncio
async def test_embed_type():
    parser = AttributeParser()
    parser.embed_type = AsyncMock(return_value=np.array([True, False, True]))
    records = [{"commit": {"record": {"embed": {"$type": EMBED_TYPES["image"]}}}}] * 3
    operator = "=="
    embed_name = "image"
    
    result = await parser.embed_type(records, operator, embed_name)
    assert np.array_equal(result, np.array([True, False, True]))

@pytest.mark.asyncio
async def test_register_operations():
    parser = AttributeParser()
    logic_evaluator = AsyncMock()
    
    await parser.register_operations(logic_evaluator)
    logic_evaluator.add_operation.assert_any_await("attribute_compare", parser.attribute_compare)
    logic_evaluator.add_operation.assert_any_await("embed_type", parser.embed_type)
    logic_evaluator.add_operation.assert_any_await("post_type", parser.post_type)
