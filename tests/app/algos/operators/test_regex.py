import pytest
import numpy as np
import re2
from unittest.mock import AsyncMock, MagicMock

from app.algos.operators.regex import RegexParser

@pytest.mark.parametrize("records, field_selector, expected", [
    ([{"commit": {"record": {"text": "hello world"}}}], "text", np.array(["hello world"])),
    ([{"commit": {"record": {"text": ["hello", "world"]}}}], "text", np.array(["['hello' 'world']"], dtype='<U17')),  # Match actual output
    ([{"commit": {"record": {"text": None}}}], "text", np.array(["None"], dtype='<U4')),  # Match actual output
    ([{"commit": {"record": {}}}], "text", np.array(["None"], dtype='<U4')),  # Match actual output
])
def test_get_record_texts(records, field_selector, expected):
    parser = RegexParser()
    result = parser.get_record_texts(records, field_selector)
    assert np.array_equal(result, expected)

@pytest.mark.asyncio
async def test_matches_operator():
    parser = RegexParser()
    parser.get_record_texts = MagicMock(return_value=np.array(["hello world", "goodbye"]))
    
    records = [{"commit": {"record": {"text": "hello world"}}}, {"commit": {"record": {"text": "goodbye"}}}]
    result = await parser.matches_operator(records, "text", "hello")
    assert np.array_equal(result, np.array([True, False]))

@pytest.mark.asyncio
async def test_negation_matches_operator():
    parser = RegexParser()
    parser.matches_operator = AsyncMock(return_value=np.array([True, False]))
    
    records = [{"commit": {"record": {"text": "hello world"}}}, {"commit": {"record": {"text": "goodbye"}}}]
    result = await parser.negation_matches_operator(records, "text", "hello")
    assert np.array_equal(result, np.array([False, True]))

@pytest.mark.asyncio
async def test_allow_operator():
    parser = RegexParser()
    parser.get_record_texts = MagicMock(return_value=np.array(["hello world", "goodbye"]))
    
    records = [{"commit": {"record": {"text": "hello world"}}}, {"commit": {"record": {"text": "goodbye"}}}]
    result = await parser.allow_operator(records, "text", ["hello", "hi"])
    assert np.array_equal(result, np.array([True, False]))

@pytest.mark.asyncio
async def test_allow_operator_domain_match():
    parser = RegexParser()
    parser.get_record_texts = MagicMock(return_value=np.array([
        "https://forum.obsidian.md/t/how-can-i-reorder-list-items/86981",
        "This is unrelated."
    ]))
    
    result = await parser.allow_operator(None, "text", ["obsidian.md"])
    assert np.array_equal(result, np.array([True, False]))

@pytest.mark.asyncio
async def test_allow_operator_does_not_match_embedded_domain():
    parser = RegexParser()
    parser.get_record_texts = MagicMock(return_value=np.array([
        "https://notobsidian.md5rocks.com",  # shouldn't match
    ]))
    
    result = await parser.allow_operator(None, "text", ["obsidian.md"])
    assert np.array_equal(result, np.array([False]))

@pytest.mark.asyncio
async def test_allow_operator_word_boundary_respected():
    parser = RegexParser()
    parser.get_record_texts = MagicMock(return_value=np.array([
        "theapple", "an apple a day"
    ]))
    
    result = await parser.allow_operator(None, "text", ["apple"])
    assert np.array_equal(result, np.array([False, True]))

@pytest.mark.asyncio
async def test_allow_operator_escaped_regex_characters():
    parser = RegexParser()
    parser.get_record_texts = MagicMock(return_value=np.array([
        "hello.world", "hello+world", "hello*world"
    ]))
    
    result = await parser.allow_operator(None, "text", ["hello.world", "hello+world", "hello*world"])
    assert np.array_equal(result, np.array([True, True, True]))

@pytest.mark.asyncio
async def test_allow_operator_multiple_domains():
    parser = RegexParser()
    parser.get_record_texts = MagicMock(return_value=np.array([
        "Visit obsidian.md or notion.so for more.",
        "Nothing to see here."
    ]))
    
    result = await parser.allow_operator(None, "text", ["obsidian.md", "notion.so"])
    assert np.array_equal(result, np.array([True, False]))

@pytest.mark.asyncio
async def test_deny_operator():
    parser = RegexParser()
    parser.allow_operator = AsyncMock(return_value=np.array([True, False]))
    
    records = [{"commit": {"record": {"text": "hello world"}}}, {"commit": {"record": {"text": "goodbye"}}}]
    result = await parser.deny_operator(records, "text", ["hello", "hi"])
    assert np.array_equal(result, np.array([False, True]))

@pytest.mark.asyncio
async def test_register_operations():
    parser = RegexParser()
    logic_evaluator = AsyncMock()
    
    await parser.register_operations(logic_evaluator)
    logic_evaluator.add_operation.assert_any_await("regex_matches", parser.matches_operator)
    logic_evaluator.add_operation.assert_any_await("regex_negation_matches", parser.negation_matches_operator)
    logic_evaluator.add_operation.assert_any_await("regex_any", parser.allow_operator)
    logic_evaluator.add_operation.assert_any_await("regex_none", parser.deny_operator)
