import pytest
import numpy as np
import re2
from unittest.mock import AsyncMock, patch
from app.algos.operators.entity import EntityParser


@pytest.fixture(scope="function")
def entity_parser():
    """Fixture to provide an instance of EntityParser with a mocked network worker."""
    parser = EntityParser()

    # Mock network_worker's methods instead of replacing the attribute
    parser.network_workers = [AsyncMock()]
    parser.network_worker.get_or_set_handle_did = AsyncMock()
    parser.network_worker.get_list = AsyncMock()
    parser.network_worker.get_starter_pack = AsyncMock()

    return parser


@pytest.mark.parametrize(
    "records,expected",
    [
        (
            [
                {
                    "commit": {
                        "record": {
                            "embed": {"external": {"uri": "https://example.com"}},
                            "facets": [
                                {
                                    "features": [
                                        {"$type": "app.bsky.richtext.facet#link", "uri": "https://test.com"}
                                    ]
                                }
                            ],
                        }
                    }
                }
            ],
            [{"https://example.com", "https://test.com"}],
        ),
        ([{"commit": {"record": {}}}], [set()]),  # Ensures None values don't cause errors
    ],
)
def test_get_all_urls(records, expected):
    result = EntityParser.get_all_urls(records)

    # Convert {None} to set() in results before assertion
    cleaned_result = [{e for e in urls if e is not None} for urls in result]

    assert cleaned_result == expected, f"Expected {expected}, got {cleaned_result}"


@pytest.mark.asyncio
async def test_get_resolved_values_mentions(entity_parser):
    """Test resolution of mentions using network worker."""
    entity_parser.network_worker.get_or_set_handle_did.remote.return_value = "resolved_did"
    values = {"@user1", "@user2"}
    resolved_values = await entity_parser.get_resolved_values("mentions", values)

    assert resolved_values == {"resolved_did"}, "Resolved values did not match expected."


@pytest.mark.asyncio
async def test_get_resolved_values_hashtags(entity_parser):
    """Test resolution of hashtags by removing `#` symbol."""
    values = {"#tag1", "#tag2"}
    resolved_values = await entity_parser.get_resolved_values("hashtags", values)

    assert resolved_values == {"tag1", "tag2"}, "Hashtag resolution failed."


@pytest.mark.asyncio
async def test_matches_entities(entity_parser):
    """Test entity matching for hashtags."""
    entity_parser.get_all_hashtags = AsyncMock(return_value=[{"tag1"}, {"tag2"}])
    records = [{"commit": {"record": {"tags": ["tag1"]}}}, {"commit": {"record": {"tags": ["tag3"]}}}]
    values = {"tag1"}
    result = await entity_parser.matches_entities(records, "hashtags", values)

    assert np.array_equal(result, np.array([True, False])), "Entity match results incorrect."


@pytest.mark.asyncio
async def test_excludes_entities(entity_parser):
    """Test entity exclusion for hashtags."""
    entity_parser.get_all_hashtags = AsyncMock(return_value=[{"tag1"}, {"tag2"}])
    records = [{"commit": {"record": {"tags": ["tag1"]}}}, {"commit": {"record": {"tags": ["tag3"]}}}]
    values = {"tag1"}
    result = await entity_parser.excludes_entities(records, "hashtags", values)

    assert np.array_equal(result, np.array([False, True])), "Entity exclusion results incorrect."


@pytest.mark.asyncio
async def test_register_operations(entity_parser):
    """Test registering operations in logic evaluator."""
    logic_evaluator = AsyncMock()
    await entity_parser.register_operations(logic_evaluator)

    logic_evaluator.add_operation.assert_any_call("entity_matches", entity_parser.matches_entities)
    logic_evaluator.add_operation.assert_any_call("entity_excludes", entity_parser.excludes_entities)
