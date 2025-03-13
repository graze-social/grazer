import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from app.algos.operators.social import SocialParser
from app.logic_evaluator import LogicEvaluator

@pytest.mark.parametrize("asset, expected", [
    ({"asset_value": ["did1", "did2"]}, np.array(["did1", "did2"])),
    (np.array(["did3", "did4"]), np.array(["did3", "did4"])),
    ({}, {}),
])
def test_temp_hack(asset, expected):
    parser = SocialParser()
    result = parser.temp_hack(asset)
    assert np.array_equal(result, expected)

@pytest.mark.asyncio
async def test_get_starter_pack():
    parser = SocialParser()
    parser.network_workers = [MagicMock()]
    parser.network_workers[0].get_starter_pack.remote = AsyncMock(return_value="starter_pack_data")
    result = await parser.get_starter_pack("test_url")
    assert result == "starter_pack_data"

@pytest.mark.asyncio
async def test_get_list():
    parser = SocialParser()
    parser.network_workers = [MagicMock()]
    parser.network_workers[0].get_list.remote = AsyncMock(return_value="list_data")
    result = await parser.get_list("test_url")
    assert result == "list_data"

@pytest.mark.asyncio
async def test_get_user_collection():
    parser = SocialParser()
    parser.network_workers = [MagicMock()]
    parser.network_workers[0].get_user_collection.remote = AsyncMock(return_value="collection_data")
    result = await parser.get_user_collection("actor_handle", "direction")
    assert result == "collection_data"

@pytest.mark.asyncio
async def test_check_memberships():
    parser = SocialParser()
    logic_evaluator = LogicEvaluator()
    records = [{"did": "did1"}, {"did": "did2"}]
    result = await logic_evaluator.compare(np.array(["did1", "did2"]), "in", ["did1"])
    assert np.array_equal(result, np.array([True, False]))

@pytest.mark.asyncio
async def test_starter_pack_member():
    parser = SocialParser()
    parser.get_starter_pack = AsyncMock(return_value=["did1", "did2"])
    parser.check_memberships = AsyncMock(return_value=np.array([True, False]))
    records = [{"did": "did1"}, {"did": "did3"}]
    result = await parser.starter_pack_member(records, "test_url", "in")
    assert np.array_equal(result, np.array([True, False]))

@pytest.mark.asyncio
async def test_register_operations():
    parser = SocialParser()
    logic_evaluator = AsyncMock()
    
    await parser.register_operations(logic_evaluator)
    logic_evaluator.add_operation.assert_any_await("social_graph", parser.social_graph)
    logic_evaluator.add_operation.assert_any_await("social_list", parser.social_list)
    logic_evaluator.add_operation.assert_any_await("starter_pack_member", parser.starter_pack_member)
    logic_evaluator.add_operation.assert_any_await("list_member", parser.list_member)
    logic_evaluator.add_operation.assert_any_await("magic_audience", parser.magic_audience)
