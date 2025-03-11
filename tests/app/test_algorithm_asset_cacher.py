import pytest
from app.algorithm_asset_cacher import AlgorithmAssetCacher

@pytest.mark.parametrize("actor_handle, direction, expected", [
    ("user1", "inbound", {
        "asset_type": "user_collection",
        "asset_parameters": {"actor_handle": "user1", "direction": "inbound"},
        "keyname_template": "{actor_handle}__{direction}",
    }),
    ("user2", "outbound", {
        "asset_type": "user_collection",
        "asset_parameters": {"actor_handle": "user2", "direction": "outbound"},
        "keyname_template": "{actor_handle}__{direction}",
    }),
])
def test_get_user_collection_params(actor_handle, direction, expected):
    assert AlgorithmAssetCacher.get_user_collection_params(actor_handle, direction) == expected

@pytest.mark.parametrize("starter_pack_url, expected", [
    ("http://example.com/starter1", {
        "asset_type": "starter_pack",
        "asset_parameters": {"starter_pack_url": "http://example.com/starter1"},
        "keyname_template": "{starter_pack_url}",
    }),
    ("http://example.com/starter2", {
        "asset_type": "starter_pack",
        "asset_parameters": {"starter_pack_url": "http://example.com/starter2"},
        "keyname_template": "{starter_pack_url}",
    }),
])
def test_get_starter_pack_params(starter_pack_url, expected):
    assert AlgorithmAssetCacher.get_starter_pack_params(starter_pack_url) == expected

@pytest.mark.parametrize("list_url, expected", [
    ("http://example.com/list1", {
        "asset_type": "user_list",
        "asset_parameters": {"list_url": "http://example.com/list1"},
        "keyname_template": "{list_url}",
    }),
    ("http://example.com/list2", {
        "asset_type": "user_list",
        "asset_parameters": {"list_url": "http://example.com/list2"},
        "keyname_template": "{list_url}",
    }),
])
def test_get_list_params(list_url, expected):
    assert AlgorithmAssetCacher.get_list_params(list_url) == expected
