"""
Unit tests for app/helpers.py
"""

import pytest
import numpy as np
from app.helpers import (
    chunk,
    dict_to_sorted_string,
    split_into_k_lists,
    is_truthy,
    create_exception_json,
    get_url_domain,
    get_all_links,
    transform_dict,
)

def test_chunk_empty_iterable():
    result = list(chunk([], 3))
    assert result == []

def test_chunk_size_larger_than_list():
    result = list(chunk([1, 2], 5))
    assert result == [[1, 2]]

def test_chunk_exact_division():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_chunk_nonexact_division():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_dict_to_sorted_string_empty():
    assert dict_to_sorted_string({}) == ""

def test_dict_to_sorted_string_normal():
    data = {"b": 2, "a": 1, "c": 3}
    result = dict_to_sorted_string(data)
    assert result == "a:1__b:2__c:3"

def test_split_into_k_lists_valid():
    elements = [1, 2, 3, 4, 5]
    k = 2
    result = split_into_k_lists(elements, k)
    # e.g. first list gets 3, second list gets 2
    assert result == [[1, 2, 3], [4, 5]]

def test_split_into_k_lists_extra_distribution():
    elements = [1, 2, 3, 4, 5, 6, 7]
    k = 3
    result = split_into_k_lists(elements, k)
    # With base_size=2 and leftover=1, only the first sublist gets the extra element.
    assert result == [[1, 2, 3], [4, 5], [6, 7]]

def test_split_into_k_lists_k_equals_length():
    elements = [1, 2, 3]
    k = 3
    result = split_into_k_lists(elements, k)
    # Each should get exactly one element
    assert result == [[1], [2], [3]]

def test_split_into_k_lists_k_zero_raises_valueerror():
    with pytest.raises(ValueError, match="k must be a positive integer"):
        split_into_k_lists([1, 2, 3], 0)

def test_split_into_k_lists_k_larger_than_elements():
    with pytest.raises(ValueError, match="k cannot be greater than the number of elements"):
        split_into_k_lists([1, 2, 3], 4)

def test_is_truthy_simple_types():
    assert is_truthy(True) is True
    assert is_truthy(False) is False
    assert is_truthy(0) is False
    assert is_truthy(42) is True
    assert is_truthy("") is False
    assert is_truthy("non-empty") is True

def test_is_truthy_numpy_array():
    empty_arr = np.array([])
    non_empty_arr = np.array([1, 2, 3])
    assert is_truthy(empty_arr) is False
    assert is_truthy(non_empty_arr) is True

def test_is_truthy_collections():
    assert is_truthy([]) is False
    assert is_truthy([1]) is True
    assert is_truthy(()) is False
    assert is_truthy((1,)) is True

def test_create_exception_json():
    try:
        raise ValueError("Test error message")
    except ValueError as e:
        result = create_exception_json(e)
        assert result["error_type"] == "ValueError"
        assert "Test error message" in result["error_message"]
        assert "ValueError" in "".join(result["error_stack_trace"])  # stack trace check

def test_get_url_domain_basic():
    """Checks for removing leading www."""
    url = "https://www.example.com/some/path?query=param"
    domain = get_url_domain(url)
    assert domain == "example.com"

def test_get_url_domain_no_scheme():
    url = "www.EXAMPLE.com"
    # urlparse with no scheme can give an empty netloc, so the code returns an empty string.
    domain = get_url_domain(url)
    assert domain == ""

def test_get_url_domain_media_tenor():
    url = "https://media.tenor.com/file.gif"
    domain = get_url_domain(url)
    # Despite the docstring, the function actually returns the netloc, not a boolean check.
    assert domain == "media.tenor.com"

def test_get_all_links_empty():
    records = []
    result = get_all_links(records)
    assert result == []

def test_get_all_links_partial_keys():
    records = [
        {"commit": {"record": {}}},
        {"commit": {"record": {"embed": None}}},
        {"commit": {"record": {"embed": {"external": None}}}},
        {"commit": {"record": {"embed": {"external": {}}}}},
    ]
    result = get_all_links(records)
    assert result == [None, None, None, None]

def test_get_all_links_normal():
    records = [
        {
            "commit": {
                "record": {
                    "embed": {"external": {"uri": "http://example.com/link1"}}
                }
            }
        },
        {
            "commit": {
                "record": {
                    "embed": {"external": {"uri": "http://example.com/link2"}}
                }
            }
        },
    ]
    result = get_all_links(records)
    assert result == ["http://example.com/link1", "http://example.com/link2"]

def test_transform_dict_simple():
    data = {"hello": "world"}
    result = transform_dict(data)
    assert result == {"hello": "world"}

def test_transform_dict_remove_omitted():
    data = {"keep": "this", "remove": "that"}
    result = transform_dict(data, omitted_keys=["remove"])
    assert result == {"keep": "this"}

def test_transform_dict_nested():
    data = {
        "my~key": {
            "another/key": ["list_item", {"weird:key": "value"}]
        },
        "my/other": "valid"
    }
    result = transform_dict(data)
    # "my~key" => "my-key"
    # "weird:key" => "weird-key"
    assert "my-key" in result
    assert isinstance(result["my-key"], dict)
    assert "another/key" in result["my-key"]
    assert isinstance(result["my-key"]["another/key"], list)
    nested_dict = result["my-key"]["another/key"][1]
    assert "weird-key" in nested_dict

def test_transform_dict_list_handling():
    data = [
        {"my(key)": "val1", "ok_key": "val2"},
        "string_item",
        42,
        ["sublist_item"],
        {"A/B": {"subdict_key": "subdict_val"}},
    ]
    result = transform_dict(data)
    # Because "(" -> "-" and ")" -> "-", "my(key)" becomes "my-key-"
    assert result[0]["my-key-"] == "val1"
    # The other entries remain the same data type
    assert result[1] == "string_item"
    assert result[2] == 42
    # The sublist is also left alone
    assert result[3] == ["sublist_item"]
    # "A/B" is valid; slash is allowed, so it remains
    assert "A/B" in result[4]
    assert result[4]["A/B"]["subdict_key"] == "subdict_val"
