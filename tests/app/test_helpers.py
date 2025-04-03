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
    check_empty_string,
    is_list_of_lists,
    extract_all_text_fields,
)
def make_record(embed=None, text=None):
    return {
        "commit": {
            "record": {
                **({"text": text} if text is not None else {}),
                **({"embed": embed} if embed is not None else {})
            }
        }
    }

def test_text_only():
    records = [make_record(text="Hello world!")]
    assert extract_all_text_fields(records) == ["Hello world!"]

def test_text_and_image_alts():
    records = [make_record(
        text="Main text",
        embed={
            "images": [
                {"alt": "Image 1 alt"},
                {"alt": "Image 2 alt"},
                {"alt": ""},  # should be ignored
                {}            # missing alt
            ]
        }
    )]
    assert extract_all_text_fields(records) == ["Main text\nImage 1 alt\nImage 2 alt"]

def test_external_link_only():
    records = [make_record(
        embed={
            "external": {
                "title": "Article Title",
                "description": "Article description here."
            }
        }
    )]
    assert extract_all_text_fields(records) == ["Article Title\nArticle description here."]

def test_external_link_partial():
    records = [make_record(embed={"external": {"description": "Just description"}})]
    assert extract_all_text_fields(records) == ["Just description"]

def test_all_sources_present():
    records = [make_record(
        text="Text here",
        embed={
            "images": [{"alt": "Alt text"}],
            "external": {
                "title": "Title",
                "description": "Desc"
            }
        }
    )]
    assert extract_all_text_fields(records) == ["Text here\nAlt text\nTitle\nDesc"]

def test_missing_everything():
    records = [make_record()]
    assert extract_all_text_fields(records) == [""]

def test_malformed_embed_structures():
    records = [make_record(embed={"images": "not a list"})]
    assert extract_all_text_fields(records) == [""]

def test_multiple_records():
    records = [
        make_record(text="First"),
        make_record(embed={"images": [{"alt": "Second alt"}]}),
        make_record(embed={"external": {"title": "Third"}})
    ]
    assert extract_all_text_fields(records) == ["First", "Second alt", "Third"]

def test_check_empty_string_basic():
    value = np.array([["", "test"], ["hello", "world"]], dtype=object)
    threshold = ""
    result = check_empty_string(value, threshold)
    expected = np.array([True, False], dtype=bool)
    assert np.array_equal(result, expected)

def test_check_empty_string_all_true():
    value = np.array([["", ""], ["", ""]], dtype=object)
    threshold = ""
    result = check_empty_string(value, threshold)
    expected = np.array([True, True], dtype=bool)
    assert np.array_equal(result, expected)

def test_check_empty_string_all_false():
    value = np.array([["foo", "bar"], ["baz", "qux"]], dtype=object)
    threshold = ""
    result = check_empty_string(value, threshold)
    expected = np.array([False, False], dtype=bool)
    assert np.array_equal(result, expected)

def test_check_empty_string_mixed_data_types():
    value = np.array([["", 1], [2, ""]], dtype=object)
    threshold = ""
    result = check_empty_string(value, threshold)
    expected = np.array([True, True], dtype=bool)
    assert np.array_equal(result, expected)

def test_check_empty_string_empty_list():
    value = np.array([], dtype=object)
    threshold = ""
    result = check_empty_string(value, threshold)
    expected = np.array([], dtype=bool)
    assert np.array_equal(result, expected)

def test_check_empty_string_raises_valueerror():
    with pytest.raises(ValueError, match="Input must be a list or NumPy array"):
        check_empty_string(42, "")

def test_is_list_of_lists_true():
    assert is_list_of_lists([['a', 'b'], ['c', 'd']]) is True
    assert is_list_of_lists(np.array([[1, 2], [3, 4]])) is True

def test_is_list_of_lists_false():
    assert is_list_of_lists(['a', 'b', 'c']) is False
    assert is_list_of_lists(np.array(['a', 'b', 'c'], dtype=object)) is False
    assert is_list_of_lists([1, 2, 3]) is False
    assert is_list_of_lists(np.array([1, 2, 3])) is False

def test_is_list_of_lists_edge_cases():
    assert is_list_of_lists([[[]]]) is True  # List of lists, even if empty
    assert is_list_of_lists(np.array([[], []], dtype=object)) is True
    assert is_list_of_lists([[], "not a list"]) is False

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
    assert result == [[], [], [], []]

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
    assert result == [["http://example.com/link1"], ["http://example.com/link2"]]

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
