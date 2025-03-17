import numpy as np
from urllib.parse import urlparse
import re2
import traceback
import sys
from itertools import islice
from typing import List


def chunk(iterable, size):
    """Yield successive chunks of a given size from an iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


def dict_to_sorted_string(d: dict) -> str:
    """
    Convert a dictionary into a structured string format "key:value__key2:value2",
    sorted alphabetically by key names.
    """
    sorted_items = sorted(d.items())  # Sort items by key
    return "__".join(f"{key}:{value}" for key, value in sorted_items)


def split_into_k_lists(elements: List, k: int) -> List[List]:
    """
    Splits a list into k even-length lists. Extra elements are distributed to the first few lists.

    Args:
        elements (List): The input list to be split.
        k (int): The number of lists to split into.

    Returns:
        List[List]: A list of k lists with elements distributed as evenly as possible.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > len(elements):
        raise ValueError("k cannot be greater than the number of elements.")

    # Calculate base size of each sublist and the number of extra elements
    n = len(elements)
    base_size = n // k
    extra = n % k

    # Generate the k lists
    result = []
    start = 0
    for i in range(k):
        end = (
            start + base_size + (1 if i < extra else 0)
        )  # Add one extra element to the first 'extra' lists
        result.append(elements[start:end])
        start = end

    return result


def is_truthy(value):
    """
    Safely checks if a value is truthy, avoiding ambiguous truth value errors.
    """
    if isinstance(value, (np.ndarray, list, tuple)):
        # Check for empty array or collection
        return bool(value.size if isinstance(value, np.ndarray) else len(value))
    return bool(value)


def create_exception_json(exc: Exception) -> dict:
    """
    Converts an exception into a JSON-serializable dictionary.
    """
    exc_type, exc_value, exc_tb = sys.exc_info()

    return {
        "error_type": exc_type.__name__ if exc_type else None,
        "error_message": str(exc_value),
        "error_stack_trace": traceback.format_exception(exc_type, exc_value, exc_tb),
    }


def get_url_domain(url: str) -> bool:
    """Return domain, ignoring a leading 'www.'."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")
    return domain


def get_all_links(records):
    return [
        (
            (record["commit"]["record"].get("embed", {}) or {}).get("external", {})
            or {}
        ).get("uri")
        for record in records
    ]


def transform_dict(data, omitted_keys=None):
    """
    Transforms a nested dictionary by:
    1. Replacing any non A-Za-z, underscore, or slash character in keys with '-'.
    2. Removing any values corresponding to keys in the omitted_keys list.

    Args:
        data (dict or list): The nested dictionary or list to transform.
        omitted_keys (list): A list of keys to omit from the result.

    Returns:
        dict or list: The transformed nested structure.
    """
    if omitted_keys is None:
        omitted_keys = []

    def clean_key(key):
        """Replaces non A-Za-z, underscore, or slash characters in key names with '-'."""
        return re2.sub(r"[^A-Za-z_/]", "-", key)

    if isinstance(data, dict):
        transformed = {}
        for key, value in data.items():
            clean_key_name = clean_key(key)
            if clean_key_name not in omitted_keys:
                transformed[clean_key_name] = transform_dict(value, omitted_keys)
        return transformed
    elif isinstance(data, list):
        return [transform_dict(item, omitted_keys) for item in data]
    else:
        return data
