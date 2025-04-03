import numpy as np
from urllib.parse import urlparse
import re2
import traceback
import sys
from itertools import islice
from typing import List, Dict, Any

def is_likely_domain_or_url(term: str) -> bool:
    return '.' in term or term.startswith('http')

def extract_all_text_fields(records: List[Dict[str, Any]]) -> List[str]:
    """Extract all relevant textual content from records including post text, image alt text, and link preview text."""
    def safe_get(d: Dict, path: List[str]) -> Any:
        for key in path:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return None
        return d
    def gather_text(record: Dict[str, Any]) -> str:
        parts = []
        # Top-level post text
        text = record.get("text")
        if text:
            parts.append(text)
        # ALT text from images
        images = safe_get(record, ["embed", "images"])
        if images and isinstance(images, list):
            for img in images:
                alt = img.get("alt")
                if alt:
                    parts.append(alt)
        # Text from link previews
        external = safe_get(record, ["embed", "external"])
        if external and isinstance(external, dict):
            for key in ["title", "description"]:
                val = external.get(key)
                if val:
                    parts.append(val)
        return "\n".join(parts)
    return [gather_text(e["commit"]["record"]) for e in records]

def check_empty_string(value, threshold):
    """Return a boolean NumPy array where each element is True if '' is in that element."""
    if isinstance(value, np.ndarray):
        return np.array([(threshold in sub) for sub in value], dtype=bool)
    elif isinstance(value, list):
        return np.array([(threshold in sub) for sub in value], dtype=bool)
    else:
        raise ValueError("Input must be a list or NumPy array")


def is_list_of_lists(value):
    """Check if the value is a list of lists or a NumPy array of lists."""
    if isinstance(value, (list, np.ndarray)):
        if all(isinstance(sub, (list, np.ndarray)) for sub in value):
            return True
    return False


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


def get_url_domain(url: str) -> str:
    """Return domain, ignoring a leading 'www.', ensuring compatibility with both str and bytes."""
    parsed = urlparse(url)
    
    # Ensure netloc is always a string
    netloc = parsed.netloc
    if isinstance(netloc, bytes):
        netloc = netloc.decode("utf-8")
    domain = netloc.lower().removeprefix("www.")
    return domain


def get_all_links(records):
    url_sets = []
    for record in records:
        facet_urls = [feature["uri"] for facet in record["commit"]["record"].get("facets", []) for feature in facet.get("features", []) if feature.get("$type") == "app.bsky.richtext.facet#link"]
        primary_url = (
            (record["commit"]["record"].get("embed", {}) or {}).get("external", {})
            or {}
        ).get("uri")
        if primary_url:
            facet_urls.insert(0, primary_url)
        url_sets.append(facet_urls)
    return url_sets


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
