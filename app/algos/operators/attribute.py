import re2
import numpy as np
from app.helpers import get_all_links, get_url_domain
from app.algos.base import BaseParser
from app.logic_evaluator import LogicEvaluator

EMBED_TYPES = {
    "image": "app.bsky.embed.images",
    "link": "app.bsky.embed.external",
    "post": "app.bsky.embed.record",
    "image_group": "app.bsky.embed.recordWithMedia",
    "video": "app.bsky.embed.video",
    "gif": "",
}


def resolve_path_batch(records, path):
    """
    Resolve a JSONPath-like path on a batch of JSON-like objects, with support for `*`, bracket syntax, and mixed paths.

    Args:
        records (list[dict]): List of root JSON objects to traverse.
        path (str): Dot/bracket notation path, e.g., "langs[0]", "embed.images[*].alt", or "embed.images[0].alt".

    Returns:
        list: Singular values for non-wildcard paths, or a 2D array for wildcard paths.
    """

    def resolve_path_single(obj, parts):
        """
        Resolve parts of the path recursively, handling keys, indices, and `*` operator.
        """
        if not parts:
            return [obj]
        part = parts[0]
        # Handle wildcard `*` or `[*]`
        if part in ("*", "[*]"):
            if isinstance(obj, list):
                return [
                    item
                    for sublist in (
                        resolve_path_single(item, parts[1:]) for item in obj
                    )
                    for item in sublist
                ]
            elif isinstance(obj, dict):
                return [
                    item
                    for sublist in (
                        resolve_path_single(value, parts[1:]) for value in obj.values()
                    )
                    for item in sublist
                ]
            else:
                return []  # `*` on non-iterable returns no results.
        # Handle key/index navigation
        match = re2.match(
            r"([a-zA-Z_]\w*)\[(\d+)\]$", part
        )  # Match key with index, e.g., "key[0]"
        try:
            if match:
                key, index = match.groups()
                return resolve_path_single(obj[key][int(index)], parts[1:])
            elif part.isdigit():  # Handle numeric index alone, like "[0]"
                index = int(part)
                return resolve_path_single(obj[index], parts[1:])
            else:
                return resolve_path_single(obj[part], parts[1:])
        except (TypeError, IndexError, KeyError):
            return []  # Gracefully handle invalid paths.

    # Split path into parts, handling both `.` and bracket notations
    parts = re2.split(r"\.(?![^\[]*\])|\[|\]", path)
    parts = [p for p in parts if p]  # Remove empty parts from splitting

    results = []
    for record in records:
        try:
            obj = record["commit"]["record"]  # Adjust root if necessary
            resolved = resolve_path_single(obj, parts)
            results.append(
                resolved if resolved else [None]
            )  # Use empty string for no results
        except (TypeError, KeyError):
            results.append([""])  # Gracefully handle invalid paths

    # Determine if path contains wildcard
    is_wildcard = "*" in path or "[*]" in path
    if is_wildcard:
        # Return a 2D array where each row corresponds to one record
        max_length = max(len(row) for row in results)
        padded_results = [row + [""] * (max_length - len(row)) for row in results]
        return np.array(padded_results, dtype=object)
    else:
        # Return a list of singular values for non-wildcard paths
        return np.array([row[0] if row else "" for row in results])


class AttributeParser(BaseParser):
    async def attribute_compare(self, records, field_selector, operator, target_value):
        """
        Resolve the attribute path for a batch of records and apply a comparison operation.

        Args:
            records (list[dict]): Batch of JSON-like records to evaluate.
            field_selector (dict): Contains the "var" key for the field path.
            operator (str): The comparison operator (e.g., "==", ">", etc.).
            target_value (Any): The value to compare against.

        Returns:
            np.ndarray: Boolean array indicating whether each record satisfies the comparison.
        """
        values = resolve_path_batch(records, field_selector)
        if target_value is None and "*" in field_selector:
            target_value = [None]
        return np.array(
            await LogicEvaluator.compare(values, operator, target_value), dtype=bool
        )

    async def post_type(self, records, operator, post_names):
        matches_by_type = {}
        set_post_names = set(post_names)
        if "quote" in post_names:
            values = resolve_path_batch(records, "embed.$type")
            matches_by_type["quote"] = np.array(
                await LogicEvaluator.compare(values, "==", EMBED_TYPES["post"]),
                dtype=bool,
            )

        if "reply" in post_names:
            values = resolve_path_batch(records, "reply")
            matches_by_type["reply"] = np.array(
                await LogicEvaluator.compare(values, "!=", None),
                dtype=bool,
            )

        # Convert boolean arrays into lists of type labels
        labeled_types = []
        for i in range(len(records)):
            types = [key if matches_by_type.get(key, np.zeros(len(records), dtype=bool))[i] else "not"
                     for key in matches_by_type]
            filtered_types = [t for t in types if t != "not"]
            labeled_types.append(filtered_types if filtered_types else ["not"])

        values = np.array([bool(len(set(l)&set_post_names)) for l in labeled_types])
        if operator in ["in", "=="]:
            return values
        elif operator in ["not_in", "!="]:
            return ~values

    async def embed_type(self, records, operator, embed_name):
        """
        Resolve the attribute path for a batch of records and apply a comparison operation.

        Args:
            records (list[dict]): Batch of JSON-like records to evaluate.
            field_selector (dict): Contains the "var" key for the field path.
            operator (str): The comparison operator (e.g., "==", ">", etc.).
            target_value (Any): The value to compare against.

        Returns:
            np.ndarray: Boolean array indicating whether each record satisfies the comparison.
        """
        if embed_name == "gif" or embed_name == "link":
            values = resolve_path_batch(records, "embed.$type")
            is_link_type = await LogicEvaluator.compare(
                values, "==", EMBED_TYPES["link"]
            )
            gif_domains = np.array(
                [get_url_domain(e) == "media.tenor.com" for e in get_all_links(records)]
            )
            if embed_name == "gif":
                if operator == "==":
                    return gif_domains
                else:
                    return ~gif_domains
            else:
                if operator == "==":
                    return is_link_type & ~gif_domains
                else:
                    return ~is_link_type & ~gif_domains
        elif embed_name == "video":
            embed_types = resolve_path_batch(records, "embed.$type")
            embed_media_type = resolve_path_batch(records, "embed.media.$type")
            values = np.array(
                [
                    (
                        "app.bsky.embed.video"
                        if e[0] == "app.bsky.embed.video"
                        or e[1] == "app.bsky.embed.video"
                        else None
                    )
                    for e in zip(embed_types, embed_media_type)
                ]
            )
            return np.array(
                await LogicEvaluator.compare(values, operator, EMBED_TYPES[embed_name]),
                dtype=bool,
            )
        elif embed_name == "image_group":
            embed_types = resolve_path_batch(records, "embed.$type")
            embed_media_type = resolve_path_batch(records, "embed.media.$type")
            values = np.array(
                [
                    (
                        "app.bsky.embed.recordWithMedia"
                        if e[0] and e[1] == "app.bsky.embed.images"
                        else None
                    )
                    for e in zip(embed_types, embed_media_type)
                ]
            )
            return np.array(
                await LogicEvaluator.compare(values, operator, EMBED_TYPES[embed_name]),
                dtype=bool,
            )
        else:
            values = resolve_path_batch(records, "embed.$type")
            return np.array(
                await LogicEvaluator.compare(values, operator, EMBED_TYPES[embed_name]),
                dtype=bool,
            )

    async def register_operations(self, logic_evaluator):
        # Register attribute comparison operation
        await logic_evaluator.add_operation("attribute_compare", self.attribute_compare)
        await logic_evaluator.add_operation("embed_type", self.embed_type)
        await logic_evaluator.add_operation("post_type", self.post_type)
