import re2
import numpy as np

from app.algos.base import BaseParser
from app.helpers import get_all_links, get_url_domain


class EntityParser(BaseParser):
    @staticmethod
    def get_all_items(records, facet_type, accessor):
        items = []
        for record in records:
            items.append(
                {
                    e.get(accessor)
                    for e in [
                        item
                        for row in [
                            e.get("features")
                            for e in (
                                record["commit"]["record"].get("facets", []) or []
                            )
                        ]
                        for item in row
                    ]
                    if e.get("$type") == facet_type and e.get(accessor)
                }
            )
        return items

    @staticmethod
    def get_all_langs(records):
        return [
            set(record["commit"]["record"].get("langs", []) or []) for record in records
        ]

    @staticmethod
    def get_all_hashtags(records):
        secret_tags = [
            set(record["commit"]["record"].get("tags", []) or []) for record in records
        ]
        hashtags = EntityParser.get_all_items(
            records, "app.bsky.richtext.facet#tag", "tag"
        )
        return [
            {e.lower() for e in hashtags_set} | {e.lower() for e in secret_tags_set}
            for hashtags_set, secret_tags_set in zip(hashtags, secret_tags)
        ]

    @staticmethod
    def get_all_mentions(records):
        return EntityParser.get_all_items(
            records, "app.bsky.richtext.facet#mention", "did"
        )

    @staticmethod
    def get_all_urls(records):
        embed_urls = [
            set(
                [
                    (
                        (record["commit"]["record"].get("embed", {}) or {}).get(
                            "external", {}
                        )
                        or {}
                    ).get("uri")
                ]
            )
            for record in records
        ]
        links = EntityParser.get_all_items(
            records, "app.bsky.richtext.facet#link", "uri"
        )
        return [
            embed_urls_set | links_set
            for embed_urls_set, links_set in zip(embed_urls, links)
        ]

    @staticmethod
    def get_all_labels(records):
        return [
            {
                e.get("val")
                for e in (
                    (record["commit"]["record"].get("labels", {}) or {}).get(
                        "values", []
                    )
                    or []
                )
            }
            for record in records
        ]

    async def get_resolved_values(self, entity_type, values):
        resolved_values = values
        users = None
        if entity_type == "hashtags":
            resolved_values = {e.replace("#", "") for e in values}
        elif entity_type == "mentions":
            list_pattern = (
                r"https://bsky\.app/profile/([-a-zA-Z0-9\.]+)/lists/([a-zA-Z0-9_-]+)"
            )
            starter_pack_pattern = (
                r"https://bsky\.app/starter-pack/([-a-zA-Z0-9\.]+)/([a-zA-Z0-9_-]+)"
            )
            resolved_values = set()
            for value in values:
                if re2.match(list_pattern, value):
                    users = await self.network_worker.get_list.remote(value)
                    for user in users:
                        resolved_values.add(user)
                elif re2.match(starter_pack_pattern, value):
                    users = await self.network_worker.get_starter_pack.remote(value)
                    for user in users:
                        resolved_values.add(user)
                else:
                    resolved_value = (
                        await self.network_worker.get_or_set_handle_did.remote(value)
                    )
                    resolved_values.add(resolved_value)
        elif entity_type == "domains":
            #Kind of dumb but safest possible way to clean up potentially bad input
            resolved_values = {e.removeprefix("https://").removeprefix("http://").removeprefix("www.").removesuffix("/") for e in values}
        return resolved_values

    async def matches_entities(self, records, entity_type, values, insensitive=True):
        """
        Returns a NumPy array of booleans indicating whether the field in each
        record matches the pattern.
        """
        values = await self.get_resolved_values(entity_type, values)
        comparison_frozenset = frozenset([e.lower() for e in values if e])
        if entity_type == "domains":
            record_values = [set([get_url_domain(ee) for ee in e]) for e in get_all_links(records)]
        elif entity_type == "labels":
            record_values = EntityParser.get_all_labels(records)
        else:
            record_values = getattr(EntityParser, f"get_all_{entity_type}")(records)
        return np.array([bool(s & comparison_frozenset) for s in record_values])

    async def excludes_entities(self, records, entity_type, values, insensitive=True):
        """
        Returns a NumPy array of booleans indicating whether the field in each
        record does NOT match the pattern.
        """
        response = await self.matches_entities(
            records, entity_type, values, insensitive
        )
        return ~response

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation("entity_matches", self.matches_entities)
        await logic_evaluator.add_operation("entity_excludes", self.excludes_entities)
