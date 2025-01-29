import random
import asyncio
import numpy as np
from hashlib import sha256


def get_cache_key(model_name, url, category):
    return sha256(f"{model_name}:{url}:{category}".encode()).hexdigest()


class BaseParser:
    @classmethod
    async def initialize(cls, algo_manager):
        instance = cls()
        await instance.register_operations(algo_manager.logic_evaluator)
        instance.cache = algo_manager.cache
        instance.gpu_embedding_workers = algo_manager.gpu_embedding_workers
        instance.gpu_classifier_workers = algo_manager.gpu_classifier_workers
        instance.network_workers = algo_manager.network_workers
        return instance

    async def register_operations(self):
        """Register operations specific to each parser."""
        raise NotImplementedError("Subclasses must implement register_operations")

    @property
    def network_worker(self):
        if not self.network_workers:
            raise ValueError("No network_workers are available")
        return random.choice(self.network_workers)

    @property
    def gpu_embedding_worker(self):
        if not self.gpu_embedding_workers:
            raise ValueError("No gpu_embedding_workers are available")
        return random.choice(self.gpu_embedding_workers)

    @property
    def gpu_classifier_worker(self):
        if not self.gpu_classifier_workers:
            raise ValueError("No gpu_classifier_workers are available")
        return random.choice(self.gpu_classifier_workers)


class ImageParser(BaseParser):
    async def fetch_image(self, url):
        await self.network_worker.fetch_image.remote(url)

    async def fetch_images(self, urls):
        tasks = [self.fetch_image(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def precache_predictions(self, image_urls, category):
        # Generate cache keys for the URLs
        keys = [get_cache_key(self.MODEL_NAME, url, category) for url in image_urls]
        # Use bulk_get_prediction to retrieve cached values
        cache_results = await self.cache.bulk_get_prediction.remote(keys)
        cached_probs = {}
        uncached_urls = []
        # Process cache results
        for url, key, cached_value in zip(image_urls, keys, cache_results):
            if cached_value:
                cached_probs[url] = cached_value
            else:
                uncached_urls.append(url)
        return cached_probs, uncached_urls

    async def get_probabilities(self, image_urls, category, batch_size=10):
        """
        Returns a list of dictionaries for each image_url.
        We'll include two entries in each dictionary:
          {category: probability_of_category, "none": probability_of_none}
        """
        if isinstance(image_urls, str):
            image_urls = [image_urls]

        # Use cache
        cached_probs, uncached_urls = await self.precache_predictions(
            image_urls, category
        )
        all_out_probs = cached_probs.copy()

        # Fetch and compute if not in cache
        if uncached_urls:
            images = await self.fetch_images(uncached_urls)
            valid_images = []
            url_indices = []
            cache_keys = []
            for url, img in zip(uncached_urls, images):
                if isinstance(img, Exception):
                    print(f"Skipping image at {url} due to error: {img}")
                elif img is not None:
                    valid_images.append(img)
                    url_indices.append(url)
                    cache_keys.append(get_cache_key(self.MODEL_NAME, url, category))
            if valid_images:
                probs = await self.probability_function(cache_keys, valid_images, category)
                for single_url, single_probs in zip(url_indices, probs):
                    category_prob_map = {
                        category: single_probs[0],
                        f"not_{category}": single_probs[1],
                    }
                    all_out_probs[single_url] = category_prob_map
        return {
            url: (all_out_probs.get(url, {}) or {}).get(category) for url in image_urls
        }

    async def get_all_image_urls(self, records):
        urls = {}
        for record in records:
            embed = record.get("commit", {}).get("record", {}).get("embed")
            if not embed:
                continue

            did = record["did"]

            if embed["$type"] == "app.bsky.embed.images":
                image_urls = []
                for image in embed.get("images", []):
                    cid = image.get("image", {}).get("ref", {}).get("$link")
                    if cid:
                        image_urls.append(
                            f"https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}@jpeg"
                        )
                urls[record["commit"]["cid"]] = image_urls

            elif embed["$type"] == "app.bsky.embed.recordWithMedia":
                media = embed.get("media", {})
                if media["$type"] == "app.bsky.embed.images":
                    image_urls = []
                    for image in media.get("images", []):
                        cid = image.get("image", {}).get("ref", {}).get("$link")
                        if not cid:
                            cid = (
                                media.get("external", {})
                                .get("thumb", {})
                                .get("ref", {})
                                .get("$link")
                            )
                        image_urls.append(
                            f"https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}@jpeg"
                        )
                    urls[record["commit"]["cid"]] = image_urls

        return urls

    async def get_ml_scores(
        self, records, category, comparator, threshold, default_value=True
    ):
        image_urls_by_cid = await self.get_all_image_urls(records)
        all_cids = [e["commit"]["cid"] for e in records]
        # Flatten all URLs for batch processing
        all_urls = [url for urls in image_urls_by_cid.values() for url in urls]
        all_probs = await self.get_probabilities(all_urls, category)
        # Map probabilities back to records
        cid_to_scores = {}
        for cid, urls in image_urls_by_cid.items():
            probs = [all_probs.get(url) for url in urls if all_probs.get(url)]
            if len(probs) == 1:
                cid_to_scores[cid] = probs[0]
            elif len(probs) > 1:
                cid_to_scores[cid] = np.max(probs)
        return np.array(
            [
                (cid_to_scores.get(c, 0.0) or 0.0)
                for c in all_cids
                if cid_to_scores.get(c)
            ]
        )

    async def classifier_operator(
        self, records, category, comparator, threshold, default_value=True
    ):
        from app.logic_evaluator import LogicEvaluator

        image_urls_by_cid = await self.get_all_image_urls(records)
        all_cids = [e["commit"]["cid"] for e in records]
        # Flatten all URLs for batch processing
        all_urls = [url for urls in image_urls_by_cid.values() for url in urls]
        all_probs = await self.get_probabilities(all_urls, category)
        # Map probabilities back to records
        cid_to_scores = {}
        for cid, urls in image_urls_by_cid.items():
            probs = [all_probs.get(url) for url in urls if all_probs.get(url)]
            if len(probs) == 1:
                cid_to_scores[cid] = probs[0]
            elif len(probs) > 1:
                cid_to_scores[cid] = np.max(probs)
        scores = np.array(
            [cid_to_scores.get(c) for c in all_cids if cid_to_scores.get(c)]
        )
        bools = await LogicEvaluator.compare(scores, comparator, threshold)
        out_bools = []
        i = 0
        for c in all_cids:
            if cid_to_scores.get(c) is not None:
                out_bools.append(bools[i])
                i += 1
            else:
                if not image_urls_by_cid.get(c, []):
                    out_bools.append(True)
                else:
                    out_bools.append(False)
        return np.array(out_bools)

    async def precache_all_images(self, records):
        image_urls_by_cid = await self.get_all_image_urls(records)
        all_urls = [url for urls in image_urls_by_cid.values() for url in urls]
        await self.precache_images(all_urls)

    async def precache_images(self, image_urls):
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        # Use bulk_get_image to retrieve cached values for all image URLs
        cache_results = await self.cache.bulk_get_image.remote(
            [f"images_{url}" for url in image_urls]
        )
        # Identify uncached URLs (those with None in cache_results)
        uncached_urls = [
            url for url, result in zip(image_urls, cache_results) if result is None
        ]
        # Fetch images for the uncached URLs
        if uncached_urls:
            await self.fetch_images(uncached_urls)
