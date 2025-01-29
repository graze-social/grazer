import ray
from asyncio import Lock
from cachetools import LRUCache, TTLCache
from app.egress import Egress
from app.ray.timing_base import TimingBase, measure_time


@ray.remote(max_concurrency=1000)
class Cache(TimingBase):
    def __init__(self, key_prefix="ray_workers", batch_size=100):
        """
        Initialize the Cache actor.

        Args:
            key_prefix: Namespace prefix for all keys in the cache.
        """
        self._images = LRUCache(maxsize=10000)
        self._predictions = LRUCache(maxsize=40000)
        self._embeddings = LRUCache(maxsize=40000)
        self._resolved_dids = LRUCache(maxsize=40000)
        self._assets = TTLCache(maxsize=40000, ttl=60 * 60 * 3)
        self.key_prefix = key_prefix
        self.outputs = []
        self.batch_size = batch_size
        self.lock = Lock()
        super().__init__()

    async def report_output(self, data):
        """
        Append data to the outputs list and flush to Egress when the batch size is reached.

        Args:
            data: Data to append to the outputs list.
        """
        async with self.lock:
            self.outputs.append(data)
            if len(self.outputs) >= self.batch_size:
                await self.flush_to_egress()

    async def flush_to_egress(self):
        """
        Flush the accumulated outputs to Egress in a single transaction.
        """
        if not self.outputs:
            return
        await Egress.send_results(self.outputs, f"{self.key_prefix}:output")
        self.outputs = []

    @measure_time
    async def get_asset(self, asset_id):
        """Retrieve an asset from the cache."""
        return self._assets.get(asset_id)

    @measure_time
    async def cache_asset(self, asset_id, asset_data):
        """Cache an asset."""
        self._assets[asset_id] = asset_data

    @measure_time
    async def get_image(self, image_id):
        """Retrieve an image from the cache."""
        return self._images.get(image_id)

    @measure_time
    async def bulk_get_image(self, image_ids):
        """Retrieve an image from the cache."""
        return [self._images.get(image_id) for image_id in image_ids]

    @measure_time
    async def cache_image(self, image_id, image_data):
        """Cache an image."""
        self._images[image_id] = image_data

    @measure_time
    async def bulk_get_prediction(self, prediction_ids):
        """Get a cached prediction."""
        return [
            self._predictions.get(prediction_id) for prediction_id in prediction_ids
        ]

    @measure_time
    async def bulk_cache_prediction(self, prediction_ids, prediction_datas):
        """Cache a prediction."""
        for prediction_id, prediction_data in zip(prediction_ids, prediction_datas):
            self._predictions[prediction_id] = prediction_data

    @measure_time
    async def get_did(self, prediction_id):
        """Get a cached prediction."""
        return self._resolved_dids.get(prediction_id)

    @measure_time
    async def cache_did(self, prediction_id, prediction_data):
        """Cache a prediction."""
        self._resolved_dids[prediction_id] = prediction_data

    @measure_time
    async def bulk_get_embedding(self, keys):
        """Retrieve an image from the cache."""
        return [self._embeddings.get(key) for key in keys]

    @measure_time
    async def bulk_cache_embedding(self, keys, values):
        """Retrieve an image from the cache."""
        for key, value in zip(keys, values):
            self._embeddings[key] = value
