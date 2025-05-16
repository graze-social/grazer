import random
import ray
import time
from cachetools import LRUCache
from typing import Any
from app.ray.timing_base import TimingBase, measure_time
from app.models.huggingface_classifier import HuggingfaceClassifier
from app.models.image_nsfw_classifier import ImageNSFWClassifier
from app.models.image_arbitrary_classifier import ImageArbitraryClassifier
from app.models.text_embedder import TextEmbedder
from app.models.text_arbitrary_classifier import TextArbitraryClassifier

from app.logger import logger

@ray.remote(num_gpus=1, max_task_retries=-1, max_restarts=-1)
class GPUWorker(TimingBase):
    @property
    def network_worker(self):
        """
        Choose a NetworkWorker at random from the available network workers.
        Returns:
            A randomly selected NetworkWorker reference.
        """
        if not self.network_workers:
            raise ValueError("No network workers are available")
        return random.choice(self.network_workers)

    def __init__(self, network_workers, cache):
        self.network_workers = network_workers
        self.cache = cache
        self._models = LRUCache(maxsize=50)
        self._probability_models = LRUCache(maxsize=50)
        super().__init__()

    @measure_time
    def _get_or_load_model(self, model_name: str, loader_function: callable) -> Any:
        if model_name in self._models:
            return self._models[model_name]
        self._models[model_name] = loader_function(model_name)
        return self._models[model_name]

    @measure_time
    def get_vector_model(self, model_name: str) -> Any:
        return self._get_or_load_model(model_name, TextEmbedder.get_model)

    @measure_time
    def get_text_arbitrary_model(self, model_name: str) -> Any:
        return self._get_or_load_model(model_name, TextArbitraryClassifier.get_model)

    @measure_time
    def get_huggingface_classifier(self, model_name: str) -> Any:
        return self._get_or_load_model(model_name, HuggingfaceClassifier.get_model)

    @measure_time
    def get_image_nsfw_model(self, model_name: str) -> Any:
        return self._get_or_load_model(model_name, ImageNSFWClassifier.get_model)

    @measure_time
    def get_image_arbitrary_model(self, model_name: str) -> Any:
        return self._get_or_load_model(model_name, ImageArbitraryClassifier.get_model)

    @measure_time
    def get_sentence_embedding_dimension(self, model_name: str) -> int:
        return self.get_vector_model(model_name).get_sentence_embedding_dimension()

    @measure_time
    async def huggingface_classifier_classify(
        self, model_name, cache_keys, texts, label_map, batch_size=200
    ):
        predictions = await HuggingfaceClassifier.compute_predictions(
            self.get_huggingface_classifier(model_name), texts, label_map, batch_size
        )
        await self.cache.bulk_cache_prediction.remote(cache_keys, predictions)
        return predictions

    @measure_time
    async def image_nsfw_classify(self, model_name, cache_keys, images, batch_size=10):
        predictions = await ImageNSFWClassifier.compute_predictions(
            self.get_image_nsfw_model(model_name), images, batch_size
        )
        await self.cache.bulk_cache_prediction.remote(cache_keys, predictions)
        return predictions

    @measure_time
    async def image_arbitrary_classify(
        self, model_name, cache_keys, images, category, batch_size=10
    ):
        predictions = await ImageArbitraryClassifier.compute_predictions(
            self.get_image_arbitrary_model(model_name), images, category, batch_size
        )
        await self.cache.bulk_cache_prediction.remote(cache_keys, predictions)
        return predictions

    @measure_time
    async def get_embedding(self, model_name, cache_keys, texts, batch_size=200):
        predictions = await TextEmbedder.compute_predictions(
            self.get_vector_model(model_name), texts, batch_size
        )
        await self.cache.bulk_cache_embedding.remote(cache_keys, predictions)
        return predictions

    async def text_arbitrary_classify(
        self, model_name, cache_keys, texts, labels, multi_label
    ):
        predictions = await TextArbitraryClassifier.compute_predictions(
            self.get_text_arbitrary_model(model_name), texts, labels, multi_label
        )
        await self.cache.bulk_cache_prediction.remote(cache_keys, predictions)
        return predictions

    async def run(self):
        # Keep the script running to maintain the actor
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info(f"CPU Worker worker stopped.")
