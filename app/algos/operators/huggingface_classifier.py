import numpy as np
from hashlib import sha256
from app.logic_evaluator import LogicEvaluator
from app.algos.base import BaseParser
from app.helpers import extract_all_text_fields

class HuggingfaceClassifierParser(BaseParser):
    async def precache_text_predictions(self, texts, model_name):
        # Generate cache keys for all texts
        cache_keys = [
            sha256(f"{model_name}:{text}".encode()).hexdigest() for text in texts
        ]
        # Use bulk_get_prediction to retrieve cached values
        cache_results = await self.cache.bulk_get_prediction.remote(cache_keys)
        cached_probs = {}
        uncached_keys = []
        uncached_texts = []
        # Process cache results
        for text, cache_key, cached_value in zip(texts, cache_keys, cache_results):
            if cached_value:
                cached_probs[text] = cached_value
            else:
                uncached_keys.append(cache_key)
                uncached_texts.append(text)
        return cached_probs, uncached_keys, uncached_texts

    async def get_probabilities(self, texts, model_name, batch_size=200):
        if isinstance(texts, str):
            texts = [texts]
        cached_probs, uncached_keys, uncached_texts = (
            await self.precache_text_predictions(texts, model_name)
        )
        all_out_probs = cached_probs.copy()

        if uncached_texts:
            results = (
                await self.gpu_classifier_worker.huggingface_classifier_classify.remote(
                    model_name, uncached_keys, uncached_texts, self.LABEL_MAP
                )
            )
            # Update all_out_probs and the primary cache
            if results:
                for text, result in zip(uncached_texts, results):
                    all_out_probs[text] = result
        return [all_out_probs[text] for text in texts]

    async def get_ml_scores(self, records, category, comparator, threshold):
        sampled_texts = extract_all_text_fields(records)
        probabilities = await self.get_probabilities(sampled_texts, self.MODEL_NAME)
        return np.array([e.get(category, 0) for e in probabilities])

    async def classifier_operator(self, records, category, comparator, threshold):
        scores = await self.get_ml_scores(records, category, comparator, threshold)
        bools = LogicEvaluator.compare(scores, comparator, threshold)
        return bools.tolist()
