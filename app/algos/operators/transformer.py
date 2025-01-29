import pdb
from hashlib import sha256
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.logic_evaluator import LogicEvaluator
from app.algos.base import BaseParser


class TransformerParser(BaseParser):
    async def get_embedding(self, model_name, texts):
        if isinstance(texts, str):
            texts = [texts]  # Normalize to list for consistent handling
        # Step 1: Check cache and identify uncached texts
        cached_embeddings = {}
        cache_keys = []
        uncached_texts = []
        cache_keys = [
            sha256(f"{model_name}:{text}".encode()).hexdigest() for text in texts
        ]
        cached_values = await self.cache.bulk_get_embedding.remote(cache_keys)
        for cache_key, text, cached_value in zip(cache_keys, texts, cached_values):
            if cached_value is not None:
                cached_embeddings[text] = cached_value
            else:
                cache_keys.append(cache_key)
                uncached_texts.append(text)
        all_embeddings = cached_embeddings.copy()
        # Step 2: Handle uncached texts
        if uncached_texts:
            batch_results = await self.gpu_embedding_worker.get_embedding.remote(
                model_name, cache_keys, uncached_texts
            )
            for text, batch_result in zip(uncached_texts, batch_results):
                all_embeddings[text] = batch_result
        # Step 3: Return results in the same order as input
        return np.array([all_embeddings[text] for text in texts])

    async def get_ml_scores(
        self, records, field_selector, model_params, comparator, threshold
    ):
        sampled_texts = [
            (e["commit"]["record"].get(field_selector, "") or "") for e in records
        ]
        model_name = model_params["model_name"]
        anchor_text = model_params["anchor_text"]
        embedding = await self.get_embedding(model_name, sampled_texts)
        anchor_embedding = await self.get_embedding(model_name, anchor_text)
        return cosine_similarity(embedding, anchor_embedding.reshape(1, -1)).flatten()

    async def text_similarity_operator(
        self, records, field_selector, model_params, comparator, threshold
    ):
        similarities = await self.get_ml_scores(
            records, field_selector, model_params, comparator, threshold
        )
        return await LogicEvaluator.compare(similarities, comparator, threshold)

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "text_similarity", self.text_similarity_operator, True
        )
