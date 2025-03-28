import numpy as np
from hashlib import sha256
from app.logic_evaluator import LogicEvaluator
from app.models.text_arbitrary_classifier import TextArbitraryClassifier
from app.algos.operators.huggingface_classifier import HuggingfaceClassifierParser


def get_cache_key(model_name, text, labels, multi_label):
    joined_labels = "|".join(labels)
    return sha256(
        f"{model_name}:{text}:{joined_labels}:{multi_label}".encode()
    ).hexdigest()


class TextArbitraryParser(HuggingfaceClassifierParser):
    MODEL_NAME = TextArbitraryClassifier.MODEL_NAME

    async def precache_text_predictions(self, texts, labels, multi_label):
        cached_probs = {}
        uncached_keys = []
        uncached_texts = []

        # Generate cache keys for all texts
        cache_keys = [
            get_cache_key(self.MODEL_NAME, text, labels, multi_label) for text in texts
        ]
        # Use bulk_get_prediction to retrieve cached values
        cache_results = await self.cache.bulk_get_prediction.remote(cache_keys)
        # Process cache results
        for text, cache_key, cached_value in zip(texts, cache_keys, cache_results):
            if cached_value:
                label_probs = dict(zip(cached_value["labels"], cached_value["scores"]))
                cached_probs[text] = label_probs
            else:
                uncached_keys.append(cache_key)
                uncached_texts.append(text)
        return cached_probs, uncached_keys, uncached_texts

    async def get_predictions(self, texts, labels, multi_label=False, batch_size=200):
        """
        Perform zero-shot classification on a batch of texts with given labels.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Set to none since '' and None are non-interpretable by this model
        texts = [e or 'none' for e in texts]
        # Cache setup
        cached_probs, uncached_keys, uncached_texts = (
            await self.precache_text_predictions(texts, labels, multi_label)
        )
        all_predictions = cached_probs.copy()

        # Fetch new predictions for uncached texts
        if uncached_texts:
            results = await self.gpu_classifier_worker.text_arbitrary_classify.remote(
                self.MODEL_NAME, uncached_keys, uncached_texts, labels, multi_label
            )
            for text, result in zip(uncached_texts, results):
                label_probs = dict(zip(result["labels"], result["scores"]))
                all_predictions[text] = label_probs

        # Return results in the order of input texts
        return [all_predictions[text or "none"] for text in texts]

    async def get_ml_scores(self, records, category, comparator, threshold):
        """
        Compute scores for a specific category from the zero-shot classification output.
        """
        labels = [category, f"not_{category}"]
        texts = [record["commit"]["record"]["text"] for record in records]
        predictions = await self.get_predictions(texts, labels)
        return np.array([pred.get(category, 0.0) for pred in predictions])

    async def classifier_operator(self, records, category, comparator, threshold):
        """
        Classify records using zero-shot classification. The `labels` parameter should
        include the category of interest and any other relevant labels.
        """

        scores = await self.get_ml_scores(records, category, comparator, threshold)
        bools = LogicEvaluator.compare(scores, comparator, threshold)
        return bools

    async def register_operations(self, logic_evaluator):
        """
        Register the zero-shot text classification operation with the logic evaluator.
        """
        await logic_evaluator.add_operation(
            "text_arbitrary", self.classifier_operator, True, True
        )
