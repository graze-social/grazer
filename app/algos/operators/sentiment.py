from app.algos.operators.huggingface_classifier import HuggingfaceClassifierParser


class SentimentParser(HuggingfaceClassifierParser):
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    LABEL_MAP = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "sentiment_analysis", self.classifier_operator, True
        )
