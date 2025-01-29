from app.algos.operators.huggingface_classifier import HuggingfaceClassifierParser


class FinancialSentimentParser(HuggingfaceClassifierParser):
    MODEL_NAME = "ProsusAI/finbert"
    LABEL_MAP = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "financial_sentiment_analysis", self.classifier_operator, True
        )
