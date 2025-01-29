from app.algos.operators.huggingface_classifier import HuggingfaceClassifierParser


class EmotionSentimentParser(HuggingfaceClassifierParser):
    MODEL_NAME = "SamLowe/roberta-base-go_emotions"
    LABEL_MAP = {
        "admiration": "Admiration",
        "amusement": "Amusement",
        "anger": "Anger",
        "annoyance": "Annoyance",
        "approval": "Approval",
        "caring": "Caring",
        "confusion": "Confusion",
        "curiosity": "Curiosity",
        "desire": "Desire",
        "disappointment": "Disappointment",
        "disapproval": "Disapproval",
        "disgust": "Disgust",
        "embarrassment": "Embarrassment",
        "excitement": "Excitement",
        "fear": "Fear",
        "gratitude": "Gratitude",
        "grief": "Grief",
        "joy": "Joy",
        "love": "Love",
        "nervousness": "Nervousness",
        "optimism": "Optimism",
        "pride": "Pride",
        "realization": "Realization",
        "relief": "Relief",
        "remorse": "Remorse",
        "sadness": "Sadness",
        "surprise": "Surprise",
        "neutral": "Neutral",
    }

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "emotion_sentiment_analysis", self.classifier_operator, True
        )
