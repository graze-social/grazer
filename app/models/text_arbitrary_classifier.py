from transformers import pipeline
from app.models.ml_model import MLModel


class TextArbitraryClassifier(MLModel):
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    @classmethod
    def get_model(cls, model_name):
        """
        Load the zero-shot classification pipeline with the model and tokenizer.
        """
        device = 0 if cls.get_device().type == "cuda" else -1
        zero_shot_pipe = pipeline(
            "zero-shot-classification", model=model_name, device=device, batch_size=16
        )
        zero_shot_pipe.model.half()
        return {"pipeline": zero_shot_pipe}

    @classmethod
    async def compute_predictions(
        cls, model_data, texts, labels, multi_label, batch_size=200
    ):
        return model_data["pipeline"](
            texts, candidate_labels=labels, multi_label=multi_label
        )
