from sentence_transformers import SentenceTransformer
from app.models.ml_model import MLModel


class TextEmbedder(MLModel):
    @classmethod
    def get_model(cls, model_name):
        return SentenceTransformer(model_name, model_kwargs={"torch_dtype": "float16"})

    @classmethod
    async def compute_predictions(cls, model, texts, batch_size=200):
        batch_results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [text for text in texts[i : i + batch_size]]
            embeddings = model.encode(batch_texts)  # Perform batch encoding
            for text, embedding in zip(batch_texts, embeddings):
                batch_results.append(embedding)
        return batch_results
