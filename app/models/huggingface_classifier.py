import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from app.models.ml_model import MLModel


class HuggingfaceClassifier(MLModel):
    @classmethod
    def get_model(cls, model_name):
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            cls.get_device()
        )
        model.half()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        return {"model": model, "tokenizer": tokenizer, "config": config}

    @classmethod
    async def compute_predictions(cls, model_data, texts, label_map, batch_size=200):
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        config = model_data["config"]
        batch_results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [text for text in texts[i : i + batch_size]]
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(cls.get_device())
            with torch.no_grad():
                with autocast():
                    outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            for text, probs in zip(batch_texts, probabilities):
                labels = [label_map[config.id2label[idx]] for idx in range(len(probs))]
                prob_dict = dict(zip(labels, probs))
                batch_results.append(prob_dict)
        return batch_results
