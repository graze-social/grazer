import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
from app.models.ml_model import MLModel


class ImageNSFWClassifier(MLModel):
    @classmethod
    def get_model(cls, model_name):
        model = (
            AutoModelForImageClassification.from_pretrained(model_name)
            .to(cls.get_device())
            .eval()
        )
        model.half()
        processor = ViTImageProcessor.from_pretrained(model_name)
        keys = ["SFW", "NSFW"]
        return {"model": model, "processor": processor, "class_names": keys}

    @classmethod
    async def compute_predictions(cls, model_data, images, batch_size=20):
        full_probs = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = model_data["processor"](
                images=batch_imgs,
                return_tensors="pt",
            ).to(cls.get_device())
            with torch.no_grad():
                outputs = model_data["model"](**inputs)
                probs = outputs.logits.softmax(dim=1).cpu().tolist()
                for prob in probs:
                    full_probs.append(prob)
        return full_probs
