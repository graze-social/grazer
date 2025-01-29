import torch
from transformers import CLIPProcessor, CLIPModel
from app.models.ml_model import MLModel


class ImageArbitraryClassifier(MLModel):
    @classmethod
    def get_model(cls, model_name):
        model = CLIPModel.from_pretrained(model_name).to(cls.get_device()).eval()
        model.half()
        processor = CLIPProcessor.from_pretrained(model_name)
        return {"model": model, "processor": processor}

    @classmethod
    async def compute_predictions(cls, model_data, images, category, batch_size=10):
        prompts = [
            f"This is a photo of {category}.",
            f"This is not a photo of {category}.",
        ]
        model = model_data["model"]
        processor = model_data["processor"]
        full_probs = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = processor(
                text=prompts, images=batch_imgs, return_tensors="pt", padding=True
            ).to(cls.get_device())
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                for prob in logits_per_image.softmax(dim=1).cpu().tolist():
                    full_probs.append(prob)
        return full_probs
