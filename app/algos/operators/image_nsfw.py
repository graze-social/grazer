from app.algos.base import ImageParser


class ImageNSFWParser(ImageParser):
    MODEL_NAME = "Falconsai/nsfw_image_detection"

    def process_probs(self, probs, category):
        if category == "NSFW":
            probs = list(reversed(probs))
        return {
            category: probs[0],
            f"not_{category}": probs[1],
        }

    async def probability_function(self, cache_keys, images, category):
        return await self.gpu_classifier_worker.image_nsfw_classify.remote(
            self.MODEL_NAME, cache_keys, images
        )

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "image_nsfw", self.classifier_operator, True
        )
