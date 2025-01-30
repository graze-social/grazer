from app.algos.base import ImageParser


class ImageNSFWParser(ImageParser):
    MODEL_NAME = "Falconsai/nsfw_image_detection"

    async def probability_function(self, cache_keys, images, category):
        return await self.gpu_classifier_worker.image_nsfw_classify.remote(
            self.MODEL_NAME, cache_keys, images
        )

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "image_nsfw", self.classifier_operator, True
        )
