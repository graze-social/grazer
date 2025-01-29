from app.algos.base import ImageParser


class ImageArbitraryParser(ImageParser):
    MODEL_NAME = "openai/clip-vit-base-patch16"

    async def probability_function(self, cache_keys, images, category):
        return await self.gpu_classifier_worker.image_arbitrary_classify.remote(
            self.MODEL_NAME, cache_keys, images, category
        )

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "image_arbitrary", self.classifier_operator, True, True
        )
