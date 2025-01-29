import asyncio
from app.algos.base import BaseParser
from app.algos.operators.image_nsfw import ImageNSFWParser
from app.algos.operators.toxicity import ToxicityParser
from app.algos.operators.entity import EntityParser


class ComboParser(BaseParser):
    async def no_nsfw(self, records, threshold):
        no_labels, non_toxic, no_nsfw = await asyncio.gather(
            EntityParser().excludes_entities(
                records, "labels", ["porn", "sexual", "graphic-media", "nudity"]
            ),
            ToxicityParser().classifier_operator(records, "Toxic", "<=", threshold),
            ImageNSFWParser().classifier_operator(records, "NSFW", "<=", threshold),
        )
        return no_labels & non_toxic & no_nsfw

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation("no_nsfw", self.no_nsfw)
