from app.data_types import EnrichedData
from app.kube.processor import KubeProcessor
from app.logger import logger


class KubeRouter:
    @classmethod
    async def process_request(cls, dispatcher, params: EnrichedData, noop: bool):
        logger.info(params.wrap)
        logger.info(params)
        if noop:
            logger.info("noop")
        else:
            if params.task == "process_algos":

                await KubeProcessor.process_algos(
                    dispatcher, params.transactions()
                )
