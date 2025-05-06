from app.kube.processor import KubeProcessor
from app.logger import logger


class KubeRouter:
    @classmethod
    async def process_request(cls, dispatcher, params, noop: bool):
        logger.info("Params are here!")
        logger.info(type(params))
        logger.info(params.keys())
        if noop:
            logger.info("noop")
        else:
            if params.get("task") == "process_algos":
                await KubeProcessor.process_algos(
                    dispatcher, params.get("transactions")
                )
