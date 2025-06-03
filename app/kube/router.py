from app.stream_data import StreamData
from app.kube.processor import KubeProcessor
from app.logger import logger


class KubeRouter:
    @classmethod
    async def process_request(cls, dispatcher, params: StreamData, noop: bool):
        logger.info(params.wrap())
        if noop:
            logger.info("noop")
        else:
            if params.task == "process_algos":
                await KubeProcessor.process_algos(dispatcher, params.transactions())
