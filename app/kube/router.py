from app.stream_data import StreamData
from app.kube.processor import KubeProcessor
from app.logger import logger
from app.utils.profilers.timing_functions import record_timing


class KubeRouter:
    @classmethod
    @record_timing(prefix="KubeRouter")
    async def process_request(cls, dispatcher, params: StreamData, noop: bool):
        logger.info(params.wrap())
        if noop:
            logger.info("noop")
        else:
            # Note: See stream_data.py
            # The task is always "process_algos"
            if params.task == "process_algos":
                await KubeProcessor.process_algos(dispatcher, params.transactions())
                return
