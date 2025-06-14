import os
from typing import Dict
from dataclasses import dataclass, field
import pyroscope
from app.logger import logger


@dataclass
class CpuProfiler:
    application_name: str = "grazer.m.graze.social"
    server_address: str = os.getenv(
        "PYROSCOPE_URL", "http://pyroscope-headless.pyroscope.svc.cluster.local:4040"
    )
    sample_rate: int = 100
    detect_subprocesses: bool = True
    oncpu: bool = True
    gil_only: bool = True
    enable_logging: bool = False
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.tags.update(self.default_tags)

    @property
    def default_tags(self) -> Dict[str, str]:
        return {
            "ray_node": f"{os.getenv('RAY_CLOUD_INSTANCE_ID')}",
            "ray_cluster": f"{os.getenv('RAY_CLUSTER_NAME')}",
            "ray_worker_group": f"{os.getenv('RAY_NODE_TYPE_NAME')}",
        }

    def load(self):
        # Declare in boot config or use default
        # Starts the agent
        logger.info("Starting pyroscope prof")
        pyroscope.configure(**self.__dict__)
