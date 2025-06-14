import os
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import yaml

DEFAULT_BOOT_CONFIG_PATH = os.getenv("BOOT_CONFIG_YAML", "/app/boot_config.yaml")


class ActorConfig(BaseModel):
    """Optional actor-level config"""

    num_workers: Optional[int] = None
    num_cpus: Optional[float] = None
    num_gpus: Optional[float] = None

    @property
    def cfg(self) -> Dict[str, Any]:
        """don't return values that are unset for passing as kwargs to different function signatures"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ActorInputConfig(BaseModel):
    cache_worker: ActorConfig
    network_worker: ActorConfig
    gpu_worker: ActorConfig
    cpu_worker: ActorConfig
    consumer_worker: ActorConfig


class BootSettings(BaseSettings):
    """Some switches to control the behavior of the booting script"""

    namespace: str = "main"
    boot_gpu: bool = False
    boot_cpu: bool = True
    boot_cache: bool = True
    boot_network: bool = True
    boot_consumer: bool = False
    max_actor_restarts: int = -1
    max_task_retries: int = -1
    init_profiler: bool = False

    """Option to extend the lifetime of all actors for beyond the job termination"""
    extended_lifetime: bool = True

    def lifetimes(self) -> Tuple[int, int, Optional[str]]:
        # TODO: idk what purpose this serves
        return (
            self.max_actor_restarts,
            self.max_task_retries,
            "detached" if self.extended_lifetime else None,
        )


class BootConfig(BaseModel):
    """The main boot config, loaded from file"""

    actors: ActorInputConfig
    boot_settings: Optional[BootSettings]

    @staticmethod
    def load(path: str = DEFAULT_BOOT_CONFIG_PATH) -> "BootConfig":
        data: Dict[str, Any] = {}
        with open(path, "r") as yaml_file:
            data.update(yaml.safe_load(yaml_file))
        return BootConfig(**data)
