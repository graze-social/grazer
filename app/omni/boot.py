import asyncio
import ray
import uuid


from app.logger import logger
from app.ray.utils import (
    discover_named_actor,
    discover_named_actors,
    get_or_create_actor,
)
from app.omni.boot_settings import BootConfig, BootSettings

class OmniBoot:
    def __init__(self, config: BootConfig) -> None:
        self.actors = config.actors
        self.boot_settings = config.boot_settings if config.boot_settings else BootSettings()
        # get boot-specific settings
        self.actor_lifetime = self.boot_settings.lifetimes()[2]
        self.namespace = self.boot_settings.namespace

    async def boot_cache(self, num_cpus: float, num_gpus: float):
        from app.ray.cache import Cache
        from app.ray.semaphore import SemaphoreActor

        get_or_create_actor(
            SemaphoreActor,
            namespace=self.namespace,
            name="semaphore:bluesky",
            lifetime=self.actor_lifetime,
            kill=True,
        )

        logger.info(
            f"Semaphore Actor 'sempaphore:bluesky' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
        )

        get_or_create_actor(
            SemaphoreActor,
            10,
            name="semaphore:graze",
            lifetime=self.actor_lifetime,
            namespace=self.namespace,
        )

        logger.info(
            f"Semaphore Actor 'sempaphore:graze' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
        )

        get_or_create_actor(
            Cache,
            name=f"cache:{self.namespace}",
            lifetime=self.actor_lifetime,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            namespace=self.namespace,
        )

        logger.info(
            f"Cache worker 'cache:main' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
        )


    async def boot_network(self, num_workers: int, num_cpus: float, num_gpus: int):
        from app.ray.network_worker import NetworkWorker

        bluesky_semaphore = discover_named_actor("semaphore:bluesky", timeout=10)
        graze_semaphore = discover_named_actor("semaphore:graze", timeout=10)
        cache = discover_named_actor("cache:", timeout=10)
        network_workers = []
        name = f"network:{self.namespace}"
        for i in range(num_workers):
            uuid_str = str(uuid.uuid4())
            network_workers.append(
                NetworkWorker.options(  # type: ignore
                    name=f"{name}-{i}-{uuid_str}",
                    namespace=self.namespace,
                    lifetime=self.actor_lifetime,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                ).remote(cache, bluesky_semaphore, graze_semaphore)
            )

        logger.info(
            f"NetworkWorker worker '{name}' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
        )


    async def boot_cpu(self, num_cpus: float, num_gpus: float, num_workers: int):
        from app.ray.cpu_worker import CPUWorker

        name = f"cpu:{self.namespace}"
        network_workers = discover_named_actors("network:", timeout=10)
        gpu_embedding_workers = discover_named_actors(
            f"gpu:{self.namespace}:embedders", timeout=10
        )
        gpu_classifier_workers = discover_named_actors(
            f"gpu:{self.namespace}:classifiers", timeout=10
        )
        cache = discover_named_actor("cache:", timeout=10)
        logger.info(f"Creating a new CPUWorker: {name}")
        cpu_workers = []
        for i in range(num_workers):
            uuid_str = str(uuid.uuid4())
            cpu_workers.append(
                CPUWorker.options(  # type: ignore
                    name=f"{name}-{i}-{uuid_str}",
                    lifetime=self.actor_lifetime,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    namespace=self.namespace,
                ).remote(
                    gpu_embedding_workers, gpu_classifier_workers, network_workers, cache
                )
            )


    async def boot_gpu(self, num_cpus: float, num_gpus: float, num_workers: int):
        from app.ray.gpu_worker import GPUWorker

        name = f"gpu:{self.namespace}"
        # Discover network workers and cache workers
        network_workers = discover_named_actors("network:", timeout=10)
        cache = discover_named_actor("cache:", timeout=10)
        gpu_workers = []
        for i in range(num_workers):
            uuid_str = str(uuid.uuid4())
            gpu_workers.append(
                GPUWorker.options(
                    name=f"{name}:embedders-{i}-{uuid_str}",
                    namespace=self.namespace,
                    lifetime=self.actor_lifetime,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                ).remote(network_workers, cache)
            )

        gpu_workers.append(
            GPUWorker.options(
                name=f"{name}:classifiers",
                lifetime=self.actor_lifetime,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            ).remote(network_workers, cache)
        )


    async def boot_consumer(self,num_cpus: float):
        from app.sqs_consumer import SQSConsumer

        consumer = SQSConsumer.options(
            max_concurrency=10, namespace=self.namespace, num_cpus=num_cpus
        ).remote()

        return consumer


    async def boot(self):
        if self.boot_settings.boot_cache:
            logger.info("Booting Cache")
            await self.boot_cache(**self.actors.cache_worker.cfg)

        if self.boot_settings.boot_network:
            logger.info("Booting Network")
            await self.boot_network(**self.actors.network_worker.cfg)

        if self.boot_settings.boot_gpu:
            logger.info("Booting GPU Worker")
            await self.boot_gpu(**self.actors.gpu_worker.cfg)

        if self.boot_settings.boot_cpu:
            logger.info("Booting CPU Worker")
            await self.boot_cpu(**self.actors.cpu_worker.cfg)

        if self.boot_settings.boot_consumer:
            logger.info("Booting Consumer")
            consumer = await self.boot_consumer(**self.actors.consumer_worker.cfg)

        try:
            await asyncio.sleep(20)
            consumer.receive_messages.remote()  # type: ignore
            logger.info("Starting Consumer")
            ray.get(consumer.num_gathered_tasks.remote())  # type: ignore
            while True:
                await asyncio.sleep(10)
                logger.info("[Omni] heartbeat")
        except KeyboardInterrupt:
            print("User stopped omniboot job")
        except SystemExit:
            # ray job stop <job> sends SIGTERM by default
            print("Termination signal received for omniboot job")


if __name__ == "__main__":
    omni_boot = OmniBoot(config=BootConfig.load())
    # run the thing
    asyncio.run(omni_boot.boot())
