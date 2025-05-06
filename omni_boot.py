import asyncio
import ray
import uuid
from app.settings import OmniBootSettings

from app.ray.utils import (
    get_or_create_actor,
    discover_named_actors,
    discover_named_actor,
)

from app.logger import logger

boot_settings = OmniBootSettings()

DEFAULT_NAMESPACE = "main"

actor_lifetime = "detached" if boot_settings.extended_lifetime else None

async def boot_cache(num_cpus: float, num_gpus: float):
    from app.ray.semaphore import SemaphoreActor
    from app.ray.cache import Cache

    get_or_create_actor(
        SemaphoreActor,
        namespace=DEFAULT_NAMESPACE,
        name="semaphore:bluesky",
        lifetime=actor_lifetime,
        kill=True,
    )

    logger.info(
        f"Semaphore Actor 'sempaphore:bluesky' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )

    get_or_create_actor(
        SemaphoreActor,
        10,
        name="semaphore:graze",
        lifetime=actor_lifetime,
        namespace=DEFAULT_NAMESPACE,
    )

    logger.info(
        f"Semaphore Actor 'sempaphore:graze' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )

    get_or_create_actor(
        Cache,
        name="cache:main",
        lifetime=actor_lifetime,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        namespace=DEFAULT_NAMESPACE,
    )

    logger.info(
        f"Cache worker 'cahe:main' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )


async def boot_network(num_workers: int, num_cpus: float, num_gpus: int):
    from app.ray.network_worker import NetworkWorker

    bluesky_semaphore = discover_named_actor("semaphore:bluesky", timeout=10)
    graze_semaphore = discover_named_actor("semaphore:graze", timeout=10)
    cache = discover_named_actor("cache:", timeout=10)
    network_workers = []
    name = "network:main"
    for i in range(num_workers):
        uuid_str = str(uuid.uuid4())
        network_workers.append(
            NetworkWorker.options(  # type: ignore
                name=f"{name}-{i}-{uuid_str}",
                lifetime=actor_lifetime,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            ).remote(cache, bluesky_semaphore, graze_semaphore)
        )

    # [network_worker.run.remote() for network_worker in network_workers]

    logger.info(
        f"NetworkWorker worker '{name}' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )


async def boot_cpu(num_cpus: float, num_gpus: float, num_workers: int):
    from app.ray.cpu_worker import CPUWorker

    name = "cpu:main"
    network_workers = discover_named_actors("network:", timeout=10)
    gpu_embedding_workers = discover_named_actors("gpu:embedders", timeout=10)
    gpu_classifier_workers = discover_named_actors("gpu:classifiers", timeout=10)
    cache = discover_named_actor("cache:", timeout=10)
    logger.info(f"Creating a new CPUWorker: {name}")
    cpu_workers = []
    for i in range(num_workers):
        uuid_str = str(uuid.uuid4())
        cpu_workers.append(
            CPUWorker.options(  # type: ignore
                name=f"{name}-{i}-{uuid_str}",
                lifetime=actor_lifetime,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                namespace=DEFAULT_NAMESPACE,
            ).remote(
                gpu_embedding_workers, gpu_classifier_workers, network_workers, cache
            )
        )
    # [cpu_worker.run.remote() for cpu_worker in cpu_workers]


async def boot_gpu(num_cpus: float, num_gpus: float, num_workers: int):
    from app.ray.gpu_worker import GPUWorker

    name = "gpu:main"
    # Discover network workers and cache workers
    network_workers = discover_named_actors("network:", timeout=10)
    cache = discover_named_actor("cache:", timeout=10)
    gpu_workers = []
    for i in range(num_workers):
        uuid_str = str(uuid.uuid4())
        gpu_workers.append(
            GPUWorker.options(
                name=f"{name}:embedders-{i}-{uuid_str}",
                lifetime=actor_lifetime,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            ).remote(network_workers, cache)
        )

    gpu_workers.append(
        GPUWorker.options(
            name=f"{name}:classifiers",
            lifetime=actor_lifetime,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        ).remote(network_workers, cache)
    )

    # [gpu_worker.run.remote() for gpu_worker in gpu_workers]


async def boot_consumer():
    from app.sqs_consumer import SQSConsumer

    consumer = SQSConsumer.options(
        max_concurrency=10,
        namespace=DEFAULT_NAMESPACE,
        num_cpus=0.5,
    ).remote()

    # consumer.receive_messages.remote()  # type: ignore
    # await asyncio.sleep(20)
    return consumer


async def omni_boot():
    if boot_settings.boot_cache:
        logger.info("Booting Cache")
        await boot_cache(num_cpus=0.5, num_gpus=0)
    if boot_settings.boot_network:
        logger.info("Booting Network")
        await boot_network(num_workers=3, num_cpus=0.1, num_gpus=0)
    if boot_settings.boot_cpu:
        logger.info("Booting CPU Worker")
        await boot_cpu(num_cpus=0.5, num_gpus=0, num_workers=1)
    if boot_settings.boot_gpu:
        logger.info("Booting GPU Worker")
        await boot_gpu(num_cpus=0, num_gpus=0.2, num_workers=1)
    if boot_settings.boot_consumer:
        logger.info("Booting Consumer")
        consumer = await boot_consumer()

    try:
        await asyncio.sleep(20)
        consumer.receive_messages.remote() #type: ignore
        logger.info("Starting Consumer")
        ray.get(consumer.num_gathered_tasks.remote()) #type: ignore
        while True:
            await asyncio.sleep(10)
            logger.info("[Omni] heartbeat")
    except KeyboardInterrupt:
        print("User stopped omniboot job")
    except SystemExit:
        # ray job stop <job> sends SIGTERM by default
        print("Termination signal received for omniboot job")


if __name__ == "__main__":
    asyncio.run(omni_boot())
