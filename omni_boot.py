import asyncio
import ray
import uuid

from app.settings import OmniBootSettings

from app.ray.utils import (
    discover_named_actors,
    discover_named_actor,
)

from app.logger import logger

boot_settings = OmniBootSettings()

DEFAULT_NAMESPACE = "main"


async def boot_cache(num_cpus: float, num_gpus: float):
    from app.ray.semaphore import SemaphoreActor
    from app.ray.cache import Cache

    SemaphoreActor.options(
        name="semaphore:bluesky", lifetime="detached", namespace=DEFAULT_NAMESPACE
    ).remote()

    SemaphoreActor.options(
        name="semaphore:graze", lifetime="detached", namespace=DEFAULT_NAMESPACE
    ).remote(10)  # type: ignore

    # Start the Cache actor with the specified resources and name
    Cache.options(  # type: ignore
        name="cache:main",
        lifetime="detached",
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        namespace=DEFAULT_NAMESPACE,
    ).remote()

    # can't validate remote object ref on actors
    # ray.get([bsky_actor, graze_actor, cache_actor], timeout=300)


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
                lifetime="detached",
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            ).remote(cache, bluesky_semaphore, graze_semaphore)
        )

    logger.info(
        f"NetworkWorker worker '{name}' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )

    # run the loop for each worker
    ray.get([network_worker.run.remote() for network_worker in network_workers])  # type: ignore


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
                lifetime="detached",
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                namespace=DEFAULT_NAMESPACE,
            ).remote(
                gpu_embedding_workers, gpu_classifier_workers, network_workers, cache
            )
        )

    ray.get([cpu_worker.run.remote() for cpu_worker in cpu_workers])  # type: ignore


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
                lifetime="detached",
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            ).remote(network_workers, cache)
        )

    gpu_workers.append(
        GPUWorker.options(
            name=f"{name}:classifiers",
            lifetime="detached",
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        ).remote(network_workers, cache)
    )

    ray.get([gpu_worker.run.remote() for gpu_worker in gpu_workers])  # type: ignore


async def boot_consumer():
    from app.sqs_consumer import SQSConsumer

    consumer = SQSConsumer.options(
        max_concurrency=10,
        lifetime="detached",
        namespace=DEFAULT_NAMESPACE,
        num_cpus=0.5,
    ).remote()
    ray.get([consumer.run.remote()])  # type: ignore


async def omni_boot():
    if boot_settings.boot_cache:
        await boot_cache(num_cpus=0.5, num_gpus=0)
    if boot_settings.boot_network:
        await boot_network(num_workers=3, num_cpus=0.1, num_gpus=0)
    if boot_settings.boot_cpu:
        await boot_cpu(num_cpus=0.5, num_gpus=0, num_workers=3)
    if boot_settings.boot_gpu:
        await boot_gpu(num_cpus=0, num_gpus=0.5, num_workers=1)
    if boot_settings.boot_consumer:
        await boot_consumer()


if __name__ == "__main__":
    asyncio.run(omni_boot())
