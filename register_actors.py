import uuid
import ray
import ray.runtime_env

from app.ray.cache import Cache
from app.ray.semaphore import SemaphoreActor
from app.ray.cpu_worker import CPUWorker
from app.ray.gpu_worker import GPUWorker

from app.ray.utils import (
    parse_cache_worker_args,
    discover_named_actors,
    discover_named_actor,
)

from app.sentry import sentry_sdk


RuntimeEnv = ray.runtime_env.RuntimeEnv


def register_cache_actors(name, num_cpus, num_gpus):

    SemaphoreActor.options(
        name="semaphore:bluesky", lifetime="detached"
    ).remote()

    SemaphoreActor.options(
        name="semaphore:graze", lifetime="detached"
    ).remote(10)

    # Start the Cache actor with the specified resources and name
    Cache.options(
        name=name, lifetime="detached", num_cpus=num_cpus, num_gpus=num_gpus
    ).remote()

    print(
        f"Cache worker '{name}' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )


def register_cpu_actors(name, num_cpus, num_gpus, num_workers):
    # Discover network workers and cache workers
    network_workers = discover_named_actors("network:", timeout=10)
    gpu_embedding_workers = discover_named_actors("gpu:embedders", timeout=10)
    gpu_classifier_workers = discover_named_actors("gpu:classifiers", timeout=10)
    cache = discover_named_actor("cache:", timeout=10)
    print(f"Creating a new CPUWorker: {name}")
    cpu_workers = []
    for i in range(num_workers):
        uuid_str = str(uuid.uuid4())
        cpu_workers.append(
            CPUWorker.options(
                name=f"{name}-{i}-{uuid_str}",
                lifetime="detached",
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            ).remote(
                gpu_embedding_workers, gpu_classifier_workers, network_workers, cache
            )
        )

    print(
        f"CPUWorker worker '{name}' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )


def register_gpu_actors(name: str, num_cpus: float, num_gpus: float, num_workers: int):
    # Discover network workers and cache workers
    network_workers = discover_named_actors("network:", timeout=10)
    cache = discover_named_actor("cache:", timeout=10)
    for i in range(num_workers):
        uuid_str = str(uuid.uuid4())
        gpu_embedding_worker = GPUWorker.options(
            name=f"{name}:embedders-{i}-{uuid_str}",
            lifetime="detached",
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        ).remote(network_workers, cache)
    gpu_classifier_worker = GPUWorker.options(
        name=f"{name}:classifiers",
        lifetime="detached",
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    ).remote(network_workers, cache)
    print(
        f"GPUWorker worker '{name}' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )


def main():
    ray.init(
        ignore_reinit_error=True,
        namespace="default",
    )


    register_cache_actors("cache:main", num_cpus=2, num_gpus=0)
    register_cpu_actors("cpu:main", num_cpus=0.5, num_gpus=0, num_workers=2)

if __name__ == "__main__":
    main()
