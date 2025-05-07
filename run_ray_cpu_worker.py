import uuid
import time
import ray
from app.ray.cpu_worker import CPUWorker
from app.ray.utils import (
    parse_cache_worker_args,
    discover_named_actors,
    discover_named_actor,
)
from app.sentry import sentry_sdk


def main(name, num_workers, num_cpus, num_gpus, namespace):
    # Initialize Ray
    ray.init(ignore_reinit_error=True, namespace=namespace)

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
    # Keep the script running to maintain the actor
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"CPUWorker worker '{name}' stopped.")


if __name__ == "__main__":
    args = parse_cache_worker_args()
    main(args.name, args.num_workers, args.num_cpus, args.num_gpus, args.namespace)
