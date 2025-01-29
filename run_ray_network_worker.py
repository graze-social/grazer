import uuid
import time
import ray
from app.ray.network_worker import NetworkWorker
from app.ray.utils import (
    parse_cache_worker_args,
    discover_named_actor,
)
from app.sentry import sentry_sdk

def main(name, num_workers, num_cpus, num_gpus, namespace):
    # Initialize Ray
    ray.init(ignore_reinit_error=True, namespace=namespace)

    # Discover network workers and cache workers
    bluesky_semaphore = discover_named_actor("semaphore:bluesky", timeout=10)
    graze_semaphore = discover_named_actor("semaphore:graze", timeout=10)
    cache = discover_named_actor("cache:", timeout=10)
    network_workers = []
    for i in range(num_workers):
        uuid_str = str(uuid.uuid4())
        network_workers.append(
            NetworkWorker.options(
                name=f"{name}-{i}-{uuid_str}",
                lifetime="detached",
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            ).remote(cache, bluesky_semaphore, graze_semaphore)
        )

    print(
        f"NetworkWorker worker '{name}' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )
    # Keep the script running to maintain the actor
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"NetworkWorker worker '{name}' stopped.")


if __name__ == "__main__":
    args = parse_cache_worker_args()
    main(args.name, args.num_workers, args.num_cpus, args.num_gpus, args.namespace)
