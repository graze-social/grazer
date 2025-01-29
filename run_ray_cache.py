import ray
from app.ray.cache import Cache
from app.ray.semaphore import SemaphoreActor
from app.ray.utils import parse_cache_worker_args
from app.sentry import sentry_sdk

def main(name, num_cpus, num_gpus, namespace):
    # Initialize Ray
    ray.init(ignore_reinit_error=True, namespace=namespace)

    bluesky_semaphore = SemaphoreActor.options(
        name="semaphore:bluesky", lifetime="detached"
    ).remote()

    graze_semaphore = SemaphoreActor.options(
        name="semaphore:graze", lifetime="detached"
    ).remote(10)

    # Start the Cache actor with the specified resources and name
    cache_actor = Cache.options(
        name=name, lifetime="detached", num_cpus=num_cpus, num_gpus=num_gpus
    ).remote()

    print(
        f"Cache worker '{name}' started with {num_cpus} CPUs and {num_gpus} GPUs and running..."
    )
    # Keep the script running to maintain the actor
    try:
        import time

        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"Cache worker '{name}' stopped.")


if __name__ == "__main__":
    args = parse_cache_worker_args()
    main(args.name, args.num_cpus, args.num_gpus, args.namespace)
