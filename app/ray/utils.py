import ray
from ray.actor import ActorHandle
import time
import argparse
import random
from app.logger import logger
from app.settings import OmniBootSettings

settings = OmniBootSettings()


def get_or_create_actor(actor_cls, *constructor_args, **kwargs) -> ActorHandle:
    """Use prexisting actors with handle/namespace"""
    kill = kwargs.get("kill", False)
    kwargs.pop("kill", None)

    try:
        actor = ray.get_actor(kwargs["name"], kwargs.get("namespace", None))
        if not actor:
            return actor_cls.options(**kwargs).remote(*constructor_args)
        else:
            if kill:
                ray.kill(actor)
                return actor_cls.options(**kwargs).remote(*constructor_args)
            else:
                return actor
    except ValueError:
        # logger.warn(e)
        logger.warn("Actor does not exist, possible first boot")

        return actor_cls.options(**kwargs).remote(*constructor_args)


def discover_named_actor(prefix, timeout):
    actors = discover_named_actors(prefix, timeout)
    if actors:
        return random.choice(actors)

def discover_named_actors(prefix, timeout=10, fail_hard=True):
    """
    Discover actors with names that start with a given prefix within a timeout.

    Args:
        prefix (str): The prefix to filter actor names.
        timeout (int): Time (in seconds) to spend discovering actors.

    Returns:
        list: A list of matching actor references.
    """
    start_time = time.time()
    matching_actors = []

    while time.time() - start_time < timeout:
        # List all named actors
        named_actors = ray.util.list_named_actors(all_namespaces=True)
        # Filter by prefix
        # Listing with `all_namespaces=True` returns actors in the format {"name": name, "namespace": namespace}
        # Since we are querying specific namespaces, we change the list comprehension here to be able to specify the namespace where the actor has matched the prefix
        # This is functionally the same result when running actors in default namespaces vs heterogenous namespaces,
        # ie there should be no actors with identical names in different namespaces in any one ray cluster which would not be useable across namespaces.
        # However, there is a performance penalty as for looking up across namespaces
        matching_actors = [
            ray.get_actor(name, namespace)
            for (name, namespace) in (lambda d: [o.values() for o in d])(named_actors)
            if name.startswith(prefix)
        ]

        if matching_actors:
            return matching_actors
        elif fail_hard:
            raise Exception(f"No matching actors matching prefix {prefix} found in any namespace")

        time.sleep(1)  # Poll every second to allow actors to register

    return matching_actors


def parse_cache_worker_args():
    """
    Parse arguments for the Cache worker configuration.

    Returns:
        Namespace: Parsed arguments containing name, num_cpus, and num_gpus.
    """
    parser = argparse.ArgumentParser(
        description="Start the worker with dynamic resource allocation."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="cache_worker",
        help="Name of the worker actor (default: 'cache_worker').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Workers to allocate (default: 1).",
    )
    parser.add_argument(
        "--num-cpus",
        type=float,
        default=3,
        help="Number of CPUs to allocate for the worker (default: 1).",
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=0,
        help="Number of GPUs to allocate for the worker (default: 0).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default_namespace",
        help="Namespace for the Ray instance (default: 'default_namespace').",
    )
    return parser.parse_args()
