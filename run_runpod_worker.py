import ray
import runpod
from app.runpod.router import RunpodRouter
from app.ray.dispatcher import Dispatcher
from app.ray.utils import parse_cache_worker_args
from app.sentry import sentry_sdk


args = parse_cache_worker_args()
ray.init(namespace=args.namespace)

NUM_WORKERS = 1


async def process_request(job):
    return await RunpodRouter.process_request(Dispatcher(), job["input"])


def adjust_concurrency(current_concurrency):
    """
    Adjust the concurrency level based on the current request rate.
    For this example, we'll keep the concurrency level fixed.
    """
    return NUM_WORKERS


runpod.serverless.start(
    {"handler": process_request, "concurrency_modifier": adjust_concurrency}
)
