import os
from pydantic_settings import BaseSettings

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
HOSTNAME = "https://api"
JETSTREAM_URL = os.getenv(
    "JETSTREAM_URL",
    "wss://jetstream1.us-west.bsky.network/subscribe?wantedCollections=app.bsky.feed.post",
)
REDIS_DELETE_POST_QUEUE = "grazer_delete_posts"
CURRENT_ALGORITHMS_KEY = "current_algorithms"
SENTRY_DSN = os.getenv("SENTRY_DSN")


class EgressSettings(BaseSettings):
    egress_enabled: bool = False


class StreamerSettings(BaseSettings):
    # TODO: making this optional is a stupid LSP thing
    sqs_queue_url: str = os.getenv("SQS_QUEUE_URL", "[placeholder]")
    aws_region: str = "us-east-1"
    sqs_polling_interval: int = 10
    noop: bool = True


class OmniBootSettings(BaseSettings):
    """Some switches to control the behavior of the booting script"""


    boot_gpu: bool = False
    boot_cpu: bool = True
    boot_cache: bool = True
    boot_network: bool = True
    boot_consumer: bool = False

    """Option to extend the lifetime of all actors for beyond the job termination"""
    extended_lifetime: bool = True
