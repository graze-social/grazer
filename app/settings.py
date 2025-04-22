import os
from pydantic_settings import BaseSettings

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
HOSTNAME = "https://api.graze.social"
JETSTREAM_URL = os.getenv(
    "JETSTREAM_URL",
    "wss://jetstream2.us-west.bsky.network/subscribe?wantedCollections=app.bsky.feed.post",
)
REDIS_DELETE_POST_QUEUE = "grazer_delete_posts"
CURRENT_ALGORITHMS_KEY = "current_algorithms"
SENTRY_DSN = os.getenv(
    "SENTRY_DSN"
)

class StreamerSettings(BaseSettings):
    # TODO: making this optional is a stupid LSP thing
    sqs_queue_url: str = os.getenv("SQS_QUEUE_URL", "[placeholder]")
    aws_region: str = "us-east-1"
    sqs_polling_interval: int = 10
    noop: bool = True
