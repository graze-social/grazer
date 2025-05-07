import asyncio
from app.jetstream import Jetstream

if __name__ == "__main__":
    asyncio.run(Jetstream.stream_to_sqs())
