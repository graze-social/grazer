from app.redis import RedisClient


class Egress:
    @classmethod
    async def send_results(cls, outputs, keyname):
        await RedisClient.send_pipeline(outputs, keyname)
