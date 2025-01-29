import ray
from PIL import Image
from io import BytesIO
import aiohttp
import asyncio
import httpx
from atproto import IdResolver
from app.algorithm_asset_cacher import AlgorithmAssetCacher
from app.helpers import dict_to_sorted_string
from app.settings import HOSTNAME
from app.ray.timing_base import TimingBase, measure_time


@ray.remote(max_concurrency=100)
class NetworkWorker(TimingBase):
    def __init__(self, cache, bluesky_semaphore, graze_semaphore):
        """
        Initialize the NetworkWorker with a reference to the shared Cache actor.

        Args:
            cache: A reference to the Cache actor.
        """
        self.cache = cache
        self.bluesky_semaphore = bluesky_semaphore
        self.graze_semaphore = graze_semaphore
        super().__init__()

    @measure_time
    async def fetch_asset(
        self, asset_type: str, asset_name: str, asset_parameters: dict
    ) -> dict:
        """Fetch asset from remote API with retries."""
        await self.graze_semaphore.acquire.remote()
        try:
            url = f"{HOSTNAME}/app/api/v1/assets/get_cached"
            payload = {
                "asset_type": asset_type,
                "asset_name": asset_name,
                "asset_parameters": asset_parameters,
            }
            for i in range(5):
                async with httpx.AsyncClient() as client:
                    try:
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        return resp.json()
                    except httpx.HTTPError:
                        if i == 4:
                            raise
                        await asyncio.sleep(1)
        except httpx.HTTPError as e:
            raise e
        finally:
            await self.graze_semaphore.release.remote()

    @measure_time
    async def get_asset(
        self, asset_type: str, asset_parameters: dict, keyname_template: str
    ):
        """Retrieve an asset, checking and writing to the cache."""
        asset_name = f"{asset_type}__{dict_to_sorted_string(asset_parameters)}"
        key = f"{asset_type}__{asset_name}"

        # Check the cache
        cached_asset = await self.cache.get_asset.remote(key)
        if cached_asset:
            return cached_asset

        # Fetch and store in cache
        asset = await self.fetch_asset(asset_type, asset_name, asset_parameters)
        await self.cache.cache_asset.remote(key, asset)  # Cache for 3 hours
        return asset

    @measure_time
    async def fetch_image(self, url):
        try:
            retries = 3
            cache_key = f"images_{url}"
            for attempt in range(retries):
                img_data = await self.cache.get_image.remote(cache_key)
                if not img_data:
                    await self.bluesky_semaphore.acquire.remote()
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, timeout=10) as response:
                                if response.status == 200:
                                    img_data = await response.read()
                                    await self.cache.cache_image.remote(
                                        cache_key, img_data
                                    )
                                    image = Image.open(BytesIO(img_data)).convert("RGB")
                                    return image
                                elif response.status == 429 and attempt < retries - 1:
                                    retry_after = int(
                                        response.headers.get("Retry-After", 2**attempt)
                                    )
                                    print(
                                        f"Rate limited (429). Retrying in {retry_after} seconds..."
                                    )
                                    await asyncio.sleep(retry_after)
                                else:
                                    print(f"Error: HTTP {response.status} for {url}")
                                    break
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        print(f"Error fetching image from {url}: {e}")
                    except Exception as e:
                        print(f"Unexpected error fetching image from {url}: {e}")
                        break
                    finally:
                        await self.bluesky_semaphore.release.remote()
                else:
                    image = Image.open(BytesIO(img_data)).convert("RGB")
                    return image
        except httpx.HTTPError as e:
            raise e

    @measure_time
    async def get_magic_audience(self, audience_id: str) -> dict:
        """Retrieve magic audience asset."""
        return await self.get_asset(
            asset_type="magic_audience",
            asset_parameters={"audience_id": audience_id},
            keyname_template="{audience_id}",
        )

    @measure_time
    async def get_user_collection(self, actor_handle, direction):
        params = AlgorithmAssetCacher.get_user_collection_params(
            actor_handle, direction
        )
        return await self.get_asset(**params)

    @measure_time
    async def get_starter_pack(self, starter_pack_url):
        params = AlgorithmAssetCacher.get_starter_pack_params(starter_pack_url)
        return await self.get_asset(**params)

    @measure_time
    async def get_list(self, list_url):
        params = AlgorithmAssetCacher.get_list_params(list_url)
        return await self.get_asset(**params)

    @measure_time
    async def get_or_set_handle_did(self, handle):
        existing = await self.cache.get_did.remote(handle)
        if not existing:
            resolver = IdResolver()
            did = handle
            if not handle.startswith("did:plc:"):
                did = resolver.handle.resolve(handle)
            await self.cache.cache_did.remote(handle, did)
            return did
        else:
            return existing
