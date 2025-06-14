import os
from app.logger import logger
from typing import List, Dict, Optional
from dataclasses import dataclass
import aiohttp


def timestamp(time: float) -> int:
    # returns a unix timestamp in ms
    return int(round(time * 1000))


@dataclass
class GrafanaAdapter:
    """
    An adapter for the Grafana Annotation API
    """

    # Setup
    service_account_token: Optional[str] = os.getenv("GRAFANA_TOKEN")
    grafana_host: str = os.getenv("GRAFANA_HOST", "localhost:3000")
    ssl: bool = False
    tags: Optional[List[str]] = None

    # Ray Default Dashboard
    dashboard_uid: str = "56"
    # This doesn't really matter if the query has all "All Panels" enabled on the dashboard
    panel_id: str = "38"

    def __post_init__(self):
        if self.tags:
            self.tags.extend(["grazer"])

    async def annotate_session(self, txt: str, time: float, time_end: float):
        payload = {
            # "dashboardUID": self.dashboard_uid,
            # "panelId": self.panel_id,
            "time": timestamp(time),
            "timeEnd": timestamp(time_end),
            "tags": self.tags,
            "text": txt,
        }

        build_url = f"http://{self.grafana_host}/api/annotations"

        async with aiohttp.ClientSession() as session:
            result = await session.post(build_url, headers=self.default_headers, data=payload)
            logger.info(f"annotation response: {result.status}")

    @property
    def default_headers(self) -> Dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.service_account_token}",
        }


@dataclass
class RedisAdapter:
    """TODO"""

    pass
