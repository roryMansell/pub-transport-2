# src/app/providers/tfl.py
from typing import List, Optional
import os
import httpx
from ..models import Arrival
from ..cache import ttl_cache

TFL_BASE = "https://api.tfl.gov.uk"

class TflProvider:
    def __init__(self, app_id: Optional[str] = None, app_key: Optional[str] = None, **_):
        self.app_id = app_id or os.getenv("TFL_APP_ID")
        self.app_key = app_key or os.getenv("TFL_APP_KEY")

    @ttl_cache(5)
    async def get_arrivals(self, stop_id: str) -> List[Arrival]:
        params = {}
        if self.app_key:
            params["app_key"] = self.app_key
        if self.app_id:
            params["app_id"] = self.app_id

        url = f"{TFL_BASE}/StopPoint/{stop_id}/Arrivals"
        headers = {"User-Agent": "bus-rt-minimal/0.1"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                r = await client.get(url, params=params, headers=headers)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            # Re-raise a simpler message for our API layer
            raise RuntimeError(f"TfL error {e.response.status_code}: {e.response.text[:200]}") from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"TfL request failed: {e}") from e

        arrivals = [
            Arrival(
                route=item.get("lineName", "unknown"),
                stop_id=stop_id,
                vehicle_id=item.get("vehicleId", "unknown"),
                eta_seconds=max(int(item.get("timeToStation", 0)), 0),
            )
            for item in data
        ]
        arrivals.sort(key=lambda a: a.eta_seconds)
        return arrivals
