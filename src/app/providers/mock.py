from typing import List
from datetime import datetime, timedelta
from ..models import Arrival

class MockProvider:
    def __init__(self, **kwargs):
        # kwargs may contain provider options (e.g., region) — unused here
        pass

    async def get_arrivals(self, stop_id: str) -> List[Arrival]:
        now = datetime.utcnow()
        # A couple of deterministic "arrivals" for demo
        return [
            Arrival(
                route="12",
                stop_id=stop_id,
                vehicle_id="MOCK-12-001",
                eta_seconds=90,
                last_updated=now,
            ),
            Arrival(
                route="12",
                stop_id=stop_id,
                vehicle_id="MOCK-12-007",
                eta_seconds=420,
                last_updated=now,
            ),
        ]
