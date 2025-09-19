from typing import List, Protocol
from . import types  # Optional future shared types

from ..models import Arrival

class Provider(Protocol):
    async def get_arrivals(self, stop_id: str) -> List[Arrival]:
        ...
