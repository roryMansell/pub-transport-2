from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Arrival(BaseModel):
    route: str = Field(..., description="Public-facing route/line identifier")
    stop_id: str = Field(..., description="Stop identifier as used by provider")
    vehicle_id: str = Field(..., description="Unique vehicle identifier (provider-specific)")
    eta_seconds: int = Field(..., ge=0, description="Estimated time until arrival in seconds")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the estimate")

class ArrivalsResponse(BaseModel):
    stop_id: str
    arrivals: List[Arrival]
