import os, json, importlib
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


from .models import Arrival, ArrivalsResponse

# --- add near top ---
import httpx
from fastapi import Query
from typing import List, Dict, Any
import functools
import time
from .models import Arrival

TFL_BASE = "https://api.tfl.gov.uk"
HTTP_HEADERS = {"User-Agent": "bus-rt-minimal/0.1"}
TFL_KEY = os.getenv("TFL_APP_KEY")
TFL_ID = os.getenv("TFL_APP_ID")

def tfl_params():
    p = {}
    if TFL_KEY: p["app_key"] = TFL_KEY
    if TFL_ID: p["app_id"] = TFL_ID
    return p

# tiny memo cache for 5 minutes (routes & stop lists change rarely)
_cache: Dict[str, tuple[float, Any]] = {}
def memo(key: str, ttl: int, builder):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = builder()
    _cache[key] = (now, val)
    return val

@app.get("/routes")
async def routes():
    async with httpx.AsyncClient(timeout=15.0, headers=HTTP_HEADERS) as client:
        # https://api.tfl.gov.uk/Line/Mode/bus
        r = await client.get(f"{TFL_BASE}/Line/Mode/bus", params=tfl_params())
        r.raise_for_status()
        data = r.json()
    # Keep it light: id + name only
    return [{"id": x["id"], "name": x.get("name", x["id"])} for x in data]

@app.get("/line/{line_id}/stops")
async def line_stops(line_id: str):
    async with httpx.AsyncClient(timeout=20.0, headers=HTTP_HEADERS) as client:
        # route sequence gives ordered stops & lat/lon
        async def seq(direction: str):
            r = await client.get(
                f"{TFL_BASE}/Line/{line_id}/Route/Sequence/{direction}",
                params=tfl_params()
            )
            r.raise_for_status()
            return r.json()

        inbound = await seq("inbound")
        outbound = await seq("outbound")

    def simplify(seq):
        stops = seq.get("stopPointSequences", [])
        out = []
        for s in stops:
            for sp in s.get("stopPoint", []):
                out.append({
                    "id": sp["id"], "name": sp.get("name"),
                    "lat": sp.get("lat"), "lon": sp.get("lon")
                })
        return out

    return {
        "line_id": line_id,
        "inbound": simplify(inbound),
        "outbound": simplify(outbound),
    }

@app.get("/vehicles")
async def vehicles(line_id: str = Query(..., description="TfL bus line id, e.g. 88")):
    async with httpx.AsyncClient(timeout=10.0, headers=HTTP_HEADERS) as client:
        r = await client.get(
            f"{TFL_BASE}/Line/{line_id}/Arrivals",
            params=tfl_params()
        )
        r.raise_for_status()
        arr = r.json()

    # Build a stopId -> (lat, lon, name) map (memoized for 5 min)
    def build_stop_map():
        # we’ll reuse the line_stops endpoint logic
        import asyncio
        async def fetch():
            async with httpx.AsyncClient(timeout=20.0, headers=HTTP_HEADERS) as c:
                r_in = await c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params())
                r_out = await c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params())
                r_in.raise_for_status(); r_out.raise_for_status()
                return r_in.json(), r_out.json()
        in_json, out_json = asyncio.run(fetch())

        def to_map(seq_json):
            m = {}
            for s in seq_json.get("stopPointSequences", []):
                for sp in s.get("stopPoint", []):
                    m[sp["id"]] = (sp.get("lat"), sp.get("lon"), sp.get("name"))
            return m
        m = to_map(in_json)
        m.update(to_map(out_json))
        return m

    stop_map = memo(f"stops:{line_id}", 300, build_stop_map)

    vehicles = []
    for it in arr:
        stop_id = it.get("naptanId") or it.get("stationId") or it.get("id")
        lat, lon, name = (None, None, None)
        if stop_id and stop_id in stop_map:
            lat, lon, name = stop_map[stop_id]
        vehicles.append({
            "line_id": line_id,
            "route": it.get("lineName"),
            "vehicle_id": it.get("vehicleId"),
            "eta_seconds": max(int(it.get("timeToStation", 0)), 0),
            "next_stop_id": stop_id,
            "next_stop_name": name,
            "lat": lat, "lon": lon,  # rough: at next stop
            "direction": it.get("direction"),
        })

    # soonest-first
    vehicles.sort(key=lambda v: v["eta_seconds"])
    return {"line_id": line_id, "vehicles": vehicles}


load_dotenv()

app = FastAPI(title="bus-rt-minimal", version="0.1.0")

def load_provider():
    provider_name = os.getenv("PROVIDER", "mock")
    opts_raw = os.getenv("PROVIDER_OPTS", "{}")
    try:
        opts = json.loads(opts_raw)
    except json.JSONDecodeError:
        opts = {}
    try:
        module = importlib.import_module(f"app.providers.{provider_name}")
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Provider module not found: {provider_name}") from e
    # Expect a class named <Name>Provider, e.g., MockProvider
    class_name = f"{provider_name.capitalize()}Provider"
    if not hasattr(module, class_name):
        # Fallback: if provider module exposes "Provider" symbol
        class_name = "Provider"
    ProviderClass = getattr(module, class_name, None)
    if ProviderClass is None:
        raise RuntimeError(f"Provider class not found in module '{provider_name}'")
    return ProviderClass(**opts)

provider = load_provider()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/arrivals", response_model=ArrivalsResponse)
async def arrivals(stop_id: str = Query(..., description="Provider stop id")):
    if not stop_id:
        raise HTTPException(status_code=400, detail="stop_id is required")
    try:
        arrivals = await provider.get_arrivals(stop_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"provider error: {e}")
    return ArrivalsResponse(stop_id=stop_id, arrivals=arrivals)

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")  # dev default

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)