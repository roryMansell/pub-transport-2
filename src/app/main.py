import os
import json
import importlib
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .models import Arrival, ArrivalsResponse

# -------------------------
# App & CORS (define FIRST)
# -------------------------
load_dotenv()

app = FastAPI(title="bus-rt-minimal", version="0.2.0")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Provider loader (mock/TfL)
# -------------------------
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
    class_name = f"{provider_name.capitalize()}Provider"
    if not hasattr(module, class_name):
        class_name = "Provider"
    ProviderClass = getattr(module, class_name, None)
    if ProviderClass is None:
        raise RuntimeError(f"Provider class not found in module '{provider_name}'")
    return ProviderClass(**opts)

provider = load_provider()

# -------------------------
# Basic endpoints
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/arrivals", response_model=ArrivalsResponse)
async def arrivals(stop_id: str = Query(..., description="Provider stop id")):
    try:
        items = await provider.get_arrivals(stop_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"provider error: {e}")
    return ArrivalsResponse(stop_id=stop_id, arrivals=items)

# -------------------------
# TfL-backed helper endpoints
# -------------------------
TFL_BASE = "https://api.tfl.gov.uk"
HTTP_HEADERS = {"User-Agent": "bus-rt-minimal/0.2.0"}
TFL_APP_KEY = os.getenv("TFL_APP_KEY")
TFL_APP_ID = os.getenv("TFL_APP_ID")

def tfl_params() -> Dict[str, str]:
    p: Dict[str, str] = {}
    if TFL_APP_KEY:
        p["app_key"] = TFL_APP_KEY
    if TFL_APP_ID:
        p["app_id"] = TFL_APP_ID
    return p

# tiny in-proc memo cache for infrequently changing data
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
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=HTTP_HEADERS) as client:
            r = await client.get(f"{TFL_BASE}/Line/Mode/bus", params=tfl_params())
            r.raise_for_status()
            data = r.json()
        return [{"id": x["id"], "name": x.get("name", x["id"])} for x in data]
    except httpx.HTTPStatusError as e:
        # Show TfL error details (most common: 401/403/429)
        txt = (e.response.text or "")[:500]
        raise HTTPException(status_code=502, detail=f"TfL HTTP {e.response.status_code}: {txt}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"TfL request failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.get("/debug/tfl")
async def debug_tfl():
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=HTTP_HEADERS) as client:
            r = await client.get(f"{TFL_BASE}/Line/Mode/bus", params=tfl_params())
            status = r.status_code
            ok = r.status_code == 200
            count = len(r.json()) if ok else None
            return {"ok": ok, "status": status, "count": count}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/line/{line_id}/stops")
async def line_stops(line_id: str):
    """Ordered stops for inbound/outbound."""
    async with httpx.AsyncClient(timeout=20.0, headers=HTTP_HEADERS) as client:
        r_in = await client.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params())
        r_out = await client.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params())
        r_in.raise_for_status(); r_out.raise_for_status()
        inbound = r_in.json(); outbound = r_out.json()

    def simplify(seq_json):
        out = []
        for s in seq_json.get("stopPointSequences", []):
            for sp in s.get("stopPoint", []):
                out.append({
                    "id": sp["id"], "name": sp.get("name"),
                    "lat": sp.get("lat"), "lon": sp.get("lon"),
                })
        return out

    return {"line_id": line_id, "inbound": simplify(inbound), "outbound": simplify(outbound)}

@app.get("/vehicles")
async def vehicles(line_id: str = Query(..., description="TfL bus line id, e.g. 88")):
    """Approximate vehicle positions by placing them at their next stop."""
    async with httpx.AsyncClient(timeout=12.0, headers=HTTP_HEADERS) as client:
        r = await client.get(f"{TFL_BASE}/Line/{line_id}/Arrivals", params=tfl_params())
        r.raise_for_status()
        arr = r.json()

        # Build or reuse stop map for this line (5 min)
        def build_stop_map():
            # NOTE: we are in a sync wrapper for memo(); build with already-fetched HTTP client via .run()
            # but since memo expects sync, we perform blocking fetch with httpx (open new client).
            with httpx.Client(timeout=20.0, headers=HTTP_HEADERS) as c:
                rin = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params()); rin.raise_for_status()
                rout = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params()); rout.raise_for_status()
                in_json = rin.json(); out_json = rout.json()
            def to_map(seq_json):
                m = {}
                for s in seq_json.get("stopPointSequences", []):
                    for sp in s.get("stopPoint", []):
                        m[sp["id"]] = (sp.get("lat"), sp.get("lon"), sp.get("name"))
                return m
            m = to_map(in_json); m.update(to_map(out_json))
            return m

        stop_map = memo(f"stops:{line_id}", 300, build_stop_map)

    vehicles: List[Dict[str, Any]] = []
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
            "lat": lat, "lon": lon,
            "direction": it.get("direction"),
        })
    vehicles.sort(key=lambda v: v["eta_seconds"])
    return {"line_id": line_id, "vehicles": vehicles}
