import os
import json
import importlib
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from datetime import datetime, timezone

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

@app.get("/line/{line_id}/shape")
async def line_shape(line_id: str):
    """Return TfL 'lineStrings' geometry for a bus line (inbound/outbound)."""
    def build():
        with httpx.Client(timeout=20.0, headers=HTTP_HEADERS) as c:
            rin = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params()); rin.raise_for_status()
            rout = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params()); rout.raise_for_status()
            return rin.json(), rout.json()

    inbound_json, outbound_json = memo(f"rawseq:{line_id}", 300, build)

    def extract_line_strings(seq_json):
        # TfL returns lineStrings as arrays of "lat,lon lat,lon ..." or arrays of [lat,lon] pairs depending on mode/version.
        # We’ll normalize to [[lat, lon], ...] arrays for Leaflet.
        ls = []
        for s in seq_json.get("stopPointSequences", []):
            for raw in s.get("lineStrings", []) or []:
                if isinstance(raw, str):
                    # "lat,lon lat,lon ..." -> split -> floats
                    pts = []
                    for token in raw.strip().split(" "):
                        if not token: continue
                        latlon = token.split(",")
                        if len(latlon) == 2:
                            lat = float(latlon[0]); lon = float(latlon[1])
                            pts.append([lat, lon])
                    if len(pts) >= 2:
                        ls.append(pts)
                elif isinstance(raw, list):
                    # [[lat,lon], [lat,lon], ...] OR [[lon,lat], ...] -> detect order
                    if len(raw) >= 2 and len(raw[0]) == 2:
                        # simple heuristic: London lon ~ -0.x to -0.6; lat ~ 51.x
                        a, b = raw[0]
                        looks_latlon = (50 <= a <= 52) and (-1.5 <= b <= 0.5)
                        pts = [[p[0], p[1]] if looks_latlon else [p[1], p[0]] for p in raw]
                        ls.append(pts)
        return ls

    inbound = extract_line_strings(inbound_json)
    outbound = extract_line_strings(outbound_json)

    # Deduplicate very-similar lines (simple length-based heuristic)
    def dedupe(lines):
        seen = set(); out = []
        for pts in lines:
            key = (len(pts), round(pts[0][0], 5), round(pts[0][1], 5), round(pts[-1][0], 5), round(pts[-1][1], 5))
            if key in seen: continue
            seen.add(key); out.append(pts)
        return out

    return {
        "line_id": line_id,
        "inbound": dedupe(inbound),
        "outbound": dedupe(outbound),
    }


@app.get("/vehicles")
async def vehicles(line_id: str = Query(..., description="TfL bus line id, e.g. 88")):
    """Approximate positions by interpolating from previous stop -> next stop using timeToStation."""
    async with httpx.AsyncClient(timeout=12.0, headers=HTTP_HEADERS) as client:
        r = await client.get(f"{TFL_BASE}/Line/{line_id}/Arrivals", params=tfl_params())
        r.raise_for_status()
        arr = r.json()

    # Cache the ordered stop sequences for 5 minutes
    def build_sequences():
        with httpx.Client(timeout=20.0, headers=HTTP_HEADERS) as c:
            rin = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params()); rin.raise_for_status()
            rout = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params()); rout.raise_for_status()
            in_json, out_json = rin.json(), rout.json()

        def flatten(seq_json):
            out = []
            for s in seq_json.get("stopPointSequences", []):
                for sp in s.get("stopPoint", []):
                    out.append({"id": sp["id"], "name": sp.get("name"), "lat": sp.get("lat"), "lon": sp.get("lon")})
            # de-dup preserving order
            seen, ordered = set(), []
            for sp in out:
                if sp["id"] in seen: continue
                seen.add(sp["id"]); ordered.append(sp)
            return ordered

        return {"inbound": flatten(in_json), "outbound": flatten(out_json)}

    sequences = memo(f"seq:{line_id}", 300, build_sequences)
    fetched_at = datetime.now(timezone.utc).isoformat()

    def seq_for(direction: str):
        d = (direction or "").lower()
        if d.startswith("in"): return sequences["inbound"]
        if d.startswith("out"): return sequences["outbound"]
        return sequences["inbound"]

    vehicles = []
    for it in arr:
        next_stop_id = it.get("naptanId") or it.get("stationId") or it.get("id")
        direction = it.get("direction")
        eta = max(int(it.get("timeToStation", 0)), 0)
        route = it.get("lineName")
        vehicle_id = it.get("vehicleId")

        seq = seq_for(direction)
        idx = None
        for i, sp in enumerate(seq):
            if sp["id"] == next_stop_id:
                idx = i
                break

        prev_sp = seq[idx-1] if (idx is not None and idx > 0) else None
        next_sp = seq[idx] if (idx is not None) else None

        def val(sp, key):
            return None if sp is None else sp.get(key)

        vehicles.append({
            "line_id": line_id,
            "route": route,
            "vehicle_id": vehicle_id,
            "direction": direction,
            "eta_seconds": eta,
            "fetched_at": fetched_at,
            "next_stop_id": next_stop_id,
            "next_stop_name": val(next_sp, "name"),
            # legacy fields (next stop coords)
            "lat": val(next_sp, "lat"),
            "lon": val(next_sp, "lon"),
            # interpolation fields
            "prev_stop_id": val(prev_sp, "id"),
            "prev_stop_name": val(prev_sp, "name"),
            "prev_lat": val(prev_sp, "lat"),
            "prev_lon": val(prev_sp, "lon"),
            "next_lat": val(next_sp, "lat"),
            "next_lon": val(next_sp, "lon"),
        })

    vehicles.sort(key=lambda v: v["eta_seconds"])
    return {"line_id": line_id, "fetched_at": fetched_at, "vehicles": vehicles}
