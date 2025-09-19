import os
import json
import importlib
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime, timezone

from .models import Arrival, ArrivalsResponse

# =========================
# App & CORS
# =========================
load_dotenv()

app = FastAPI(title="bus-rt-minimal", version="0.3.0")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],  # e.g. https://rorymansell.github.io
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Provider loader (for /arrivals)
# =========================
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

# =========================
# Basic endpoints
# =========================
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

# =========================
# TfL constants / helpers
# =========================
TFL_BASE = "https://api.tfl.gov.uk"
HTTP_HEADERS = {"User-Agent": "bus-rt-minimal/0.3.0"}
TFL_APP_KEY = os.getenv("TFL_APP_KEY")
TFL_APP_ID = os.getenv("TFL_APP_ID")

def tfl_params() -> Dict[str, str]:
    p: Dict[str, str] = {}
    if TFL_APP_KEY:
        p["app_key"] = TFL_APP_KEY
    if TFL_APP_ID:
        p["app_id"] = TFL_APP_ID
    return p

# tiny in-proc memo cache (key -> (ts, value))
_cache: Dict[str, Tuple[float, Any]] = {}

def memo(key: str, ttl: int, builder):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = builder()
    _cache[key] = (now, val)
    return val

# =========================
# Routes list
# =========================
@app.get("/routes")
async def routes():
    try:
        # cache for 5 minutes
        def build():
            with httpx.Client(timeout=15.0, headers=HTTP_HEADERS) as c:
                r = c.get(f"{TFL_BASE}/Line/Mode/bus", params=tfl_params())
                r.raise_for_status()
                data = r.json()
            return [{"id": x["id"], "name": x.get("name", x["id"])} for x in data]

        data = memo("routes:list", 300, build)
        return data

    except httpx.HTTPStatusError as e:
        txt = (e.response.text or "")[:500]
        raise HTTPException(status_code=502, detail=f"TfL HTTP {e.response.status_code}: {txt}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"TfL request failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.get("/debug/tfl")
async def debug_tfl():
    try:
        with httpx.Client(timeout=10.0, headers=HTTP_HEADERS) as c:
            r = c.get(f"{TFL_BASE}/Line/Mode/bus", params=tfl_params())
            status = r.status_code
            ok = r.status_code == 200
            count = len(r.json()) if ok else None
            return {"ok": ok, "status": status, "count": count}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# =========================
# Stops (ordered for in/out)
# =========================
@app.get("/line/{line_id}/stops")
async def line_stops(line_id: str):
    """Ordered stops for inbound/outbound, cached ~10 minutes."""

    def build_raw():
        with httpx.Client(timeout=20.0, headers=HTTP_HEADERS) as c:
            r_in = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params()); r_in.raise_for_status()
            r_out = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params()); r_out.raise_for_status()
            return r_in.json(), r_out.json()

    inbound_json, outbound_json = memo(f"rawstops:{line_id}", 600, build_raw)

    def simplify(seq_json):
        out = []
        for s in seq_json.get("stopPointSequences", []):
            for sp in s.get("stopPoint", []):
                out.append({
                    "id": sp["id"],
                    "name": sp.get("name"),
                    "lat": sp.get("lat"),
                    "lon": sp.get("lon"),
                })
        return out

    return {
        "line_id": line_id,
        "inbound": simplify(inbound_json),
        "outbound": simplify(outbound_json),
    }


# =========================
# Road-following geometry (lineStrings)
# =========================

ORS_BASE = "https://api.openrouteservice.org/v2/directions/driving-car"
ORS_TOKEN = os.getenv("ORS_TOKEN")

@app.get("/line/{line_id}/shape")
async def line_shape(line_id: str):
    """
    Road-following shape using ORS Directions (no cache).
    Falls back to straight stops if ORS fails.
    """
    ORS_BASE = "https://api.openrouteservice.org/v2/directions/driving-car"
    ORS_TOKEN = os.getenv("ORS_TOKEN")

    def simplify(seq_json):
        pts = []
        for s in (seq_json or {}).get("stopPointSequences", []):
            for sp in s.get("stopPoint", []):
                lat, lon = sp.get("lat"), sp.get("lon")
                if lat is not None and lon is not None:
                    pts.append((float(lat), float(lon)))
        return pts

    async def fetch_seq(direction: str):
        url = f"{TFL_BASE}/Line/{line_id}/Route/Sequence/{direction}"
        async with httpx.AsyncClient(timeout=20.0, headers=HTTP_HEADERS) as c:
            r = await c.get(url, params=tfl_params())
            r.raise_for_status()
            return r.json()

    inbound_json, outbound_json = await fetch_seq("inbound"), await fetch_seq("outbound")
    inbound_pts, outbound_pts = simplify(inbound_json), simplify(outbound_json)

    def decode_polyline(enc):
        """Decode ORS encoded polyline to [[lat, lon], ...]."""
        coords, index, lat, lon = [], 0, 0, 0
        while index < len(enc):
            shift, result = 0, 0
            while True:
                b = ord(enc[index]) - 63; index += 1
                result |= (b & 0x1f) << shift; shift += 5
                if b < 0x20: break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat
            shift, result = 0, 0
            while True:
                b = ord(enc[index]) - 63; index += 1
                result |= (b & 0x1f) << shift; shift += 5
                if b < 0x20: break
            dlon = ~(result >> 1) if (result & 1) else (result >> 1)
            lon += dlon
            coords.append([lat / 1e5, lon / 1e5])
        return coords

    async def ors_route(points_latlon):
        if len(points_latlon) < 2:
            return [], "too-few-points"
        if not ORS_TOKEN:
            return [], "no-token"

        coords = [[lon, lat] for (lat, lon) in points_latlon]
        body = {"coordinates": coords, "instructions": False}
        headers = {"Authorization": ORS_TOKEN, "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(ORS_BASE, json=body, headers=headers)

        if r.status_code == 429:
            return [], "quota"
        if r.status_code >= 400:
            return [], f"bad-request:{r.text[:120]}"

        data = r.json()
        geom = data.get("routes", [{}])[0].get("geometry")

        if isinstance(geom, dict) and "coordinates" in geom:
            return [[lat, lon] for lon, lat in geom["coordinates"]], "ors-geojson"
        elif isinstance(geom, str):
            coords = decode_polyline(geom)
            return coords, "ors-polyline"
        return [], "no-geometry"

    in_shape, in_src = await ors_route(inbound_pts)
    out_shape, out_src = await ors_route(outbound_pts)

    if not in_shape and inbound_pts:
        in_shape, in_src = inbound_pts, "fallback"
    if not out_shape and outbound_pts:
        out_shape, out_src = outbound_pts, "fallback"

    return {
        "line_id": line_id,
        "inbound": [in_shape] if in_shape else [],
        "outbound": [out_shape] if out_shape else [],
        "meta": {"in_source": in_src, "out_source": out_src}
    }



# =========================
# Vehicles with prev/next stop for interpolation
# =========================
@app.get("/vehicles")
async def vehicles(line_id: str = Query(..., description="TfL bus line id, e.g. 88")):
    """Approximate positions by interpolating from previous stop -> next stop using timeToStation."""
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=HTTP_HEADERS) as client:
            r = await client.get(f"{TFL_BASE}/Line/{line_id}/Arrivals", params=tfl_params())
            r.raise_for_status()
            arr = r.json()
    except httpx.HTTPStatusError as e:
        txt = (e.response.text or "")[:300]
        raise HTTPException(status_code=502, detail=f"TfL HTTP {e.response.status_code}: {txt}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"TfL request failed: {e}")

    # Cache the ordered stop sequences for 5 minutes
    def build_sequences():
        with httpx.Client(timeout=20.0, headers=HTTP_HEADERS) as c:
            rin = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params()); rin.raise_for_status()
            rout = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params()); rout.raise_for_status()
            in_json, out_json = rin.json(), rout.json()

        def flatten(seq_json):
            out = []
            for s in (seq_json or {}).get("stopPointSequences", []):
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

    def seq_for(direction: Optional[str]):
        d = (direction or "").lower()
        if d.startswith("in"): return sequences["inbound"]
        if d.startswith("out"): return sequences["outbound"]
        return sequences["inbound"]

    vehicles_out: List[Dict[str, Any]] = []
    for it in arr:
        next_stop_id = it.get("naptanId") or it.get("stationId") or it.get("id")
        direction = it.get("direction")
        eta = max(int(it.get("timeToStation", 0) or 0), 0)
        route = it.get("lineName")
        vehicle_id = it.get("vehicleId")

        seq = seq_for(direction)
        idx = None
        for i, sp in enumerate(seq):
            if sp["id"] == next_stop_id:
                idx = i
                break

        prev_sp = seq[idx - 1] if (idx is not None and idx > 0) else None
        next_sp = seq[idx] if (idx is not None) else None

        def val(sp, key):
            return None if sp is None else sp.get(key)

        vehicles_out.append({
            "line_id": line_id,
            "route": route,
            "vehicle_id": vehicle_id,
            "direction": direction,
            "eta_seconds": eta,
            "fetched_at": fetched_at,
            "next_stop_id": next_stop_id,
            "next_stop_name": val(next_sp, "name"),
            # legacy (next stop coords)
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

    vehicles_out.sort(key=lambda v: v["eta_seconds"])
    return {"line_id": line_id, "fetched_at": fetched_at, "vehicles": vehicles_out}

@app.get("/debug/ors")
def debug_ors():
    """Check ORS token + show the exact response."""
    import os, httpx, json
    token = os.getenv("ORS_TOKEN")
    if not token:
        return {"ok": False, "reason": "ORS_TOKEN not set in environment"}

    body = {
        "coordinates": [[-0.1278, 51.5074], [-0.1, 51.51]],  # [lon,lat]
        "format": "geojson"
    }
    try:
        r = httpx.post(
            "https://api.openrouteservice.org/v2/directions/driving-car",
            json=body,
            headers={"Authorization": token, "Content-Type": "application/json"},
            timeout=20.0,
        )
        out = {
            "ok": r.status_code == 200,
            "status": r.status_code,
            "headers": dict(r.headers),
        }
        # Try to parse JSON; otherwise return text snippet
        try:
            j = r.json()
            out["json_keys"] = list(j.keys())
            # If success, count points
            if r.status_code == 200 and "features" in j and j["features"]:
                pts = len(j["features"][0]["geometry"]["coordinates"])
                out["points_in_route"] = pts
        except Exception:
            out["body_snippet"] = r.text[:500]
        return out
    except Exception as e:
        return {"ok": False, "reason": str(e)}

@app.get("/debug/ors-route")
async def debug_ors_route():
    import httpx
    coords = [[-0.10355, 51.42952], [-0.16276, 51.5223]]  # start+end of route 2 inbound
    headers = {"Authorization": ORS_TOKEN}
    body = {"coordinates": coords, "format": "geojson"}
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(ORS_BASE, json=body, headers=headers)
        return {"status": r.status_code, "json": r.json()}