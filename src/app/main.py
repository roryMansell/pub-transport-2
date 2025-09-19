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
    """Ordered stops for inbound/outbound."""
    try:
        with httpx.Client(timeout=20.0, headers=HTTP_HEADERS) as c:
            r_in = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params()); r_in.raise_for_status()
            r_out = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params()); r_out.raise_for_status()
            inbound = r_in.json(); outbound = r_out.json()

        def simplify(seq_json):
            out = []
            for s in (seq_json or {}).get("stopPointSequences", []):
                for sp in s.get("stopPoint", []):
                    out.append({
                        "id": sp["id"],
                        "name": sp.get("name"),
                        "lat": sp.get("lat"),
                        "lon": sp.get("lon"),
                    })
            return out

        return {"line_id": line_id, "inbound": simplify(inbound), "outbound": simplify(outbound)}

    except httpx.HTTPStatusError as e:
        txt = (e.response.text or "")[:400]
        raise HTTPException(status_code=502, detail=f"TfL HTTP {e.response.status_code}: {txt}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"TfL request failed: {e}")

# =========================
# Road-following geometry (lineStrings)
# =========================
@app.get("/line/{line_id}/shape")
async def line_shape(line_id: str):
    """
    Return road-following 'lineStrings' for inbound/outbound as [[lat, lon], ...] lists.
    TfL places geometry in several different spots and coordinate orders; we check them all.
    """

    def fetch_all():
        with httpx.Client(timeout=25.0, headers=HTTP_HEADERS) as c:
            # Route/Sequence (dir-specific)
            rin = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/inbound", params=tfl_params()); rin.raise_for_status()
            rout = c.get(f"{TFL_BASE}/Line/{line_id}/Route/Sequence/outbound", params=tfl_params()); rout.raise_for_status()
            inbound_seq, outbound_seq = rin.json(), rout.json()

            # Generic Route endpoint (sometimes only here)
            r = c.get(f"{TFL_BASE}/Line/{line_id}/Route", params=tfl_params()); r.raise_for_status()
            route_root = r.json()
        return inbound_seq, outbound_seq, route_root

    inbound_seq, outbound_seq, route_root = memo(f"rawshape:{line_id}", 300, fetch_all)

    def norm_pair(a: float, b: float) -> List[float]:
        # If looks like [lon,lat] (London lon ~ -1.5..0.5, lat ~ 50..52.5), swap.
        if -1.5 <= a <= 0.5 and 50.0 <= b <= 52.5:
            return [b, a]
        return [a, b]

    def parse_string_ls(s: str) -> Optional[List[List[float]]]:
        # Accept forms like "51.5,-0.12 51.51,-0.13" or "[-0.12,51.5] [-0.13,51.51]"
        pts: List[List[float]] = []
        s = s.replace("[", "").replace("]", "").strip()
        if not s:
            return None
        for token in s.split():
            parts = token.split(",")
            if len(parts) != 2:
                continue
            try:
                a = float(parts[0]); b = float(parts[1])
            except ValueError:
                continue
            pts.append(norm_pair(a, b))
        return pts if len(pts) >= 2 else None

    def parse_list_ls(raw_list: List[Any]) -> Optional[List[List[float]]]:
        # raw_list: [[a,b],[a,b],...] but order might be [lon,lat]
        pts: List[List[float]] = []
        for p in raw_list:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                return None
            try:
                a = float(p[0]); b = float(p[1])
            except (TypeError, ValueError):
                return None
            pts.append(norm_pair(a, b))
        return pts if len(pts) >= 2 else None

    def collect_from(obj: Dict[str, Any]) -> List[List[List[float]]]:
        out: List[List[List[float]]] = []
        # 1) top-level lineStrings
        for raw in (obj.get("lineStrings") or []):
            pts = parse_string_ls(raw) if isinstance(raw, str) else parse_list_ls(raw) if isinstance(raw, list) else None
            if pts: out.append(pts)
        # 2) stopPointSequences[*].lineStrings
        for s in (obj.get("stopPointSequences") or []):
            for raw in (s.get("lineStrings") or []):
                pts = parse_string_ls(raw) if isinstance(raw, str) else parse_list_ls(raw) if isinstance(raw, list) else None
                if pts: out.append(pts)
        # 3) orderedLineRoutes[*].lineStrings
        for r in (obj.get("orderedLineRoutes") or []):
            for raw in (r.get("lineStrings") or []):
                pts = parse_string_ls(raw) if isinstance(raw, str) else parse_list_ls(raw) if isinstance(raw, list) else None
                if pts: out.append(pts)
        return out

    inbound_lines = collect_from(inbound_seq)
    outbound_lines = collect_from(outbound_seq)

    # 4) Fallback: /Line/{id}/Route → routeSections[].lineStrings
    # This endpoint isn't split by direction, so we just stuff any geometry we find into both if empty.
    def collect_from_route_root(root: Any) -> List[List[List[float]]]:
        out: List[List[List[float]]] = []
        for sec in (root.get("routeSections") or []):
            for raw in (sec.get("lineStrings") or []):
                pts = parse_string_ls(raw) if isinstance(raw, str) else parse_list_ls(raw) if isinstance(raw, list) else None
                if pts: out.append(pts)
        return out

    if not inbound_lines or not outbound_lines:
        # /Line/{id}/Route can be an array or an object depending on API; handle both
        route_objs = route_root if isinstance(route_root, list) else [route_root]
        extra: List[List[List[float]]] = []
        for obj in route_objs:
            if isinstance(obj, dict):
                extra.extend(collect_from_route_root(obj))
        if not inbound_lines:
            inbound_lines = extra[:]
        if not outbound_lines:
            outbound_lines = extra[:]

    # Deduplicate near-duplicates (length + endpoints)
    def dedupe(lines: List[List[List[float]]]) -> List[List[List[float]]]:
        seen = set(); result = []
        for pts in lines:
            key = (len(pts),
                   round(pts[0][0], 5), round(pts[0][1], 5),
                   round(pts[-1][0], 5), round(pts[-1][1], 5))
            if key in seen: continue
            seen.add(key); result.append(pts)
        return result

    return {
        "line_id": line_id,
        "inbound": dedupe(inbound_lines),
        "outbound": dedupe(outbound_lines),
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
