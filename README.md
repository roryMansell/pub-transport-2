# bus-rt-minimal

A fresh, minimal starting point for a real-time bus routes service. Start tiny, then build upwards.

## Guiding Principles
- Keep it boring and simple: one API, one provider (mock), zero DB, zero queues.
- Fast to run locally: `uvicorn` + `httpx`. No Docker required (add later if wanted).
- Pluggable providers so we can swap a real feed (e.g., GTFS-RT/SIRI/TfL) without rewriting the app.

## Quickstart

```bash
# 1) Create and activate a virtualenv (any method you like)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run API (hot reload for dev)
uvicorn app.main:app --reload

# 4) Hit endpoints
curl 'http://127.0.0.1:8000/health'
curl 'http://127.0.0.1:8000/arrivals?stop_id=STOP_123'
```

## Configuration
Copy `.env.example` to `.env` and tweak if needed.

- `PROVIDER`: which provider module to use. Defaults to `mock`.
- `PROVIDER_OPTS`: JSON string with provider-specific options (optional).

## Endpoints

- `GET /health` → `{"status":"ok"}`
- `GET /arrivals?stop_id=...` → List of arrivals for a stop.

## Project Structure
```
bus-rt-minimal/
  ├─ src/app/
  │  ├─ main.py
  │  ├─ models.py
  │  └─ providers/
  │     ├─ base.py
  │     └─ mock.py
  ├─ tests/
  │  └─ test_health.py
  ├─ .env.example
  ├─ requirements.txt
  ├─ .gitignore
  └─ README.md
```

## Roadmap (add as we grow)
- Provider: add a real feed adapter (e.g., GTFS-RT VehiclePositions).
- Caching: simple in-memory TTL cache for stable polling.
- Frontend: tiny Leaflet map that consumes `/arrivals`.
- Docker: dev-compose for API + static site, optional Redis cache.
- CI: lint + tests + type-check.
