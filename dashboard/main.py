"""
main.py — PPE Supervisor Monitoring Dashboard (FastAPI Backend)
================================================================
Runs on the Jetson Orin Nano at http://0.0.0.0:8000

Features:
  - REST endpoints for daily analytics and event history
  - WebSocket broadcaster for real-time violation alerts
  - Webhook receiver endpoint (POST /webhook/violation)
  - Static file serving for JPEG snapshots
  - JSONL log file watcher (polls every 2 seconds)

Install:
  pip3 install fastapi uvicorn aiofiles watchdog python-multipart

Run:
  cd ~/ppe_detection
  python3 dashboard/main.py

  OR with uvicorn directly:
  uvicorn dashboard.main:app --host 0.0.0.0 --port 8000 --reload
================================================================
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Configuration ──────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent   # ~/ppe_detection
LOG_DIR       = BASE_DIR / "logs" / "events"
SNAPSHOT_DIR  = BASE_DIR / "logs" / "snapshots"
TEMPLATE_DIR  = Path(__file__).resolve().parent / "templates"

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="PPE Supervisor Dashboard",
    description="Real-time PPE violation monitoring for NVIDIA Jetson Orin Nano",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve snapshot JPEGs as static files
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")


# ── WebSocket Connection Manager ───────────────────────────
class ConnectionManager:
    """Manages active WebSocket connections and broadcasts events."""

    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] Client connected — total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        print(f"[WS] Client disconnected — total: {len(self.active)}")

    async def broadcast(self, data: dict):
        """Send JSON payload to all connected clients."""
        message = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── JSONL Log Utilities ────────────────────────────────────
def today_log_path() -> Path:
    return LOG_DIR / f"ppe_violations_{date.today().isoformat()}.jsonl"


def parse_log_file(log_path: Path) -> List[dict]:
    """Read and parse a JSONL log file, newest first."""
    if not log_path.exists():
        return []
    events = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return list(reversed(events))   # newest first


def compute_analytics(events: List[dict]) -> dict:
    """
    Compute daily KPI counters from a list of violation events.

    Returns:
        total          — total violation events today
        by_ppe         — {helmet: N, vest: N, gloves: N}
        by_zone        — {zone_id: N, ...}
        hourly         — {0..23: N} counts per hour
        latest_time    — ISO timestamp of most recent event
    """
    total    = len(events)
    by_ppe   = defaultdict(int)
    by_zone  = defaultdict(int)
    hourly   = defaultdict(int)

    for ev in events:
        for item in ev.get("missing_ppe", []):
            by_ppe[item.lower()] += 1
        zone = ev.get("zone_id", "unknown")
        by_zone[zone] += 1
        ts = ev.get("timestamp", "")
        if ts:
            try:
                hour = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
                hourly[hour] += 1
            except ValueError:
                pass

    latest_time = events[0].get("timestamp", "") if events else ""

    return {
        "total":       total,
        "by_ppe":      dict(by_ppe),
        "by_zone":     dict(by_zone),
        "hourly":      {str(h): hourly.get(h, 0) for h in range(24)},
        "latest_time": latest_time,
        "date":        date.today().isoformat(),
    }


def enrich_event(ev: dict) -> dict:
    """
    Add a web-accessible snapshot URL to an event dict.
    Converts the absolute snapshot_path to a /snapshots/<filename> URL.
    """
    snap = ev.get("snapshot_path", "")
    if snap:
        ev["snapshot_url"] = f"/snapshots/{Path(snap).name}"
    else:
        ev["snapshot_url"] = None
    return ev


# ── Background Log Watcher ─────────────────────────────────
async def log_watcher():
    """
    Polls the today's JSONL log file every 2 seconds.
    When new lines are appended, broadcasts them to all WebSocket clients.
    Lightweight enough for the Jetson 15W power budget.
    """
    print("[Watcher] Log watcher started")
    last_size = 0

    while True:
        await asyncio.sleep(2)
        log_path = today_log_path()

        if not log_path.exists():
            last_size = 0
            continue

        current_size = log_path.stat().st_size

        if current_size <= last_size:
            continue

        # Read only new bytes appended since last check
        try:
            async with aiofiles.open(log_path, "r", encoding="utf-8") as f:
                await f.seek(last_size)
                new_content = await f.read()
        except OSError:
            continue

        last_size = current_size

        # Parse each new line as a violation event
        for line in new_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                enriched = enrich_event(event)
                # Broadcast to all connected dashboard clients
                await manager.broadcast({
                    "type":  "violation",
                    "event": enriched,
                })
                print(f"[Watcher] Broadcast: zone={event.get('zone_id')} missing={event.get('missing_ppe')}")
            except json.JSONDecodeError:
                continue


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(log_watcher())
    print("[Server] PPE Dashboard running at http://0.0.0.0:8000")


# ── REST Endpoints ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML page."""
    html_path = TEMPLATE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    async with aiofiles.open(html_path, "r", encoding="utf-8") as f:
        content = await f.read()
    return HTMLResponse(content=content)


@app.get("/api/analytics")
async def get_analytics():
    """
    Returns daily KPI counters computed from today's log file.

    Response:
      {
        "total": 42,
        "by_ppe": {"helmet": 10, "vest": 20, "gloves": 12},
        "by_zone": {"zone_A": 30, "zone_B": 12},
        "hourly": {"0": 0, "1": 3, ... "23": 0},
        "latest_time": "2025-08-01T09:14:32Z",
        "date": "2025-08-01"
      }
    """
    events = parse_log_file(today_log_path())
    return JSONResponse(content=compute_analytics(events))


@app.get("/api/events")
async def get_events(limit: int = 50, offset: int = 0):
    """
    Returns paginated list of today's violation events (newest first).

    Query params:
      limit  — max events to return (default 50)
      offset — skip N events (for pagination)
    """
    events = parse_log_file(today_log_path())
    enriched = [enrich_event(e) for e in events[offset: offset + limit]]
    return JSONResponse(content={
        "total":  len(events),
        "offset": offset,
        "limit":  limit,
        "events": enriched,
    })


@app.get("/api/events/history")
async def get_history(days: int = 7):
    """
    Returns analytics for the past N days (for trend charts).
    """
    results = []
    for i in range(days):
        from datetime import timedelta
        day = date.today() - timedelta(days=i)
        log_path = LOG_DIR / f"ppe_violations_{day.isoformat()}.jsonl"
        events = parse_log_file(log_path)
        analytics = compute_analytics(events)
        analytics["date"] = day.isoformat()
        results.append(analytics)
    return JSONResponse(content=list(reversed(results)))


@app.post("/webhook/violation")
async def receive_webhook(request: Request):
    """
    Webhook receiver — the existing inference pipeline POSTs here
    on every violation. Broadcasts the event to WebSocket clients
    in addition to the log watcher (double-coverage for reliability).

    Expected payload:
      {
        "timestamp": "...",
        "zone_id": "zone_A",
        "zone_label": "Assembly Line A",
        "missing_ppe": ["helmet", "gloves"],
        "snapshot": "/abs/path/to/snapshot.jpg",
        "frame_index": 12847
      }
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Normalise snapshot path to URL
    snap = data.get("snapshot", "")
    data["snapshot_url"] = f"/snapshots/{Path(snap).name}" if snap else None

    await manager.broadcast({
        "type":  "violation",
        "event": data,
    })

    return JSONResponse(content={"status": "ok", "broadcast_to": len(manager.active)})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for the dashboard frontend.
    Clients connect here to receive real-time violation broadcasts.
    """
    await manager.connect(ws)

    # Send current analytics immediately on connect
    events = parse_log_file(today_log_path())
    analytics = compute_analytics(events)
    await ws.send_text(json.dumps({"type": "analytics", "data": analytics}))

    # Send last 20 events on connect so the sidebar populates immediately
    recent = [enrich_event(e) for e in events[:20]]
    await ws.send_text(json.dumps({"type": "history", "events": recent}))

    try:
        while True:
            # Keep the connection alive; client sends pings
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Entry Point ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,       # reload=False saves CPU on Jetson
        workers=1,          # single worker — no multiprocessing overhead
        log_level="info",
    )
