"""
lumq.api.ws.handler
~~~~~~~~~~~~~~~~~~~~
WebSocket endpoint for real-time job status updates.

Protocol
--------
Client connects to ws://host/ws

Server push events (JSON):
  {"event": "job_queued",    "job_id": "...", "data": {...}}
  {"event": "job_running",   "job_id": "...", "data": {}}
  {"event": "job_completed", "job_id": "...", "data": {result}}
  {"event": "job_failed",    "job_id": "...", "data": {"error": "..."}}

Client messages (JSON):
  {"event": "ping"}            → server responds {"event": "pong"}
  {"event": "subscribe",       "job_id": "..."}  → subscribe to a specific job
  {"event": "submit", "data":  {JobSubmitRequest}} → submit + stream result

Connection lifecycle
--------------------
- Connections are tracked in a module-level set.
- On job completion, the executing task pushes the result to all subscribed clients.
- Connections auto-clean on disconnect.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])

# Active WebSocket connections: {connection_id: WebSocket}
_connections: dict[str, WebSocket] = {}
# Subscriptions: {job_id: set of connection_ids}
_subscriptions: dict[str, set[str]] = {}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time job events."""
    await websocket.accept()

    conn_id = id(websocket).__str__()
    _connections[conn_id] = websocket

    try:
        await _handle_connection(conn_id, websocket)
    except WebSocketDisconnect:
        pass
    finally:
        _connections.pop(conn_id, None)
        # Clean subscriptions
        for subs in _subscriptions.values():
            subs.discard(conn_id)


async def _handle_connection(conn_id: str, ws: WebSocket) -> None:
    """Main message loop for a single WebSocket connection."""
    # Send welcome
    await _send(ws, {"event": "connected", "data": {"conn_id": conn_id}})

    while True:
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
        except asyncio.TimeoutError:
            # Send keepalive
            await _send(ws, {"event": "ping"})
            continue

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await _send(ws, {"event": "error", "data": {"detail": "Invalid JSON."}})
            continue

        event = msg.get("event", "")

        if event == "ping":
            await _send(ws, {"event": "pong", "timestamp": _now()})

        elif event == "subscribe":
            job_id = msg.get("job_id")
            if job_id:
                _subscriptions.setdefault(job_id, set()).add(conn_id)
                await _send(ws, {"event": "subscribed", "job_id": job_id})
            else:
                await _send(ws, {"event": "error", "data": {"detail": "subscribe requires job_id"}})

        elif event == "submit":
            # Submit a job and stream the result back to this connection
            asyncio.create_task(
                _submit_and_stream(conn_id, ws, msg.get("data", {}))
            )

        else:
            await _send(ws, {
                "event": "error",
                "data": {"detail": f"Unknown event '{event}'"},
            })


# ---------------------------------------------------------------------------
# Submit and stream
# ---------------------------------------------------------------------------


async def _submit_and_stream(conn_id: str, ws: WebSocket, data: dict) -> None:
    """Submit a job via the HTTP layer and stream its result to the WS client."""
    import httpx

    try:
        from lumq.api.schemas import JobSubmitRequest
        request = JobSubmitRequest.model_validate(data)
    except Exception as exc:
        await _send(ws, {"event": "error", "data": {"detail": str(exc)}})
        return

    # Inline execution (for direct WS submissions bypass the job queue)
    await _send(ws, {"event": "job_running", "timestamp": _now()})

    try:
        job_result_schema, report_schema = await asyncio.to_thread(
            _execute_inline, request
        )
        await _send(ws, {
            "event":     "job_completed",
            "timestamp": _now(),
            "data":      job_result_schema.model_dump(),
        })
    except Exception as exc:
        await _send(ws, {
            "event":     "job_failed",
            "timestamp": _now(),
            "data":      {"error": str(exc)},
        })


def _execute_inline(request):
    """Synchronous execution — runs in a thread pool."""
    from lumq.api.bridge import circuit_schema_to_ir, job_result_to_schema
    from lumq.backends.simulators.fock import FockSimulator, FockSimulatorConfig
    from lumq.backends.simulators.gaussian import GaussianSimulator, GaussianSimulatorConfig
    from lumq.compiler import Compiler, PassPipeline

    circuit = circuit_schema_to_ir(request.circuit)
    cc = request.compiler
    pipeline = (
        PassPipeline.for_hardware() if cc.for_hardware else PassPipeline.default()
    ) if cc.run_passes else PassPipeline([])

    compiled, report = Compiler(
        pipeline=pipeline,
        decompose_interferometers=cc.decompose_interferometers,
    ).compile(circuit)

    sc = request.simulator
    if sc.backend.value == "fock":
        sim = FockSimulator(FockSimulatorConfig(
            n_modes=compiled.n_modes, cutoff=sc.cutoff, seed=sc.seed
        ))
    else:
        sim = GaussianSimulator(GaussianSimulatorConfig(
            n_modes=compiled.n_modes, seed=sc.seed,
            track_state=sc.track_state, compute_ev=sc.compute_ev,
        ))

    result = sim.run(compiled, shots=sc.shots)
    return job_result_to_schema(result, report), report


# ---------------------------------------------------------------------------
# Push to subscribers
# ---------------------------------------------------------------------------


async def push_to_subscribers(job_id: str, event: dict) -> None:
    """Push an event to all WebSocket clients subscribed to a job."""
    conn_ids = _subscriptions.get(job_id, set()).copy()
    for conn_id in conn_ids:
        ws = _connections.get(conn_id)
        if ws:
            try:
                await _send(ws, event)
            except Exception:
                _connections.pop(conn_id, None)
                _subscriptions.get(job_id, set()).discard(conn_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _send(ws: WebSocket, data: dict) -> None:
    try:
        await ws.send_text(json.dumps(data))
    except Exception:
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
