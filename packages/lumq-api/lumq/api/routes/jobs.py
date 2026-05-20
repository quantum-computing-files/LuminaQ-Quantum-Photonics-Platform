"""
lumq.api.routes.jobs
~~~~~~~~~~~~~~~~~~~~~
Circuit execution job endpoints.

POST   /jobs          — submit circuit, run immediately or queue
GET    /jobs/{job_id} — poll job status and retrieve result
GET    /jobs          — list recent jobs (paginated)
DELETE /jobs/{job_id} — cancel a queued job
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from lumq.api.schemas import (
    JobResponse,
    JobStatus,
    JobSubmitRequest,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# In-process job store (replace with Redis for production)
# ---------------------------------------------------------------------------

_JOB_STORE: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# POST /jobs — submit
# ---------------------------------------------------------------------------


@router.post("", response_model=JobResponse, status_code=202)
async def submit_job(
    request: JobSubmitRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Submit a photonic circuit for compilation and simulation.

    The circuit is compiled and executed asynchronously.
    Poll GET /jobs/{job_id} or subscribe to WebSocket /ws to receive the result.
    """
    job_id = str(uuid4())[:8]

    _JOB_STORE[job_id] = {
        "job_id":     job_id,
        "status":     JobStatus.queued,
        "created_at": _now(),
        "request":    request,
        "result":     None,
        "error":      None,
        "circuit_id": request.circuit.circuit_id,
    }

    # Run in background so the 202 response is immediate
    background_tasks.add_task(_execute_job, job_id, request)

    return JobResponse(
        job_id=job_id,
        status=JobStatus.queued,
        created_at=_JOB_STORE[job_id]["created_at"],
        circuit_id=request.circuit.circuit_id,
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} — poll
# ---------------------------------------------------------------------------


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Retrieve the current status and result of a job."""
    job = _JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        result=job["result"],
        error=job["error"],
        circuit_id=job.get("circuit_id"),
    )


# ---------------------------------------------------------------------------
# GET /jobs — list recent
# ---------------------------------------------------------------------------


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[JobStatus] = Query(default=None),
) -> list[JobResponse]:
    """List recent jobs, optionally filtered by status."""
    jobs = list(_JOB_STORE.values())
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    jobs = sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:limit]
    return [
        JobResponse(
            job_id=j["job_id"],
            status=j["status"],
            created_at=j["created_at"],
            result=j["result"],
            error=j["error"],
        )
        for j in jobs
    ]


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id} — cancel
# ---------------------------------------------------------------------------


@router.delete("/{job_id}", status_code=204)
async def cancel_job(job_id: str) -> None:
    """Cancel a queued job.  Has no effect on already-running or completed jobs."""
    job = _JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] == JobStatus.queued:
        job["status"] = JobStatus.failed
        job["error"]  = "Cancelled by client."


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------


async def _execute_job(job_id: str, request: JobSubmitRequest) -> None:
    """Compile and simulate the circuit, then update the job store."""
    job = _JOB_STORE[job_id]
    job["status"] = JobStatus.running

    try:
        result_schema, report_schema = await asyncio.to_thread(
            _run_synchronous, request
        )
        job["result"] = result_schema
        job["status"] = JobStatus.completed
    except Exception as exc:
        job["status"] = JobStatus.failed
        job["error"]  = str(exc)


def _run_synchronous(request: JobSubmitRequest):
    """CPU-bound: compile + simulate.  Runs in a thread pool."""
    from lumq.api.bridge import (
        circuit_schema_to_ir,
        job_result_to_schema,
        resource_report_to_schema,
    )
    from lumq.backends.simulators.fock import FockSimulator, FockSimulatorConfig
    from lumq.backends.simulators.gaussian import (
        GaussianSimulator,
        GaussianSimulatorConfig,
    )
    from lumq.compiler import Compiler, PassPipeline

    # 1. Convert schema → IR
    circuit = circuit_schema_to_ir(request.circuit)

    # 2. Compile
    cc = request.compiler
    pipeline = (
        PassPipeline.for_hardware() if cc.for_hardware else PassPipeline.default()
    ) if cc.run_passes else PassPipeline([])

    compiler = Compiler(
        pipeline=pipeline,
        decompose_interferometers=cc.decompose_interferometers,
    )
    compiled, report = compiler.compile(circuit)

    # 3. Select and configure backend
    sc = request.simulator
    if sc.backend.value == "fock":
        sim = FockSimulator(
            FockSimulatorConfig(
                n_modes=compiled.n_modes,
                cutoff=sc.cutoff,
                seed=sc.seed,
            )
        )
    else:
        sim = GaussianSimulator(
            GaussianSimulatorConfig(
                n_modes=compiled.n_modes,
                seed=sc.seed,
                track_state=sc.track_state,
                compute_ev=sc.compute_ev,
            )
        )

    # 4. Execute
    job_result = sim.run(compiled, shots=sc.shots)

    # 5. Serialise
    return job_result_to_schema(job_result, report), report

