"""lumq.api.app - FastAPI application."""
from __future__ import annotations
import asyncio, json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

API_VERSION = "0.1.0"
_JOBS: dict = {}
_WS: dict = {}

def _now(): return datetime.now(timezone.utc).isoformat()

@asynccontextmanager
async def lifespan(app):
    import lumq.backends, lumq.compiler, lumq.photonics  # noqa
    yield

def create_app():
    app = FastAPI(title="LuminaQ API", version=API_VERSION,
                  description="Photonic Quantum Computing Platform API",
                  lifespan=lifespan)
    app.add_middleware(CORSMiddleware,
        allow_origins=["http://localhost:5173","http://localhost:3000","*"],
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return JSONResponse({"status":"ok","version":API_VERSION,"backends":["gaussian","fock"],"max_modes":64})

    @app.get("/")
    def root():
        return JSONResponse({"service":"LuminaQ API","version":API_VERSION,"docs":"/docs"})

    @app.post("/jobs/submit", status_code=202)
    async def submit_job(request: dict, background_tasks: BackgroundTasks):
        from lumq.api.schemas import JobSubmitRequest, JobStatus
        req = JobSubmitRequest.model_validate(request)
        job_id = str(uuid4())[:8]
        _JOBS[job_id] = {"job_id":job_id,"status":JobStatus.queued,
                         "created_at":_now(),"result":None,"error":None}
        background_tasks.add_task(_exec_job, job_id, req)
        return {"job_id":job_id,"status":"queued","created_at":_JOBS[job_id]["created_at"]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        j = _JOBS.get(job_id)
        if not j: raise HTTPException(404, f"Job '{job_id}' not found.")
        return j

    @app.get("/jobs")
    def list_jobs(limit: int = Query(20, ge=1, le=100)):
        return sorted(_JOBS.values(), key=lambda j: j["created_at"], reverse=True)[:limit]

    @app.post("/phase-space/wigner")
    def compute_wigner(request: dict):
        from lumq.api.schemas import WignerRequest
        from lumq.photonics.phase_space import wigner
        import jax.numpy as jnp
        req = WignerRequest.model_validate(request)
        state = _run_to_state(req.circuit, req.simulator, req.compiler)
        if req.mode >= state.n_modes: raise HTTPException(422, f"mode={req.mode} >= n_modes={state.n_modes}")
        X,P,W = wigner(state, mode=req.mode, n_points=req.n_points, x_range=req.x_range)
        return {"mode":req.mode,"n_points":req.n_points,"x_range":req.x_range,
                "x_values":X[:,0].tolist(),"p_values":P[0,:].tolist(),
                "W":W.tolist(),"W_min":float(jnp.min(W)),"W_max":float(jnp.max(W)),
                "is_non_negative":bool(float(jnp.min(W))>=-1e-10)}

    @app.post("/phase-space/marginal")
    def compute_marginal(request: dict):
        from lumq.api.schemas import MarginalRequest
        from lumq.photonics.phase_space import marginal_x, marginal_p
        req = MarginalRequest.model_validate(request)
        state = _run_to_state(req.circuit, req.simulator, req.compiler)
        xvec,px = marginal_x(state, mode=req.mode, n_points=req.n_points)
        pvec,pp = marginal_p(state, mode=req.mode, n_points=req.n_points)
        return {"mode":req.mode,"xvec":xvec.tolist(),"pvec":pvec.tolist(),
                "prob_x":px.tolist(),"prob_p":pp.tolist()}

    @app.post("/compiler/clements")
    def clements_endpoint(request: dict):
        from lumq.api.schemas import ClementsRequest
        from lumq.api.bridge import decode_matrix
        from lumq.compiler.decompose.clements import clements_decompose
        req = ClementsRequest.model_validate(request)
        try: U = decode_matrix(req.U)
        except Exception as e: raise HTTPException(422, f"Matrix decode: {e}")
        try: result = clements_decompose(U, tol=req.tol)
        except ValueError as e: raise HTTPException(422, str(e))
        from lumq.compiler.ir import PhotonicCircuit as PC
        circ = result.circuit
        return {"n_modes":result.n_modes,"n_beamsplitters":result.n_beamsplitters,
                "reconstruction_error":result.reconstruction_error,
                "beamsplitters":[{"mode0":m0,"mode1":m1,"theta":t,"phi":p} for m0,m1,t,p in result.beamsplitters],
                "phases":[{"mode":m,"phi":float(phi)} for m,phi in result.phases]}

    @app.post("/compiler/resource")
    def resource_estimate(circuit: dict):
        from lumq.api.schemas import PhotonicCircuitSchema
        from lumq.api.bridge import circuit_schema_to_ir, report_to_schema
        from lumq.compiler import ResourceEstimator
        ir = circuit_schema_to_ir(PhotonicCircuitSchema.model_validate(circuit))
        _, report = ResourceEstimator()(ir)
        return report_to_schema(report).model_dump()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        conn_id = str(id(websocket))
        _WS[conn_id] = websocket
        try:
            await websocket.send_text(json.dumps({"event":"connected","conn_id":conn_id}))
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"event":"ping"})); continue
                try: msg = json.loads(raw)
                except Exception:
                    await websocket.send_text(json.dumps({"event":"error","detail":"Invalid JSON"})); continue
                ev = msg.get("event","")
                if ev=="ping":
                    await websocket.send_text(json.dumps({"event":"pong","timestamp":_now()}))
                elif ev=="submit":
                    asyncio.create_task(_ws_submit(conn_id, websocket, msg.get("data",{})))
                else:
                    await websocket.send_text(json.dumps({"event":"error","detail":f"Unknown event '{ev}'"}))
        except WebSocketDisconnect:
            pass
        finally:
            _WS.pop(conn_id, None)

    return app

def _run_to_state(circuit_schema, sim_schema, compiler_schema):
    from lumq.api.bridge import circuit_schema_to_ir
    from lumq.backends import GaussianSimulator, GaussianSimulatorConfig
    from lumq.compiler import Compiler, PassPipeline
    ir = circuit_schema_to_ir(circuit_schema)
    cc = compiler_schema
    pipeline = (PassPipeline.for_hardware() if cc.for_hardware else PassPipeline.default()) if cc.run_passes else PassPipeline([])
    compiled, _ = Compiler(pipeline=pipeline, decompose_interferometers=cc.decompose_interferometers).compile(ir)
    result = GaussianSimulator(GaussianSimulatorConfig(n_modes=compiled.n_modes, seed=sim_schema.seed, track_state=True)).run(compiled, shots=1)
    if result.state is None: raise RuntimeError("Simulator returned no state.")
    return result.state

async def _exec_job(job_id, request):
    from lumq.api.schemas import JobStatus
    _JOBS[job_id]["status"] = JobStatus.running
    try:
        result = await asyncio.to_thread(_run_sync, request)
        _JOBS[job_id]["result"] = result
        _JOBS[job_id]["status"] = JobStatus.completed
    except Exception as e:
        _JOBS[job_id]["status"] = JobStatus.failed
        _JOBS[job_id]["error"] = str(e)

def _run_sync(request):
    from lumq.api.bridge import circuit_schema_to_ir, job_result_to_schema
    from lumq.backends import GaussianSimulator, GaussianSimulatorConfig, FockSimulator, FockSimulatorConfig
    from lumq.compiler import Compiler, PassPipeline
    ir = circuit_schema_to_ir(request.circuit)
    cc = request.compiler
    pipeline = (PassPipeline.for_hardware() if cc.for_hardware else PassPipeline.default()) if cc.run_passes else PassPipeline([])
    compiled, report = Compiler(pipeline=pipeline, decompose_interferometers=cc.decompose_interferometers).compile(ir)
    sc = request.simulator
    if sc.backend.value=="fock":
        sim = FockSimulator(FockSimulatorConfig(n_modes=compiled.n_modes, cutoff=sc.cutoff, seed=sc.seed))
    else:
        sim = GaussianSimulator(GaussianSimulatorConfig(n_modes=compiled.n_modes, seed=sc.seed,
                                                         track_state=sc.track_state, compute_ev=sc.compute_ev))
    result = sim.run(compiled, shots=sc.shots)
    return job_result_to_schema(result, report).model_dump()

async def _ws_submit(conn_id, ws, data):
    from lumq.api.schemas import JobSubmitRequest
    try: req = JobSubmitRequest.model_validate(data)
    except Exception as e:
        await ws.send_text(json.dumps({"event":"error","detail":str(e)})); return
    await ws.send_text(json.dumps({"event":"job_running","timestamp":_now()}))
    try:
        result = await asyncio.to_thread(_run_sync, req)
        await ws.send_text(json.dumps({"event":"job_completed","timestamp":_now(),"data":result}))
    except Exception as e:
        await ws.send_text(json.dumps({"event":"job_failed","timestamp":_now(),"data":{"error":str(e)}}))

app = create_app()

