"""
lumq.api.routes.phase_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase-space visualisation endpoints.

POST /phase-space/wigner   — compute Wigner function on a grid
POST /phase-space/marginal — compute x and p marginal distributions
POST /phase-space/husimi   — compute Husimi Q function (future)

These are synchronous because the computation is fast (<200 ms for typical
Gaussian circuits).  Results go directly to the UI for rendering.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lumq.api.schemas import (
    MarginalRequest,
    MarginalResponse,
    WignerRequest,
    WignerResponse,
)

router = APIRouter(prefix="/phase-space", tags=["phase-space"])


# ---------------------------------------------------------------------------
# POST /phase-space/wigner
# ---------------------------------------------------------------------------


@router.post("/wigner", response_model=WignerResponse)
def compute_wigner(request: WignerRequest) -> WignerResponse:
    """Compute the Wigner quasi-probability distribution W(x, p).

    Runs the circuit on the Gaussian simulator (non-Gaussian circuits
    fall back to FockSimulator automatically), then computes W on an
    n_points × n_points grid over [-x_range, +x_range]².

    Returns a flat grid ready for the UI's heatmap component.
    """
    try:
        state = _run_to_state(request.circuit, request.simulator, request.compiler)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    from lumq.photonics.phase_space import wigner
    import jax.numpy as jnp

    mode = request.mode
    if mode >= state.n_modes:
        raise HTTPException(
            status_code=422,
            detail=f"mode={mode} exceeds circuit n_modes={state.n_modes}.",
        )

    X, P, W = wigner(
        state,
        mode=mode,
        n_points=request.n_points,
        x_range=request.x_range,
    )

    x_values = X[:, 0].tolist()
    p_values = P[0, :].tolist()
    W_list   = W.tolist()
    W_min    = float(jnp.min(W))
    W_max    = float(jnp.max(W))

    return WignerResponse(
        mode=mode,
        n_points=request.n_points,
        x_range=request.x_range,
        x_values=x_values,
        p_values=p_values,
        W=W_list,
        W_min=W_min,
        W_max=W_max,
        is_non_negative=bool(W_min >= -1e-10),
    )


# ---------------------------------------------------------------------------
# POST /phase-space/marginal
# ---------------------------------------------------------------------------


@router.post("/marginal", response_model=MarginalResponse)
def compute_marginal(request: MarginalRequest) -> MarginalResponse:
    """Compute x and p quadrature marginal distributions.

    Returns P(x) = ∫W dp and P(p) = ∫W dx as 1D probability densities.
    """
    try:
        state = _run_to_state(request.circuit, request.simulator, request.compiler)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    from lumq.photonics.phase_space import marginal_x, marginal_p

    mode = request.mode
    if mode >= state.n_modes:
        raise HTTPException(
            status_code=422,
            detail=f"mode={mode} exceeds circuit n_modes={state.n_modes}.",
        )

    xvec, prob_x = marginal_x(state, mode=mode, n_points=request.n_points)
    pvec, prob_p = marginal_p(state, mode=mode, n_points=request.n_points)

    return MarginalResponse(
        mode=mode,
        xvec=xvec.tolist(),
        pvec=pvec.tolist(),
        prob_x=prob_x.tolist(),
        prob_p=prob_p.tolist(),
    )


# ---------------------------------------------------------------------------
# Shared helper: compile + simulate → GaussianState
# ---------------------------------------------------------------------------


def _run_to_state(circuit_schema, simulator_schema, compiler_schema):
    """Compile and simulate a circuit, returning the final GaussianState."""
    from lumq.api.bridge import circuit_schema_to_ir
    from lumq.backends.simulators.gaussian import GaussianSimulator, GaussianSimulatorConfig
    from lumq.compiler import Compiler, PassPipeline

    circuit = circuit_schema_to_ir(circuit_schema)

    cc = compiler_schema
    pipeline = (
        PassPipeline.for_hardware() if cc.for_hardware else PassPipeline.default()
    ) if cc.run_passes else PassPipeline([])

    compiled, _ = Compiler(
        pipeline=pipeline,
        decompose_interferometers=cc.decompose_interferometers,
    ).compile(circuit)

    sc = simulator_schema
    result = GaussianSimulator(
        GaussianSimulatorConfig(
            n_modes=compiled.n_modes,
            seed=sc.seed,
            track_state=True,
        )
    ).run(compiled, shots=1)

    if result.state is None:
        raise RuntimeError("Simulator did not return a state.")
    return result.state
