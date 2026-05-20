"""
lumq.api.routes.compiler
~~~~~~~~~~~~~~~~~~~~~~~~~
Compiler-facing endpoints.

POST /compiler/clements  — decompose a unitary into a Clements BS mesh
POST /compiler/compile   — compile a circuit and return optimised IR + report
POST /compiler/resource  — estimate resources without running
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from lumq.api.schemas import (
    ClementsRequest,
    ClementsResponse,
    CompilerConfigSchema,
    PhotonicCircuitSchema,
    ResourceReportSchema,
    SimulatorConfigSchema,
)

router = APIRouter(prefix="/compiler", tags=["compiler"])


# ---------------------------------------------------------------------------
# POST /compiler/clements
# ---------------------------------------------------------------------------


@router.post("/clements", response_model=ClementsResponse)
def decompose_clements(request: ClementsRequest) -> ClementsResponse:
    """Decompose an N×N unitary matrix via the Clements rectangular mesh algorithm.

    Accepts a complex matrix encoded as nested lists.  Complex entries may
    be plain floats (real) or objects {"__complex__": true, "re": x, "im": y}.

    Returns the beamsplitter mesh, output phases, and a ready-to-simulate
    PhotonicCircuit.
    """
    from lumq.api.bridge import clements_result_to_schema
    from lumq.compiler.decompose.clements import clements_decompose

    # Decode the matrix
    try:
        U = _decode_matrix(request.U)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Matrix decode error: {exc}")

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=f"U must be a square matrix. Got shape {U.shape}.",
        )

    try:
        result = clements_decompose(U, tol=request.tol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return clements_result_to_schema(result)


# ---------------------------------------------------------------------------
# POST /compiler/compile
# ---------------------------------------------------------------------------


@router.post("/compile", response_model=dict)
def compile_circuit(
    circuit: PhotonicCircuitSchema,
    config: CompilerConfigSchema = CompilerConfigSchema(),
) -> dict:
    """Compile a circuit and return the optimised IR and resource report.

    Does not execute — use POST /jobs for execution.
    """
    from lumq.api.bridge import circuit_schema_to_ir, ir_to_circuit_schema, resource_report_to_schema
    from lumq.compiler import Compiler, PassPipeline

    ir = circuit_schema_to_ir(circuit)

    pipeline = (
        PassPipeline.for_hardware() if config.for_hardware else PassPipeline.default()
    ) if config.run_passes else PassPipeline([])

    try:
        compiled, report = Compiler(
            pipeline=pipeline,
            decompose_interferometers=config.decompose_interferometers,
        ).compile(ir)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "compiled_circuit": ir_to_circuit_schema(compiled).model_dump(),
        "resource_report":  resource_report_to_schema(report).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /compiler/resource
# ---------------------------------------------------------------------------


@router.post("/resource", response_model=ResourceReportSchema)
def estimate_resources(circuit: PhotonicCircuitSchema) -> ResourceReportSchema:
    """Estimate gate counts, depth, and squeezing budget without compiling.

    Useful for the UI's circuit complexity display.
    """
    from lumq.api.bridge import circuit_schema_to_ir, resource_report_to_schema
    from lumq.compiler.passes import ResourceEstimator

    ir = circuit_schema_to_ir(circuit)
    estimator = ResourceEstimator()
    _, report  = estimator(ir)
    return resource_report_to_schema(report)


# ---------------------------------------------------------------------------
# Matrix decode helper
# ---------------------------------------------------------------------------


def _decode_matrix(nested: list) -> np.ndarray:
    """Decode a nested list (with optional complex dicts) to numpy array."""
    rows = []
    for row in nested:
        decoded = []
        for el in row:
            if isinstance(el, dict) and el.get("__complex__"):
                decoded.append(complex(el["re"], el["im"]))
            elif isinstance(el, (int, float)):
                decoded.append(float(el))
            elif isinstance(el, list) and len(el) == 2:
                # [re, im] tuple shorthand
                decoded.append(complex(el[0], el[1]))
            else:
                decoded.append(complex(el))
        rows.append(decoded)
    return np.array(rows, dtype=complex)
