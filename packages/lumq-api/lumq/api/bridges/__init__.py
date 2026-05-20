"""Schema <-> lumq object bridge."""
from __future__ import annotations
import numpy as np

def circuit_schema_to_ir(schema):
    from lumq.compiler.ir import CircuitMetadata, GateOp, MeasOp, PhotonicCircuit
    meta = CircuitMetadata(name=schema.metadata.name, description=schema.metadata.description,
                           author=schema.metadata.author, tags=list(schema.metadata.tags))
    circ = PhotonicCircuit(n_modes=schema.n_modes, metadata=meta)
    if schema.circuit_id: circ.circuit_id = schema.circuit_id
    for op in schema.ops:
        circ.ops.append(GateOp(name=op.name, modes=tuple(op.modes), params=_dec(op.params), tag=op.tag))
    for m in schema.measurements:
        circ.measurements.append(MeasOp(kind=m.kind.value, modes=tuple(m.modes), params=dict(m.params)))
    return circ

def job_result_to_schema(result, report=None):
    from lumq.api.schemas import JobResultSchema
    return JobResultSchema(
        job_id=result.job_id, backend=result.backend, shots=result.shots,
        samples=_arr(result.samples) if result.samples is not None else None,
        state=state_to_schema(result.state) if result.state is not None else None,
        expectation_values={k:_arr(v) for k,v in result.expectation_values.items()},
        resource_report=report_to_schema(report) if report else None,
        metadata=result.metadata)

def state_to_schema(state):
    from lumq.api.schemas import GaussianStateSchema
    try:
        from lumq.photonics.states import GaussianState
        if not isinstance(state, GaussianState): return None
    except ImportError: return None
    return GaussianStateSchema(
        n_modes=state.n_modes, mu=_arr(state.mu),
        cov=[_arr(row) for row in state.cov],
        purity=float(state.purity),
        mean_photon_number=[float(state.mean_photon_number(k)) for k in range(state.n_modes)])

def report_to_schema(report):
    from lumq.api.schemas import ResourceReportSchema
    return ResourceReportSchema(gate_counts=dict(report.gate_counts), depth=report.depth,
        n_modes=report.n_modes, squeezing_budget_db=report.squeezing_budget_db,
        has_non_gaussian=report.has_non_gaussian,
        n_beamsplitters=report.n_beamsplitters, n_squeezers=report.n_squeezers)

def decode_matrix(nested):
    rows = []
    for row in nested:
        decoded = []
        for el in row:
            if isinstance(el,dict) and el.get("__complex__"): decoded.append(complex(el["re"],el["im"]))
            elif isinstance(el,(int,float)): decoded.append(float(el))
            else: decoded.append(complex(el))
        rows.append(decoded)
    return np.array(rows, dtype=complex)

def _arr(x):
    try:
        import jax.numpy as jnp; return jnp.asarray(x).tolist()
    except Exception: return list(x)

def _dec(params):
    out = {}
    for k,v in params.items():
        if isinstance(v,dict) and v.get("__complex__"): out[k]=complex(v["re"],v["im"])
        else: out[k]=v
    return out

