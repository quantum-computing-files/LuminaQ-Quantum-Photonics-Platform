"""Pydantic v2 API schemas."""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

class BackendName(str, Enum):
    gaussian = "gaussian"
    fock = "fock"

class MeasurementKind(str, Enum):
    homodyne = "homodyne"
    heterodyne = "heterodyne"
    pnr = "pnr"
    fock_proj = "fock"

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"

class GateOpSchema(BaseModel):
    name: str
    modes: list[int]
    params: dict[str, Any] = Field(default_factory=dict)
    tag: Optional[str] = None

class MeasOpSchema(BaseModel):
    kind: MeasurementKind
    modes: list[int]
    params: dict[str, Any] = Field(default_factory=dict)

class CircuitMetadataSchema(BaseModel):
    name: str = "unnamed"; description: str = ""; author: str = ""; tags: list[str] = Field(default_factory=list)

class PhotonicCircuitSchema(BaseModel):
    n_modes: int = Field(..., ge=1, le=64)
    ops: list[GateOpSchema] = Field(default_factory=list)
    measurements: list[MeasOpSchema] = Field(default_factory=list)
    metadata: CircuitMetadataSchema = Field(default_factory=CircuitMetadataSchema)
    circuit_id: Optional[str] = None

class SimulatorConfigSchema(BaseModel):
    backend: BackendName = BackendName.gaussian
    shots: int = Field(default=1, ge=1, le=100_000)
    cutoff: int = Field(default=10, ge=2, le=50)
    seed: int = 0; track_state: bool = True
    compute_ev: list[str] = Field(default=["mean_photon"])

class CompilerConfigSchema(BaseModel):
    run_passes: bool = True; decompose_interferometers: bool = True; for_hardware: bool = False

class JobSubmitRequest(BaseModel):
    circuit: PhotonicCircuitSchema
    simulator: SimulatorConfigSchema = Field(default_factory=SimulatorConfigSchema)
    compiler: CompilerConfigSchema = Field(default_factory=CompilerConfigSchema)

class GaussianStateSchema(BaseModel):
    n_modes: int; mu: list[float]; cov: list[list[float]]
    purity: float; mean_photon_number: list[float]

class ResourceReportSchema(BaseModel):
    gate_counts: dict[str,int]; depth: int; n_modes: int
    squeezing_budget_db: float; has_non_gaussian: bool
    n_beamsplitters: int; n_squeezers: int

class JobResultSchema(BaseModel):
    job_id: str; backend: str; shots: int
    samples: Optional[list[list[float]]] = None
    state: Optional[GaussianStateSchema] = None
    expectation_values: dict[str,list[float]] = Field(default_factory=dict)
    resource_report: Optional[ResourceReportSchema] = None
    metadata: dict[str,Any] = Field(default_factory=dict)

class JobResponse(BaseModel):
    job_id: str; status: JobStatus; created_at: str
    result: Optional[JobResultSchema] = None
    error: Optional[str] = None
    circuit_id: Optional[str] = None

class WignerRequest(BaseModel):
    circuit: PhotonicCircuitSchema
    simulator: SimulatorConfigSchema = Field(default_factory=SimulatorConfigSchema)
    compiler: CompilerConfigSchema = Field(default_factory=CompilerConfigSchema)
    mode: int = Field(default=0, ge=0); n_points: int = Field(default=80, ge=20, le=200)
    x_range: float = Field(default=5.0, ge=1.0, le=20.0)

class MarginalRequest(BaseModel):
    circuit: PhotonicCircuitSchema
    simulator: SimulatorConfigSchema = Field(default_factory=SimulatorConfigSchema)
    compiler: CompilerConfigSchema = Field(default_factory=CompilerConfigSchema)
    mode: int = Field(default=0, ge=0); n_points: int = Field(default=150, ge=50, le=500)

class ClementsRequest(BaseModel):
    U: list[list[Any]]; tol: float = 1e-12

class ClementsBeamsplitter(BaseModel):
    mode0: int; mode1: int; theta: float; phi: float

class ClementsResponse(BaseModel):
    n_modes: int; n_beamsplitters: int; reconstruction_error: float
    beamsplitters: list[ClementsBeamsplitter]
    phases: list[dict[str,float]]
    circuit: PhotonicCircuitSchema
