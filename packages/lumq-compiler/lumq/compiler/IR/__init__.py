"""PhotonicCircuit IR."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from uuid import uuid4

__all__ = ["PhotonicCircuit","GateOp","MeasOp","CircuitMetadata"]

@dataclass
class GateOp:
    name: str
    modes: tuple
    params: dict = field(default_factory=dict)
    tag: Optional[str] = None
    def __post_init__(self): self.modes = tuple(self.modes)
    def to_dict(self):
        return {"name":self.name,"modes":list(self.modes),"params":_enc(self.params),"tag":self.tag}
    @classmethod
    def from_dict(cls, d):
        return cls(name=d["name"],modes=tuple(d["modes"]),params=_dec(d.get("params",{})),tag=d.get("tag"))
    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k,v in self.params.items())
        return f"{self.name}(modes={self.modes}, {p})"

@dataclass
class MeasOp:
    kind: str
    modes: tuple
    params: dict = field(default_factory=dict)
    def __post_init__(self): self.modes = tuple(self.modes)
    def to_dict(self): return {"kind":self.kind,"modes":list(self.modes),"params":self.params}
    @classmethod
    def from_dict(cls, d): return cls(kind=d["kind"],modes=tuple(d["modes"]),params=d.get("params",{}))

@dataclass
class CircuitMetadata:
    name: str = "unnamed"
    description: str = ""
    author: str = ""
    created_by: str = "lumq"
    tags: list = field(default_factory=list)

class PhotonicCircuit:
    def __init__(self, n_modes, metadata=None):
        if n_modes < 1: raise ValueError("n_modes must be >= 1.")
        self.n_modes = n_modes
        self.metadata = metadata or CircuitMetadata()
        self.ops: list = []
        self.measurements: list = []
        self.circuit_id = str(uuid4())[:8]

    def add(self, op):
        self._chk(op.modes); self.ops.append(op); return self
    def ps(self, mode, phi):         return self.add(GateOp("PhaseShifter",(mode,),{"phi":phi}))
    def sq(self, mode, r, phi=0.0): return self.add(GateOp("Squeezer",(mode,),{"r":r,"phi":phi}))
    def d(self, mode, alpha):        return self.add(GateOp("Displacer",(mode,),{"alpha":alpha}))
    def bs(self, m0, m1, theta, phi=0.0): return self.add(GateOp("Beamsplitter",(m0,m1),{"theta":theta,"phi":phi}))
    def tms(self, m0, m1, r, phi=0.0):   return self.add(GateOp("TwoModeSqueeze",(m0,m1),{"r":r,"phi":phi}))
    def kerr(self, mode, kappa):     return self.add(GateOp("KerrGate",(mode,),{"kappa":kappa}))
    def interferometer(self, U):
        import numpy as np
        return self.add(GateOp("Interferometer",tuple(range(self.n_modes)),{"U":np.array(U).tolist()}))

    def meas_homodyne(self, mode, phi=0.0):
        self._chk((mode,)); self.measurements.append(MeasOp("homodyne",(mode,),{"phi":phi})); return self
    def meas_heterodyne(self, mode):
        self._chk((mode,)); self.measurements.append(MeasOp("heterodyne",(mode,),{})); return self
    def meas_pnr(self, mode, max_photons=10):
        self._chk((mode,)); self.measurements.append(MeasOp("pnr",(mode,),{"max_photons":max_photons})); return self

    @property
    def gate_count(self): return len(self.ops)
    @property
    def depth(self):
        last = [0]*self.n_modes; d = 0
        for op in self.ops:
            t = max(last[m] for m in op.modes)+1
            for m in op.modes: last[m]=t
            d = max(d,t)
        return d
    @property
    def has_non_gaussian(self):
        return any(op.name in {"KerrGate","CubicPhase"} for op in self.ops)
    def gate_counts_by_type(self):
        c = {}
        for op in self.ops: c[op.name]=c.get(op.name,0)+1
        return c

    def to_dict(self):
        return {"circuit_id":self.circuit_id,"n_modes":self.n_modes,
                "metadata":asdict(self.metadata),
                "ops":[op.to_dict() for op in self.ops],
                "measurements":[m.to_dict() for m in self.measurements]}
    def to_json(self, indent=2): return json.dumps(self.to_dict(),indent=indent)
    @classmethod
    def from_dict(cls, d):
        c = cls(n_modes=d["n_modes"],metadata=CircuitMetadata(**d.get("metadata",{})))
        c.circuit_id = d.get("circuit_id",c.circuit_id)
        for op in d.get("ops",[]): c.ops.append(GateOp.from_dict(op))
        for m in d.get("measurements",[]): c.measurements.append(MeasOp.from_dict(m))
        return c
    @classmethod
    def from_json(cls, s): return cls.from_dict(json.loads(s))

    def draw(self):
        lines = [f"  q{k} :" for k in range(self.n_modes)]
        for op in self.ops:
            lbl = f"[{op.name[:4]}]"; w = len(lbl)+2
            for m in op.modes: lines[m] += f"-{lbl}-"
            for m in set(range(self.n_modes))-set(op.modes): lines[m] += "-"*w
        for m in self.measurements: lines[m.modes[0]] += f"-[M:{m.kind[:3]}]"
        return f"PhotonicCircuit '{self.metadata.name}' ({self.n_modes} modes, depth={self.depth})\n"+chr(10).join(lines)

    def _chk(self, modes):
        for m in modes:
            if not (0<=m<self.n_modes):
                raise ValueError(f"Mode {m} out of range for {self.n_modes}-mode circuit.")
    def __repr__(self): return f"PhotonicCircuit(n_modes={self.n_modes}, gates={self.gate_count}, depth={self.depth})"

def _enc(params):
    out = {}
    for k,v in params.items():
        if isinstance(v, complex): out[k]={"__complex__":True,"re":v.real,"im":v.imag}
        else: out[k]=v
    return out

def _dec(params):
    out = {}
    for k,v in params.items():
        if isinstance(v,dict) and v.get("__complex__"): out[k]=complex(v["re"],v["im"])
        else: out[k]=v
    return out

