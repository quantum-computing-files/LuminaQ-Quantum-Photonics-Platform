"""lumq.compiler - circuit compiler v0.1.0"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from lumq.compiler.ir import GateOp, PhotonicCircuit, CircuitMetadata
from lumq.compiler.decompose.clements import clements_decompose, random_unitary, ClementsResult

__all__ = ["PassPipeline","RemoveIdentityGates","MergePhaseShifters","DecomposeTwoMode",
           "ResourceEstimator","ResourceReport","Compiler",
           "PhotonicCircuit","GateOp","CircuitMetadata","clements_decompose","random_unitary","ClementsResult"]

_EPS = 1e-9

class RemoveIdentityGates:
    def __call__(self, circuit):
        new_ops = [op for op in circuit.ops if not self._is_id(op)]
        return _clone(circuit, new_ops)
    def _is_id(self, op):
        p = op.params
        if op.name=="Squeezer"     and abs(p.get("r",1.))<_EPS: return True
        if op.name=="PhaseShifter" and abs(p.get("phi",1.))<_EPS: return True
        if op.name=="Beamsplitter" and abs(p.get("theta",1.))<_EPS: return True
        if op.name=="Displacer"    and abs(p.get("alpha",1.))<_EPS: return True
        return False

class MergePhaseShifters:
    def __call__(self, circuit):
        ops = circuit.ops[:]; merged = True
        while merged:
            merged = False; new_ops = []; i = 0
            while i < len(ops):
                if (i+1 < len(ops) and ops[i].name=="PhaseShifter"
                        and ops[i+1].name=="PhaseShifter" and ops[i].modes==ops[i+1].modes):
                    new_ops.append(GateOp("PhaseShifter",ops[i].modes,{"phi":ops[i].params["phi"]+ops[i+1].params["phi"]},tag="merged"))
                    i+=2; merged=True
                else:
                    new_ops.append(ops[i]); i+=1
            ops = new_ops
        return _clone(circuit, ops)

class DecomposeTwoMode:
    def __call__(self, circuit):
        new_ops = []
        for op in circuit.ops:
            if op.name=="TwoModeSqueeze":
                r,phi = op.params["r"],op.params.get("phi",0.0); m0,m1 = op.modes
                if abs(phi)>_EPS:
                    new_ops.append(GateOp("PhaseShifter",(m0,),{"phi": phi/2}))
                    new_ops.append(GateOp("PhaseShifter",(m1,),{"phi":-phi/2}))
                new_ops.append(GateOp("Beamsplitter",(m0,m1),{"theta":math.pi/4,"phi":0.}))
                new_ops.append(GateOp("Squeezer",(m0,),{"r": r,"phi":0.}))
                new_ops.append(GateOp("Squeezer",(m1,),{"r":-r,"phi":0.}))
                new_ops.append(GateOp("Beamsplitter",(m0,m1),{"theta":-math.pi/4,"phi":0.}))
            else:
                new_ops.append(op)
        return _clone(circuit, new_ops)

@dataclass
class ResourceReport:
    gate_counts: dict
    depth: int
    n_modes: int
    squeezing_budget_db: float
    has_non_gaussian: bool
    @property
    def n_beamsplitters(self): return self.gate_counts.get("Beamsplitter",0)
    @property
    def n_squeezers(self): return self.gate_counts.get("Squeezer",0)+self.gate_counts.get("TwoModeSqueeze",0)
    def __repr__(self):
        return (f"ResourceReport(n_modes={self.n_modes}, depth={self.depth}, "
                f"BSs={self.n_beamsplitters}, Sq={self.n_squeezers}, "
                f"sq_budget={self.squeezing_budget_db:.1f}dB)")

class ResourceEstimator:
    def __init__(self): self.last_report = None
    def __call__(self, circuit):
        counts = circuit.gate_counts_by_type()
        max_r = max((abs(op.params.get("r",0.)) for op in circuit.ops
                     if op.name in ("Squeezer","TwoModeSqueeze")), default=0.)
        sq_db = 10*math.log10(math.exp(2*max_r)) if max_r>0 else 0.
        self.last_report = ResourceReport(gate_counts=counts, depth=circuit.depth,
            n_modes=circuit.n_modes, squeezing_budget_db=sq_db,
            has_non_gaussian=circuit.has_non_gaussian)
        return circuit, self.last_report

class PassPipeline:
    def __init__(self, passes=None): self.passes = list(passes or [])
    def add(self, p): self.passes.append(p); return self
    def run(self, circuit):
        for p in self.passes: circuit = p(circuit)
        return circuit
    @classmethod
    def default(cls): return cls([RemoveIdentityGates(), MergePhaseShifters()])
    @classmethod
    def for_hardware(cls): return cls([RemoveIdentityGates(), MergePhaseShifters(), DecomposeTwoMode()])

class Compiler:
    def __init__(self, pipeline=None, decompose_interferometers=True, target="GaussianSimulator"):
        self.pipeline = pipeline or PassPipeline.default()
        self.decompose_interferometers = decompose_interferometers
        self.target = target
    def compile(self, circuit):
        import numpy as np
        if self.decompose_interferometers:
            circuit = self._expand(circuit)
        circuit = self.pipeline.run(circuit)
        _, report = ResourceEstimator()(circuit)
        return circuit, report
    def _expand(self, circuit):
        new_ops = []
        for op in circuit.ops:
            if op.name=="Interferometer":
                import numpy as np
                U = np.array(op.params["U"],dtype=complex)
                res = clements_decompose(U)
                for m0,m1,theta,phi in res.beamsplitters:
                    if abs(phi)>1e-12: new_ops.append(GateOp("PhaseShifter",(m0,),{"phi":phi}))
                    new_ops.append(GateOp("Beamsplitter",(m0,m1),{"theta":theta,"phi":0.}))
                for mode,phi in res.phases:
                    if abs(phi)>1e-12: new_ops.append(GateOp("PhaseShifter",(mode,),{"phi":phi}))
            else:
                new_ops.append(op)
        return _clone(circuit, new_ops)

def _clone(circuit, new_ops):
    out = PhotonicCircuit(n_modes=circuit.n_modes, metadata=circuit.metadata)
    out.circuit_id = circuit.circuit_id
    out.ops = list(new_ops)
    out.measurements = list(circuit.measurements)
    return out
