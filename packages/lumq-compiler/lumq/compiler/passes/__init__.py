"""
lumq.compiler.passes
~~~~~~~~~~~~~~~~~~~~~
Optimisation pass pipeline for PhotonicCircuit IR.

Each pass is a callable (circuit: PhotonicCircuit) -> PhotonicCircuit.
Passes are composed via PassPipeline.

Available passes
----------------
MergePhaseShifters  — collapse adjacent single-mode PS ops on the same mode
RemoveIdentityGates — drop zero-parameter gates (r=0, theta=0, etc.)
DecomposeTwoMode    — expand TwoModeSqueeze → Sq + BS + Sq (for single-squeezer hardware)
ResourceEstimator   — annotate circuit with resource counts (read-only pass)
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = [
    "PassPipeline",
    "MergePhaseShifters",
    "RemoveIdentityGates",
    "DecomposeTwoMode",
    "ResourceEstimator",
    "ResourceReport",
]

# Tolerance for "effectively zero" parameter checks
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Pass pipeline
# ---------------------------------------------------------------------------


class PassPipeline:
    """Ordered list of optimisation passes applied to a PhotonicCircuit.

    Usage
    -----
    >>> pipeline = PassPipeline([RemoveIdentityGates(), MergePhaseShifters()])
    >>> optimised = pipeline.run(circuit)
    """

    def __init__(self, passes: list[Callable] | None = None) -> None:
        self.passes = list(passes or [])

    def add(self, pass_fn: Callable) -> "PassPipeline":
        self.passes.append(pass_fn)
        return self

    def run(self, circuit):
        """Apply all passes in order and return the transformed circuit."""
        for p in self.passes:
            circuit = p(circuit)
        return circuit

    @classmethod
    def default(cls) -> "PassPipeline":
        """Recommended default pipeline for Gaussian circuits."""
        return cls([
            RemoveIdentityGates(),
            MergePhaseShifters(),
        ])

    @classmethod
    def for_hardware(cls) -> "PassPipeline":
        """Pipeline for hardware deployment (adds TMS decomposition)."""
        return cls([
            RemoveIdentityGates(),
            MergePhaseShifters(),
            DecomposeTwoMode(),
        ])


# ---------------------------------------------------------------------------
# Pass: remove identity gates
# ---------------------------------------------------------------------------


class RemoveIdentityGates:
    """Remove gates whose parameters make them the identity operation.

    - Squeezer(r=0)
    - PhaseShifter(phi=0)
    - Beamsplitter(theta=0)
    - Displacer(alpha=0)
    """

    def __call__(self, circuit):
        from lumq.compiler.ir import PhotonicCircuit
        new_ops = []
        removed = 0
        for op in circuit.ops:
            if self._is_identity(op):
                removed += 1
            else:
                new_ops.append(op)
        out = PhotonicCircuit(n_modes=circuit.n_modes, metadata=circuit.metadata)
        out.circuit_id = circuit.circuit_id
        out.ops = new_ops
        out.measurements = circuit.measurements
        if removed:
            out.metadata.tags.append(f"RemoveIdentityGates:{removed}")
        return out

    def _is_identity(self, op) -> bool:
        p = op.params
        if op.name == "Squeezer"    and abs(p.get("r",   1.0)) < _EPS:
            return True
        if op.name == "PhaseShifter" and abs(p.get("phi", 1.0)) < _EPS:
            return True
        if op.name == "Beamsplitter" and abs(p.get("theta", 1.0)) < _EPS:
            return True
        if op.name == "Displacer"    and abs(p.get("alpha", 1.0)) < _EPS:
            return True
        return False


# ---------------------------------------------------------------------------
# Pass: merge adjacent phase shifters
# ---------------------------------------------------------------------------


class MergePhaseShifters:
    """Merge consecutive PhaseShifter ops on the same mode into one.

    [PS(phi_1), PS(phi_2)] → [PS(phi_1 + phi_2)]

    Only merges pairs that are adjacent with no interleaved gate on that mode.
    """

    def __call__(self, circuit):
        from lumq.compiler.ir import GateOp, PhotonicCircuit

        ops = circuit.ops[:]
        merged = True
        while merged:
            merged = False
            new_ops = []
            i = 0
            while i < len(ops):
                if (
                    i + 1 < len(ops)
                    and ops[i].name == "PhaseShifter"
                    and ops[i + 1].name == "PhaseShifter"
                    and ops[i].modes == ops[i + 1].modes
                ):
                    combined_phi = ops[i].params["phi"] + ops[i + 1].params["phi"]
                    new_ops.append(
                        GateOp("PhaseShifter", ops[i].modes, {"phi": combined_phi}, tag="merged")
                    )
                    i += 2
                    merged = True
                else:
                    new_ops.append(ops[i])
                    i += 1
            ops = new_ops

        out = PhotonicCircuit(n_modes=circuit.n_modes, metadata=circuit.metadata)
        out.circuit_id = circuit.circuit_id
        out.ops = ops
        out.measurements = circuit.measurements
        return out


# ---------------------------------------------------------------------------
# Pass: decompose TwoModeSqueeze
# ---------------------------------------------------------------------------


class DecomposeTwoMode:
    """Decompose TwoModeSqueeze(r, phi) into Sq + BS + Sq primitives.

    This is required for hardware platforms that only implement single-mode
    squeezing (most current photonic chips).

    Decomposition (Serafini 2017, Eq. 4.81):
        TMS(r, 0) = BS(π/4) · [Sq(r) ⊗ Sq(r)] · BS(π/4)†

    For general phi, add PhaseShifter(phi/2) on each mode.
    """

    def __call__(self, circuit):
        from lumq.compiler.ir import GateOp, PhotonicCircuit

        new_ops = []
        for op in circuit.ops:
            if op.name == "TwoModeSqueeze":
                r, phi = op.params["r"], op.params.get("phi", 0.0)
                m0, m1 = op.modes[0], op.modes[1]
                # Phase pre-rotation
                if abs(phi) > _EPS:
                    new_ops.append(GateOp("PhaseShifter", (m0,), {"phi":  phi / 2}))
                    new_ops.append(GateOp("PhaseShifter", (m1,), {"phi": -phi / 2}))
                # BS(π/4) → two single-mode squeezers → BS(π/4)†
                new_ops.append(GateOp("Beamsplitter", (m0, m1), {"theta": math.pi / 4, "phi": 0.0}))
                new_ops.append(GateOp("Squeezer", (m0,), {"r":  r, "phi": 0.0}))
                new_ops.append(GateOp("Squeezer", (m1,), {"r": -r, "phi": 0.0}))
                new_ops.append(GateOp("Beamsplitter", (m0, m1), {"theta": -math.pi / 4, "phi": 0.0}))
            else:
                new_ops.append(op)

        out = PhotonicCircuit(n_modes=circuit.n_modes, metadata=circuit.metadata)
        out.circuit_id = circuit.circuit_id
        out.ops = new_ops
        out.measurements = circuit.measurements
        return out


# ---------------------------------------------------------------------------
# Resource estimator (read-only pass)
# ---------------------------------------------------------------------------


class ResourceReport:
    """Resource analysis report for a photonic circuit."""

    def __init__(
        self,
        gate_counts: dict[str, int],
        depth: int,
        n_modes: int,
        squeezing_budget_db: float,
        has_non_gaussian: bool,
    ) -> None:
        self.gate_counts        = gate_counts
        self.depth              = depth
        self.n_modes            = n_modes
        self.squeezing_budget_db = squeezing_budget_db
        self.has_non_gaussian   = has_non_gaussian

    @property
    def n_beamsplitters(self) -> int:
        return self.gate_counts.get("Beamsplitter", 0)

    @property
    def n_squeezers(self) -> int:
        return self.gate_counts.get("Squeezer", 0) + self.gate_counts.get("TwoModeSqueeze", 0)

    def __repr__(self) -> str:
        return (
            f"ResourceReport(\n"
            f"  n_modes={self.n_modes}, depth={self.depth},\n"
            f"  beamsplitters={self.n_beamsplitters}, squeezers={self.n_squeezers},\n"
            f"  squeezing_budget={self.squeezing_budget_db:.1f} dB,\n"
            f"  non_gaussian={self.has_non_gaussian}\n"
            f")"
        )


class ResourceEstimator:
    """Analyse circuit resources without modifying the circuit.

    Attaches a ResourceReport to circuit.metadata.tags and returns the
    unchanged circuit.  Access the report via `.last_report`.
    """

    def __init__(self) -> None:
        self.last_report: ResourceReport | None = None

    def __call__(self, circuit) -> tuple:
        counts = circuit.gate_counts_by_type()

        # Max squeezing parameter across all squeezing gates
        max_r = 0.0
        for op in circuit.ops:
            if op.name in ("Squeezer", "TwoModeSqueeze"):
                max_r = max(max_r, abs(op.params.get("r", 0.0)))

        squeezing_db = 10.0 * math.log10(math.exp(2 * max_r)) if max_r > 0 else 0.0

        self.last_report = ResourceReport(
            gate_counts=counts,
            depth=circuit.depth,
            n_modes=circuit.n_modes,
            squeezing_budget_db=squeezing_db,
            has_non_gaussian=circuit.has_non_gaussian,
        )
        return circuit, self.last_report
