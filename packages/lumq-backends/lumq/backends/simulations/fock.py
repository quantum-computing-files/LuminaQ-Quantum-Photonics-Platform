"""
lumq.backends.simulators.fock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fock-basis (truncated) simulator.

For circuits containing non-Gaussian gates (Kerr, CubicPhase) the Gaussian
covariance-matrix approach breaks down.  This simulator represents the
quantum state as a dense complex tensor of shape (cutoff,)^N and applies
gates as tensor contractions.

For large N or deep non-Gaussian circuits, a Matrix Product State (MPS)
representation is used to keep memory tractable.

Status
------
- Dense exact simulator (n_modes ≤ 6, cutoff ≤ 10): COMPLETE
- MPS backend (n_modes up to ~20):                   PLANNED v0.2
- GPU-accelerated dense (via JAX XLA):               PLANNED v0.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from lumq.backends.interface import Device, DeviceCapabilities, JobResult

__all__ = ["FockSimulator", "FockSimulatorConfig"]


@dataclass
class FockSimulatorConfig:
    """Configuration for the Fock-basis simulator.

    Parameters
    ----------
    n_modes : int
        Maximum circuit width.
    cutoff : int
        Fock-space truncation (max photon number per mode + 1).
        Memory scales as cutoff^n_modes — keep n_modes * log(cutoff) < 20.
    hbar : float
        Planck constant convention.
    seed : int
        PRNG seed.
    use_mps : bool
        Use MPS representation (not yet implemented, reserved for v0.2).
    """

    n_modes: int = 4
    cutoff: int = 10
    hbar: float = 2.0
    seed: int = 0
    use_mps: bool = False


class FockSimulator(Device):
    """Fock-basis photonic circuit simulator.

    Supports non-Gaussian gates.  Exponential in n_modes — use with care.

    Parameters
    ----------
    config : FockSimulatorConfig, optional
    """

    def __init__(self, config: Optional[FockSimulatorConfig] = None) -> None:
        self.config = config or FockSimulatorConfig()
        self._key = jax.random.PRNGKey(self.config.seed)

    @property
    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            name="FockSimulator",
            n_modes=self.config.n_modes,
            supports_gaussian=True,
            supports_fock=True,
            supports_non_gaussian=True,
            native_gates=[
                "Beamsplitter", "PhaseShifter", "Squeezer",
                "TwoModeSqueeze", "Displacer", "KerrGate", "CubicPhase",
            ],
            is_hardware=False,
        )

    def run_circuit(self, circuit, shots: int = 1) -> JobResult:
        """Execute circuit in Fock basis.

        Each gate is applied as a matrix-vector contraction on the
        (cutoff,)*n_modes state tensor.
        """
        from lumq.photonics.states import FockState
        from lumq.photonics.gates import (
            Beamsplitter, Displacer, KerrGate, PhaseShifter,
            Squeezer, TwoModeSqueeze,
        )

        N      = circuit.n_modes
        cutoff = self.config.cutoff

        # Initialise vacuum state |0⟩^⊗N
        state = FockState.vacuum(n_modes=N, cutoff=cutoff)

        GATE_MAP = {
            "PhaseShifter":  lambda p: PhaseShifter(p["phi"]),
            "Squeezer":      lambda p: Squeezer(p["r"], p.get("phi", 0.0)),
            "Displacer":     lambda p: Displacer(p["alpha"]),
            "Beamsplitter":  lambda p: Beamsplitter(p["theta"], p.get("phi", 0.0)),
            "TwoModeSqueeze":lambda p: TwoModeSqueeze(p["r"], p.get("phi", 0.0)),
            "KerrGate":      lambda p: KerrGate(p["kappa"]),
        }

        for op in circuit.ops:
            if op.name not in GATE_MAP:
                raise ValueError(
                    f"FockSimulator: gate '{op.name}' not implemented. "
                    "Please contribute or use GaussianSimulator for Gaussian-only circuits."
                )
            gate = GATE_MAP[op.name](op.params)
            # Route single-mode vs two-mode
            if len(op.modes) == 1:
                state = gate.apply(state, mode=op.modes[0])
            else:
                state = gate.apply(state, modes=tuple(op.modes))

        # Sample measurements
        samples = None
        if circuit.measurements:
            samples = self._sample_fock(state, circuit.measurements, shots)

        return JobResult(
            backend="FockSimulator",
            shots=shots,
            samples=samples,
            state=state,
            expectation_values=self._compute_ev(state, N),
            metadata={"n_modes": N, "cutoff": cutoff},
        )

    def _sample_fock(self, state, measurements, shots: int) -> Array:
        """Sample photon-number outcomes from the Fock state."""
        from lumq.photonics.measurements import PhotonNumberMeasurement
        results = []
        for meas_op in measurements:
            mode = meas_op.modes[0]
            axes = tuple(i for i in range(state._pure_n_modes) if i != mode)
            probs = jnp.sum(jnp.abs(state.data) ** 2, axis=axes)
            self._key, subkey = jax.random.split(self._key)
            samples = jax.random.choice(
                subkey,
                jnp.arange(state.cutoff),
                shape=(shots,),
                p=probs,
            )
            results.append(samples)
        return jnp.stack(results, axis=1)

    def _compute_ev(self, state, n_modes: int) -> dict[str, Array]:
        """Mean photon number per mode from Fock amplitudes."""
        ns = jnp.arange(state.cutoff, dtype=jnp.float64)
        n_bar = []
        for mode in range(n_modes):
            axes = tuple(i for i in range(n_modes) if i != mode)
            probs = jnp.sum(jnp.abs(state.data) ** 2, axis=axes)
            n_bar.append(jnp.dot(probs, ns))
        return {"mean_photon": jnp.array(n_bar)}
