"""
lumq.backends.interface
~~~~~~~~~~~~~~~~~~~~~~~
Abstract Device base class.

Every backend — simulator or hardware — implements this interface.
Algorithm code written against `Device` runs unchanged on any backend.

Protocol
--------
1. Build a `PhotonicCircuit` (lumq-compiler).
2. Pass it to `device.run(circuit, shots=N)`.
3. Receive a `JobResult` — always the same schema regardless of backend.

The backend is responsible for:
  - Compiling the circuit to its native representation
  - Executing (simulating or submitting to hardware)
  - Returning results in the standard `JobResult` format
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import jax.numpy as jnp
from jax import Array

__all__ = ["Device", "JobResult", "DeviceCapabilities", "BackendError"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class JobResult:
    """Standardised result container returned by every backend.

    Attributes
    ----------
    job_id : str
        Unique identifier for this execution.
    backend : str
        Name of the backend that produced this result.
    shots : int
        Number of circuit repetitions executed.
    samples : Array | None
        Raw measurement samples, shape (shots, n_modes).  None for
        expectation-value-only backends.
    state : Any | None
        Final quantum state object (GaussianState, FockState, etc.).
        None for hardware backends that don't return the state.
    expectation_values : dict[str, Array]
        Named expectation values computed from the state or samples.
    metadata : dict
        Backend-specific metadata (runtime, resource usage, etc.).
    """

    job_id: str = field(default_factory=lambda: str(uuid4())[:8])
    backend: str = "unknown"
    shots: int = 1
    samples: Optional[Array] = None
    state: Any = None
    expectation_values: dict[str, Array] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        ev_keys = list(self.expectation_values.keys())
        return (
            f"JobResult(backend='{self.backend}', shots={self.shots}, "
            f"ev={ev_keys}, has_state={self.state is not None})"
        )


# ---------------------------------------------------------------------------
# Device capabilities descriptor
# ---------------------------------------------------------------------------


@dataclass
class DeviceCapabilities:
    """Describes what a backend can do.

    Used by the compiler to choose optimisation passes and gate decompositions
    appropriate for the target device.
    """

    name: str
    n_modes: int
    supports_gaussian: bool = True
    supports_fock: bool = False
    supports_non_gaussian: bool = False
    native_gates: list[str] = field(default_factory=lambda: [
        "Beamsplitter", "PhaseShifter", "Squeezer", "Displacer"
    ])
    max_squeezing_db: float = 20.0       # practical squeezing limit
    loss_per_mode_db: float = 0.0        # propagation loss model
    detector_efficiency: float = 1.0    # PNR detector efficiency
    supports_heterodyne: bool = True
    supports_homodyne: bool = True
    supports_pnr: bool = False
    is_hardware: bool = False

    def supports_gate(self, gate_name: str) -> bool:
        return gate_name in self.native_gates


# ---------------------------------------------------------------------------
# Abstract Device
# ---------------------------------------------------------------------------


class Device(ABC):
    """Abstract base class for all LuminaQ backends.

    Subclass and implement `run_circuit()` and `capabilities` to register
    a new backend.
    """

    @property
    @abstractmethod
    def capabilities(self) -> DeviceCapabilities:
        """Return the DeviceCapabilities descriptor for this backend."""

    @abstractmethod
    def run_circuit(self, circuit, shots: int = 1) -> JobResult:
        """Execute a PhotonicCircuit and return a JobResult.

        Parameters
        ----------
        circuit : PhotonicCircuit (lumq.compiler.ir)
            The compiled photonic circuit to execute.
        shots : int
            Number of repetitions (samples).
        """

    def run(self, circuit, shots: int = 1) -> JobResult:
        """Public entry point — validates then delegates to run_circuit()."""
        n_circ = circuit.n_modes
        n_dev  = self.capabilities.n_modes
        if n_circ > n_dev:
            raise BackendError(
                f"Circuit requires {n_circ} modes but {self.capabilities.name} "
                f"supports at most {n_dev}."
            )
        return self.run_circuit(circuit, shots=shots)

    def __repr__(self) -> str:
        c = self.capabilities
        return f"{self.__class__.__name__}(n_modes={c.n_modes}, hw={c.is_hardware})"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class BackendError(RuntimeError):
    """Raised when a backend cannot execute the requested circuit."""
