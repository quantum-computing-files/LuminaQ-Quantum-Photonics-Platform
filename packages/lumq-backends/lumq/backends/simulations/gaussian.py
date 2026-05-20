"""
lumq.backends.simulators.gaussian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gaussian (covariance-matrix) simulator.

This is the fast, exact simulator for circuits composed entirely of Gaussian
(symplectic) gates — beamsplitters, phase shifters, squeezers, displacers,
and arbitrary linear-optical unitaries.

Algorithm
---------
State representation: (μ, V) — 2N displacement vector + 2N×2N covariance matrix.
Each Gaussian gate is a symplectic transformation S and displacement shift d:
    μ ← S μ + d
    V ← S V Sᵀ

This is O(N²) per gate and O(N³) for an N-mode Clements interferometer,
making it tractable for circuits with hundreds of modes.

Measurement
-----------
Homodyne and heterodyne outcomes are sampled from the exact marginal Gaussians.
Photon-number sampling uses the thermal + displacement approximation (exact
Fock-diagonal elements require the Hafnian, delegated to GaussianFockSampler).

JAX backend
-----------
All matrix operations use JAX so that gradients flow through the simulator —
essential for variational CV algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from lumq.backends.interface import Device, DeviceCapabilities, JobResult

__all__ = ["GaussianSimulator", "GaussianSimulatorConfig"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GaussianSimulatorConfig:
    """Runtime configuration for the Gaussian simulator.

    Parameters
    ----------
    n_modes : int
        Maximum number of optical modes.
    hbar : float
        Reduced Planck constant convention (default 2).
    seed : int
        PRNG seed for measurement sampling.
    track_state : bool
        If True, the full (μ, V) state is included in JobResult.state.
    compute_ev : list[str]
        Which expectation values to compute automatically.
        Supported: "mean_photon", "x_mean", "p_mean", "squeezing_spectrum".
    """

    n_modes: int = 8
    hbar: float = 2.0
    seed: int = 0
    track_state: bool = True
    compute_ev: list[str] = field(default_factory=lambda: ["mean_photon"])


# ---------------------------------------------------------------------------
# Gaussian Simulator
# ---------------------------------------------------------------------------


class GaussianSimulator(Device):
    """Exact Gaussian (covariance matrix) photonic circuit simulator.

    Parameters
    ----------
    config : GaussianSimulatorConfig, optional
        Runtime configuration.  If None, uses sensible defaults.

    Usage
    -----
    >>> from lumq.backends.simulators.gaussian import GaussianSimulator
    >>> from lumq.compiler.ir import PhotonicCircuit
    >>> from lumq.compiler.ir import GateOp
    >>>
    >>> sim = GaussianSimulator()
    >>> circuit = PhotonicCircuit(n_modes=2)
    >>> circuit.add(GateOp("Squeezer", modes=(0,), params={"r": 1.0, "phi": 0.0}))
    >>> circuit.add(GateOp("Beamsplitter", modes=(0, 1), params={"theta": 0.785, "phi": 0.0}))
    >>> result = sim.run(circuit, shots=100)
    >>> print(result)
    """

    def __init__(self, config: Optional[GaussianSimulatorConfig] = None) -> None:
        self.config = config or GaussianSimulatorConfig()
        self._key = jax.random.PRNGKey(self.config.seed)

    # ------------------------------------------------------------------
    # Device interface
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            name="GaussianSimulator",
            n_modes=self.config.n_modes,
            supports_gaussian=True,
            supports_fock=False,
            supports_non_gaussian=False,
            native_gates=[
                "Beamsplitter", "PhaseShifter", "Squeezer",
                "TwoModeSqueeze", "Displacer", "Interferometer",
            ],
            max_squeezing_db=30.0,
            is_hardware=False,
        )

    def run_circuit(self, circuit, shots: int = 1) -> JobResult:
        """Execute a PhotonicCircuit using covariance matrix evolution.

        Parameters
        ----------
        circuit : PhotonicCircuit
        shots : int
            Number of measurement samples to draw.
        """
        # Initialise vacuum state
        N = circuit.n_modes
        hbar = self.config.hbar
        mu  = jnp.zeros(2 * N)
        cov = (hbar / 2.0) * jnp.eye(2 * N)

        # Apply each gate operation
        for op in circuit.ops:
            mu, cov = self._apply_op(op, mu, cov, N, hbar)

        # Collect measurement samples
        samples = None
        if circuit.measurements:
            samples = self._sample_measurements(
                circuit.measurements, mu, cov, shots, hbar
            )

        # Build final GaussianState for return
        state = None
        if self.config.track_state:
            from lumq.photonics.states import GaussianState
            state = GaussianState(mu=mu, cov=cov, hbar=hbar)

        # Compute requested expectation values
        ev = self._compute_ev(mu, cov, N, hbar)

        return JobResult(
            backend="GaussianSimulator",
            shots=shots,
            samples=samples,
            state=state,
            expectation_values=ev,
            metadata={"n_modes": N, "n_ops": len(circuit.ops)},
        )

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def _apply_op(
        self,
        op,
        mu: Array,
        cov: Array,
        n_modes: int,
        hbar: float,
    ) -> tuple[Array, Array]:
        """Apply a single GateOp to (mu, cov) and return updated (mu, cov)."""
        from lumq.photonics.gates._symplectic import (
            S_beamsplitter, S_displacer_vec, S_phase_shifter,
            S_squeezer, S_two_mode_squeeze,
            embed_single, embed_two,
        )

        name   = op.name
        modes  = op.modes
        params = op.params

        # ── Single-mode gates ──────────────────────────────────────────
        if name == "PhaseShifter":
            S1 = S_phase_shifter(params["phi"])
            S  = embed_single(S1, modes[0], n_modes)
            return S @ mu, S @ cov @ S.T

        if name == "Squeezer":
            S1 = S_squeezer(params["r"], params.get("phi", 0.0))
            S  = embed_single(S1, modes[0], n_modes)
            return S @ mu, S @ cov @ S.T

        if name == "Displacer":
            alpha = params["alpha"]
            d_single = S_displacer_vec(alpha, hbar=hbar)
            d = jnp.zeros(2 * n_modes)
            i = 2 * modes[0]
            d = d.at[i : i + 2].set(d_single)
            return mu + d, cov      # displacement leaves cov unchanged

        # ── Two-mode gates ─────────────────────────────────────────────
        if name == "Beamsplitter":
            S2 = S_beamsplitter(params["theta"], params.get("phi", 0.0))
            S  = embed_two(S2, (modes[0], modes[1]), n_modes)
            return S @ mu, S @ cov @ S.T

        if name == "TwoModeSqueeze":
            S2 = S_two_mode_squeeze(params["r"], params.get("phi", 0.0))
            S  = embed_two(S2, (modes[0], modes[1]), n_modes)
            return S @ mu, S @ cov @ S.T

        # ── N-mode: Interferometer ──────────────────────────────────────
        if name == "Interferometer":
            U  = jnp.asarray(params["U"], dtype=jnp.complex128)
            Re = jnp.real(U)
            Im = jnp.imag(U)
            Nm = U.shape[0]
            S  = jnp.zeros((2 * n_modes, 2 * n_modes))
            # Build 2N×2N symplectic from U
            for i in range(Nm):
                for j in range(Nm):
                    S = S.at[2*i,   2*j  ].set( Re[i, j])
                    S = S.at[2*i,   2*j+1].set(-Im[i, j])
                    S = S.at[2*i+1, 2*j  ].set( Im[i, j])
                    S = S.at[2*i+1, 2*j+1].set( Re[i, j])
            return S @ mu, S @ cov @ S.T

        raise ValueError(
            f"GaussianSimulator: unknown or non-Gaussian gate '{name}'. "
            "Use GaussianFockSimulator for non-Gaussian gates."
        )

    # ------------------------------------------------------------------
    # Measurement sampling
    # ------------------------------------------------------------------

    def _sample_measurements(
        self,
        measurements: list,
        mu: Array,
        cov: Array,
        shots: int,
        hbar: float,
    ) -> Array:
        """Draw measurement samples from the final Gaussian state.

        Returns an (shots, n_measured_modes) array of quadrature outcomes.
        """
        results = []
        for meas_op in measurements:
            mode = meas_op.modes[0]
            phi  = meas_op.params.get("phi", 0.0)
            i    = 2 * mode

            # Marginal of x_phi = x cos(phi) + p sin(phi)
            c = jnp.array([jnp.cos(phi), jnp.sin(phi)])
            mu_m    = jnp.dot(c, mu[i : i + 2])
            sigma2  = jnp.dot(c, cov[i : i + 2, i : i + 2] @ c)

            # Sample shots outcomes
            self._key, subkey = jax.random.split(self._key)
            outcomes = mu_m + jnp.sqrt(sigma2) * jax.random.normal(
                subkey, shape=(shots,)
            )
            results.append(outcomes)

        return jnp.stack(results, axis=1)  # (shots, n_measured)

    # ------------------------------------------------------------------
    # Expectation values
    # ------------------------------------------------------------------

    def _compute_ev(
        self, mu: Array, cov: Array, n_modes: int, hbar: float
    ) -> dict[str, Array]:
        ev: dict[str, Array] = {}
        if "mean_photon" in self.config.compute_ev:
            n_bar = jnp.array([
                (cov[2*k, 2*k] + cov[2*k+1, 2*k+1] +
                 mu[2*k]**2 + mu[2*k+1]**2) / hbar - 0.5
                for k in range(n_modes)
            ])
            ev["mean_photon"] = n_bar
        if "x_mean" in self.config.compute_ev:
            ev["x_mean"] = mu[0::2]
        if "p_mean" in self.config.compute_ev:
            ev["p_mean"] = mu[1::2]
        return ev

    # ------------------------------------------------------------------
    # Gradient-enabled expectation value (for variational algorithms)
    # ------------------------------------------------------------------

    def expectation_value(
        self,
        circuit_fn,
        observable_fn,
        params: Array,
        shots: int = 0,
    ) -> Array:
        """Compute ⟨O⟩ with JAX-differentiable circuit execution.

        Parameters
        ----------
        circuit_fn : callable
            Function (params) -> PhotonicCircuit.  Must use JAX-compatible
            parameter handling (jnp arrays).
        observable_fn : callable
            Function (GaussianState) -> scalar Array.
        params : Array
            Trainable parameters passed to circuit_fn.
        shots : int
            0 = analytic (exact expectation), >0 = shot-noise estimate.
        """
        circuit = circuit_fn(params)
        result  = self.run_circuit(circuit, shots=max(shots, 1))
        return observable_fn(result.state)
