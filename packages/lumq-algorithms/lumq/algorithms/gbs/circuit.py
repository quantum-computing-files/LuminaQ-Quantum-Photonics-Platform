"""
lumq.algorithms.gbs.circuit
============================
Build photonic circuits for Gaussian Boson Sampling.

Two entry points:
  gbs_circuit(r, U)        - from squeezing params + interferometer
  gbs_circuit_from_graph(A) - from adjacency matrix via Takagi decomposition
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from typing import Optional

__all__ = ["gbs_circuit", "gbs_circuit_from_graph"]


def gbs_circuit(
    r: np.ndarray,
    U: np.ndarray,
    phi: Optional[np.ndarray] = None,
    n_modes: Optional[int] = None,
) -> object:
    """Build a GBS circuit: squeeze each mode then apply interferometer U.

    Circuit structure:
        |0> --[Sq(r_k)]-- [U] --[PNR]

    Parameters
    ----------
    r : 1D array, shape (N,)
        Squeezing parameters per mode.
    U : 2D complex array, shape (N, N)
        Interferometer unitary.
    phi : 1D array, shape (N,), optional
        Squeezing angles.  Defaults to zeros.
    n_modes : int, optional
        Explicit mode count.  Defaults to len(r).

    Returns
    -------
    PhotonicCircuit ready to run on GaussianSimulator.
    """
    from lumq.compiler.ir import CircuitMetadata, PhotonicCircuit

    r   = np.asarray(r, dtype=float)
    U   = np.asarray(U, dtype=complex)
    N   = n_modes or len(r)
    phi = np.zeros(N) if phi is None else np.asarray(phi, dtype=float)

    if len(r) != N or U.shape != (N, N):
        raise ValueError(f"r has length {len(r)}, U has shape {U.shape}, expected N={N}.")

    circ = PhotonicCircuit(
        n_modes=N,
        metadata=CircuitMetadata(name=f"GBS-{N}mode", created_by="lumq-algorithms"),
    )

    # Single-mode squeezers
    for k in range(N):
        if abs(r[k]) > 1e-10:
            circ.sq(k, r=float(r[k]), phi=float(phi[k]))

    # Linear interferometer
    circ.interferometer(U)

    # PNR measurements on all modes
    for k in range(N):
        circ.meas_pnr(k)

    return circ


def gbs_circuit_from_graph(
    A: np.ndarray,
    scale: float = 1.0,
    max_squeezing: float = 1.5,
) -> tuple:
    """Build a GBS circuit encoding a graph adjacency matrix.

    Uses the Takagi decomposition A = U diag(lambda) U^T to find
    squeezing parameters r_k = arctanh(scale * lambda_k).

    Parameters
    ----------
    A : 2D real symmetric array, shape (N, N)
        Graph adjacency matrix.  Edge weights in [-1, 1].
    scale : float
        Rescales eigenvalues before arctanh.  Controls mean photon number.
    max_squeezing : float
        Clip squeezing parameters to this maximum.

    Returns
    -------
    (circuit, r, U) tuple:
        circuit : PhotonicCircuit
        r       : squeezing parameters (N,)
        U       : interferometer unitary (N, N)
    """
    A = np.asarray(A, dtype=complex)
    N = A.shape[0]

    if A.shape != (N, N):
        raise ValueError(f"Adjacency matrix must be square, got {A.shape}.")

    # Takagi decomposition of symmetric A = U D U^T
    # Since A is real symmetric: eigendecomposition gives A = Q L Q^T
    # For complex symmetric: use SVD-based Takagi decomp
    lam, U = _takagi(A)

    # Squeezing parameters: r_k = arctanh(scale * lambda_k)
    lam_scaled = np.clip(scale * lam, -0.999, 0.999)
    r = np.arctanh(lam_scaled)
    r = np.clip(r, -max_squeezing, max_squeezing)

    circuit = gbs_circuit(r, U)
    return circuit, r, U


def _takagi(A: np.ndarray) -> tuple:
    """Takagi decomposition of symmetric matrix A = U D U^T.
    Returns (singular values, U).
    """
    # For real symmetric: eigendecomposition suffices
    A_real = np.real(A)
    if np.allclose(A, A_real):
        lam, Q = np.linalg.eigh(A_real)
        lam = np.abs(lam)
        return lam, Q
    # General symmetric: use SVD
    # A = U S V^H, and for symmetric A, V = U* so A = U S U^T
    U, s, Vh = np.linalg.svd(A)
    return s, U
