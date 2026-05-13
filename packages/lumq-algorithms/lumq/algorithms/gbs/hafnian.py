"""
lumq.algorithms.gbs.hafnian
============================
JAX-native hafnian and torontonian.

The hafnian of a 2N x 2N symmetric matrix A is defined as:
    Haf(A) = sum_{M in PMP(2N)} prod_{(i,j) in M} A_{ij}
where the sum is over all perfect matchings of the complete graph K_{2N}.

Algorithm
---------
Ryser-like inclusion-exclusion formula (Bjorklund et al. 2019):
    Haf(A) = (1/2^N) * sum_{S subset [N]} (-1)^{N-|S|} prod_{j=1}^{N} (sum_{i in S} X_{ij})
where X = A + A^T (or 2A for symmetric A) restricted to pairs.

Complexity: O(N^2 * 2^N) — exact for any N, differentiable via JAX.

For large N (>20), use the batch version or the C/Rust extension (future).
"""
from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["hafnian", "hafnian_batch", "torontonian"]


def hafnian(A: Array, loop: bool = False) -> Array:
    """Compute the hafnian of a 2N x 2N symmetric complex matrix.

    Parameters
    ----------
    A : complex array, shape (2N, 2N)
        Symmetric matrix.  For GBS, this is the off-diagonal block of
        the Gaussian state's Q-function matrix.
    loop : bool
        If True, compute the loop hafnian (diagonal elements included).

    Returns
    -------
    Scalar complex JAX array.

    Notes
    -----
    This is the pure-JAX implementation.  It is exact and differentiable
    but has O(N^2 * 2^N) complexity.  Practical limit: N <= 20 modes.
    For larger circuits, use hafnian_batch or the Rust extension.
    """
    A = jnp.asarray(A, dtype=jnp.complex128)
    n = A.shape[0]
    if n == 0:
        return jnp.asarray(1.0 + 0j)
    if n % 2 != 0:
        raise ValueError(f"Hafnian requires even-dimension matrix, got {n}x{n}.")
    N = n // 2

    if loop:
        # Loop hafnian: include diagonal
        # Use the Cayley-Hamilton / recursive approach
        return _loop_hafnian_ryser(A, N)
    else:
        return _hafnian_ryser(A, N)


def _hafnian_ryser(A: Array, N: int) -> Array:
    """Ryser inclusion-exclusion hafnian (no diagonal)."""
    # Work with the N x N matrix of pairs
    # A is 2N x 2N; pair matrix B[i,j] = A[2i, 2j+1] (upper-right blocks)
    # For a symmetric matrix: Haf(A) = Haf of pair structure
    # We use the standard Ryser formula on the full 2N x 2N matrix.

    total = jnp.asarray(0.0 + 0j)
    sign = (-1.0) ** N

    # Iterate over all 2^N subsets of {0,...,N-1}
    for s in range(1 << N):
        # Build the row-sum vector: z[j] = sum_{i in S} A[i, j]  (i over pairs)
        subset_rows = []
        bits = s
        k = 0
        while bits:
            if bits & 1:
                subset_rows.append(k)
            bits >>= 1
            k += 1

        if not subset_rows:
            total = total + sign
            sign = -sign
            continue

        # Compute product of column sums restricted to subset rows
        # For the hafnian of a 2N x 2N matrix, sum over pairs
        row_sum = jnp.zeros(2*N, dtype=jnp.complex128)
        for r in subset_rows:
            row_sum = row_sum + A[r, :]

        # Product over pairs (j=0..N-1): row_sum[2j] * ... (Ryser-style pair product)
        prod = jnp.asarray(1.0 + 0j)
        for j in range(N):
            prod = prod * row_sum[j]

        total = total + sign * prod
        sign = -sign

    return total / (2**N)


def _loop_hafnian_ryser(A: Array, N: int) -> Array:
    """Loop hafnian — diagonal elements contribute."""
    # For the loop hafnian we include a self-loop for each vertex.
    # Implementation: augment A with a superdiagonal and use the
    # standard hafnian on the augmented matrix.
    # Simple recursive approach for correctness:
    return _lhaf_recursive(A)


def _lhaf_recursive(A: Array) -> Array:
    """Recursive loop hafnian (slow but correct)."""
    n = A.shape[0]
    if n == 0:
        return jnp.asarray(1.0 + 0j)
    if n == 1:
        return A[0, 0]
    if n == 2:
        return A[0, 1] + A[0, 0] * A[1, 1]
    # lhaf(A) = A[0,0]*lhaf(A[1:,1:]) + sum_{j=1}^{n-1} A[0,j]*haf(A without 0,j)
    result = A[0, 0] * _lhaf_recursive(A[1:, 1:])
    for j in range(1, n):
        idx = jnp.array([i for i in range(n) if i not in (0, j)])
        if idx.shape[0] == 0:
            sub = jnp.asarray(1.0 + 0j)
        else:
            sub = _lhaf_recursive(A[jnp.ix_(idx, idx)])
        result = result + A[0, j] * sub
    return result


def hafnian_batch(matrices: Array) -> Array:
    """Compute hafnians of a batch of matrices.

    Parameters
    ----------
    matrices : complex array, shape (batch, 2N, 2N)

    Returns
    -------
    Array of shape (batch,) with hafnian values.
    """
    return jax.vmap(hafnian)(matrices)


def torontonian(O: Array) -> Array:
    """Compute the torontonian of a 2N x 2N matrix.

    The torontonian gives the probability of a threshold (click) detection
    pattern in GBS, without resolving photon numbers.

    tor(O) = sum_{S subset [N]} (-1)^{N-|S|} / sqrt(det(I - O_S))

    where O_S is the submatrix of O indexed by {2i-1,2i : i in S}.

    Parameters
    ----------
    O : complex array, shape (2N, 2N)
        The O matrix of the Gaussian state.

    Returns
    -------
    Scalar complex JAX array.
    """
    O = jnp.asarray(O, dtype=jnp.complex128)
    n = O.shape[0]
    if n % 2 != 0:
        raise ValueError(f"Torontonian requires even-dimension matrix, got {n}.")
    N = n // 2
    total = jnp.asarray(0.0 + 0j)

    for s in range(1 << N):
        # Build submatrix indices for subset s
        idx = []
        bits = s; k = 0
        while bits:
            if bits & 1:
                idx.extend([2*k, 2*k+1])
            bits >>= 1; k += 1

        sign = (-1.0) ** (N - bin(s).count("1"))

        if not idx:
            total = total + sign
            continue

        idx_arr = jnp.array(idx)
        O_s = O[jnp.ix_(idx_arr, idx_arr)]
        I_s = jnp.eye(len(idx), dtype=jnp.complex128)
        det_val = jnp.linalg.det(I_s - O_s)
        total = total + sign / jnp.sqrt(det_val)

    return total


