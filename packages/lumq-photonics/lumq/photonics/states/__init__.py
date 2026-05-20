"""GaussianState, FockState, StateVector."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from lumq.photonics._types import HBAR, ArrayLike, ModeIndex, _ensure_jax, _ensure_complex_jax

__all__ = ["GaussianState", "FockState", "StateVector"]


@dataclass
class GaussianState:
    """N-mode Gaussian state. Convention: hbar=2, order (x0,p0,x1,p1,...)."""
    mu: object
    cov: object
    hbar: float = HBAR

    def __post_init__(self):
        self.mu = _ensure_jax(self.mu)
        self.cov = _ensure_jax(self.cov)
        n = self.mu.shape[0]
        if n % 2 != 0:
            raise ValueError(f"mu must have even length, got {n}.")
        if self.cov.shape != (n, n):
            raise ValueError(f"cov shape {self.cov.shape} incompatible with mu length {n}.")

    @classmethod
    def vacuum(cls, n_modes, hbar=HBAR):
        d = 2 * n_modes
        return cls(mu=jnp.zeros(d), cov=(hbar/2)*jnp.eye(d), hbar=hbar)

    @classmethod
    def coherent(cls, alphas, hbar=HBAR):
        n = len(alphas); d = 2*n
        mu = jnp.zeros(d, dtype=jnp.float64)
        sc = jnp.sqrt(2.0*hbar)
        for k, a in enumerate(alphas):
            a = jnp.asarray(a, dtype=jnp.complex128)
            mu = mu.at[2*k].set(sc*jnp.real(a))
            mu = mu.at[2*k+1].set(sc*jnp.imag(a))
        return cls(mu=mu, cov=(hbar/2)*jnp.eye(d), hbar=hbar)

    @classmethod
    def thermal(cls, n_bar, n_modes=1, mode=0, hbar=HBAR):
        d = 2*n_modes
        cov = (hbar/2)*jnp.eye(d)
        v = (2*n_bar+1)*(hbar/2)
        i = 2*mode
        cov = cov.at[i,i].set(v).at[i+1,i+1].set(v)
        return cls(mu=jnp.zeros(d), cov=cov, hbar=hbar)

    @classmethod
    def squeezed_vacuum(cls, r, phi=0.0, n_modes=1, mode=0, hbar=HBAR):
        from lumq.photonics.gates import Squeezer
        return Squeezer(r=r, phi=phi).apply(cls.vacuum(n_modes, hbar=hbar), mode=mode)

    @classmethod
    def two_mode_squeezed_vacuum(cls, r, phi=0.0, hbar=HBAR):
        from lumq.photonics.gates import TwoModeSqueeze
        return TwoModeSqueeze(r=r, phi=phi).apply(cls.vacuum(2, hbar=hbar), modes=(0,1))

    @property
    def n_modes(self): return self.mu.shape[0] // 2

    @property
    def purity(self):
        return (self.hbar/2)**self.n_modes / jnp.sqrt(jnp.linalg.det(self.cov))

    def mean_photon_number(self, mode):
        i = 2*mode
        return (self.cov[i,i]+self.cov[i+1,i+1]+self.mu[i]**2+self.mu[i+1]**2)/self.hbar - 0.5

    def reduced(self, mode):
        i = 2*mode
        return GaussianState(mu=self.mu[i:i+2], cov=self.cov[i:i+2,i:i+2], hbar=self.hbar)

    def tensor(self, other):
        n1, n2 = 2*self.n_modes, 2*other.n_modes
        return GaussianState(
            mu=jnp.concatenate([self.mu, other.mu]),
            cov=jnp.block([[self.cov, jnp.zeros((n1,n2))],[jnp.zeros((n2,n1)), other.cov]]),
            hbar=self.hbar)

    def __repr__(self):
        return f"GaussianState(n_modes={self.n_modes}, purity={float(self.purity):.4f}, hbar={self.hbar})"


@dataclass
class FockState:
    """N-mode Fock-basis state. data shape = (cutoff,)*n_modes."""
    data: object
    cutoff: int

    def __post_init__(self):
        self.data = _ensure_complex_jax(self.data)

    @property
    def n_modes(self): return len(self.data.shape)

    @property
    def norm(self): return jnp.sqrt(jnp.sum(jnp.abs(self.data)**2))

    def normalise(self): return FockState(data=self.data/self.norm, cutoff=self.cutoff)

    @classmethod
    def vacuum(cls, n_modes, cutoff=10):
        data = jnp.zeros((cutoff,)*n_modes, dtype=jnp.complex128)
        data = data.at[(0,)*n_modes].set(1.0+0j)
        return cls(data=data, cutoff=cutoff)

    @classmethod
    def fock(cls, ns, cutoff=10):
        if any(n >= cutoff for n in ns):
            raise ValueError(f"Fock indices {ns} exceed cutoff {cutoff}.")
        data = jnp.zeros((cutoff,)*len(ns), dtype=jnp.complex128)
        data = data.at[tuple(ns)].set(1.0+0j)
        return cls(data=data, cutoff=cutoff)

    def __repr__(self):
        return f"FockState(n_modes={self.n_modes}, cutoff={self.cutoff}, norm={float(self.norm):.4f})"


class StateVector(FockState):
    """Single-mode pure Fock state."""
    @classmethod
    def number(cls, n, cutoff=10):
        data = jnp.zeros(cutoff, dtype=jnp.complex128).at[n].set(1.0+0j)
        return cls(data=data, cutoff=cutoff)




