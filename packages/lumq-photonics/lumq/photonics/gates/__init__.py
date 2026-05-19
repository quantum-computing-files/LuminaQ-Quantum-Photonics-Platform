"""Photonic gate library."""
from __future__ import annotations
import jax.numpy as jnp
from lumq.photonics._types import HBAR, ModeIndex, ModePair, _check_mode, _check_modes
from lumq.photonics.gates._symplectic import (
    embed_single, embed_two,
    S_phase_shifter, S_squeezer, S_beamsplitter,
    S_two_mode_squeeze, S_displacer_vec)

__all__ = ["PhaseShifter","Squeezer","Displacer","Beamsplitter",
           "TwoModeSqueeze","Interferometer","KerrGate","CubicPhase"]

def _gs():
    from lumq.photonics.states import GaussianState, FockState
    return GaussianState, FockState

def _symp(state, S, d=None):
    GS, _ = _gs()
    mu2 = S @ state.mu + (d if d is not None else jnp.zeros(2*state.n_modes))
    return GS(mu=mu2, cov=S @ state.cov @ S.T, hbar=state.hbar)

class PhaseShifter:
    def __init__(self, phi): self.phi = float(phi)
    def apply(self, state, mode=0, modes=None):
        _check_mode(mode, state.n_modes)
        return _symp(state, embed_single(S_phase_shifter(self.phi), mode, state.n_modes))
    def __repr__(self): return f"PhaseShifter(phi={self.phi:.4f})"

class Squeezer:
    def __init__(self, r, phi=0.0): self.r=float(r); self.phi=float(phi)
    def apply(self, state, mode=0, modes=None):
        _check_mode(mode, state.n_modes)
        return _symp(state, embed_single(S_squeezer(self.r, self.phi), mode, state.n_modes))
    def __repr__(self): return f"Squeezer(r={self.r}, phi={self.phi})"

class Displacer:
    def __init__(self, alpha): self.alpha = complex(alpha)
    def apply(self, state, mode=0, modes=None):
        GS, _ = _gs()
        _check_mode(mode, state.n_modes)
        d = jnp.zeros(2*state.n_modes).at[2*mode:2*mode+2].set(
            S_displacer_vec(self.alpha, hbar=state.hbar))
        return GS(mu=state.mu+d, cov=state.cov, hbar=state.hbar)
    def __repr__(self): return f"Displacer(alpha={self.alpha})"

class Beamsplitter:
    def __init__(self, theta, phi=0.0): self.theta=float(theta); self.phi=float(phi)
    @property
    def transmissivity(self): return float(jnp.cos(self.theta)**2)
    def apply(self, state, mode=None, modes=(0,1)):
        _check_modes(modes, state.n_modes)
        return _symp(state, embed_two(S_beamsplitter(self.theta, self.phi), modes, state.n_modes))
    def __repr__(self): return f"Beamsplitter(theta={self.theta:.4f}, phi={self.phi:.4f})"

class TwoModeSqueeze:
    def __init__(self, r, phi=0.0): self.r=float(r); self.phi=float(phi)
    def apply(self, state, mode=None, modes=(0,1)):
        _check_modes(modes, state.n_modes)
        return _symp(state, embed_two(S_two_mode_squeeze(self.r, self.phi), modes, state.n_modes))
    def __repr__(self): return f"TwoModeSqueeze(r={self.r}, phi={self.phi})"

class Interferometer:
    def __init__(self, U):
        self.U = jnp.asarray(U, dtype=jnp.complex128)
        if self.U.ndim != 2 or self.U.shape[0] != self.U.shape[1]:
            raise ValueError("U must be square.")
    @property
    def n_modes(self): return int(self.U.shape[0])
    def _S(self):
        Re, Im = jnp.real(self.U), jnp.imag(self.U)
        N = self.n_modes; S = jnp.zeros((2*N, 2*N))
        for i in range(N):
            for j in range(N):
                S = S.at[2*i,2*j].set(Re[i,j]).at[2*i,2*j+1].set(-Im[i,j])
                S = S.at[2*i+1,2*j].set(Im[i,j]).at[2*i+1,2*j+1].set(Re[i,j])
        return S
    def apply(self, state, mode=None, modes=None):
        GS, _ = _gs()
        if state.n_modes != self.n_modes:
            raise ValueError(f"Interferometer is {self.n_modes}-mode, state has {state.n_modes}.")
        S = self._S()
        return GS(mu=S@state.mu, cov=S@state.cov@S.T, hbar=state.hbar)

class KerrGate:
    def __init__(self, kappa): self.kappa = float(kappa)
    def apply(self, state, mode=0, modes=None):
        _, FS = _gs()
        if not isinstance(state, FS): raise TypeError("KerrGate requires FockState.")
        ns = jnp.arange(state.cutoff, dtype=jnp.float64)
        phases = jnp.exp(1j*self.kappa*ns**2)
        shp = [1]*state.n_modes; shp[mode] = state.cutoff
        return FS(data=state.data*phases.reshape(shp), cutoff=state.cutoff)

class CubicPhase:
    def __init__(self, gamma, cutoff=50, hbar=HBAR):
        self.gamma=float(gamma); self.cutoff=cutoff; self.hbar=hbar
    def _U(self):
        import scipy.linalg as spla, numpy as np
        ns = jnp.arange(self.cutoff, dtype=jnp.float64)
        sc = jnp.sqrt(self.hbar/2.0)
        a = jnp.diag(jnp.sqrt(ns[1:]), k=-1)
        x = sc*(a+a.T); x3 = x@x@x
        return jnp.array(spla.expm(1j*self.gamma/(3*self.hbar)*np.array(x3)), dtype=jnp.complex128)
    def apply(self, state, mode=0, modes=None):
        _, FS = _gs()
        if not isinstance(state, FS): raise TypeError("CubicPhase requires FockState.")
        return FS(data=self._U()@state.data, cutoff=state.cutoff)

