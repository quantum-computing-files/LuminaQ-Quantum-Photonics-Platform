"""Quantum optical measurements."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import jax, jax.numpy as jnp
from lumq.photonics._types import ModeIndex, _check_mode

__all__ = ["HomodyneMeasurement","HeterodyneMeasurement",
           "PhotonNumberMeasurement","FockMeasurement","MeasurementResult"]

@dataclass
class MeasurementResult:
    outcome: object
    post_state: object = None
    log_prob: object = None
    def __post_init__(self):
        if self.log_prob is None:
            self.log_prob = jnp.asarray(0.0)
    @property
    def prob(self): return jnp.exp(self.log_prob)

class HomodyneMeasurement:
    """Measure x_phi = x cos(phi) + p sin(phi)."""
    def __init__(self, phi=0.0): self.phi = float(phi)
    def apply(self, state, mode=0, key=None, outcome=None):
        from lumq.photonics.states import GaussianState
        _check_mode(mode, state.n_modes)
        N = state.n_modes; i = 2*mode
        c = jnp.array([jnp.cos(self.phi), jnp.sin(self.phi)])
        mu_m = jnp.dot(c, state.mu[i:i+2])
        sig2 = jnp.dot(c, state.cov[i:i+2,i:i+2]@c)
        if outcome is None:
            key = key or jax.random.PRNGKey(0)
            s = mu_m + jnp.sqrt(sig2)*jax.random.normal(key)
        else:
            s = jnp.asarray(float(outcome))
        lp = -0.5*jnp.log(2*jnp.pi*sig2) - 0.5*(s-mu_m)**2/sig2
        if N == 1:
            return MeasurementResult(outcome=s, log_prob=lp)
        rem = jnp.array([j for j in range(2*N) if j not in [int(i), int(i+1)]])
        meas_idx = jnp.array([i, i+1])
        mu_A = state.mu[rem]
        V_AA = state.cov[jnp.ix_(rem, rem)]
        V_AB = state.cov[jnp.ix_(rem, meas_idx)]
        mu_post = mu_A + V_AB@c*(s-mu_m)/sig2
        cov_post = V_AA - (V_AB@jnp.outer(c,c)@V_AB.T)/sig2
        post = GaussianState(mu=mu_post, cov=cov_post, hbar=state.hbar)
        return MeasurementResult(outcome=s, post_state=post, log_prob=lp)

class HeterodyneMeasurement:
    """Joint (x,p) measurement; projects onto coherent states."""
    def apply(self, state, mode=0, key=None, outcome=None):
        from lumq.photonics.states import GaussianState
        _check_mode(mode, state.n_modes)
        i = 2*mode
        V_eff = state.cov[i:i+2,i:i+2] + (state.hbar/2)*jnp.eye(2)
        mu_m = state.mu[i:i+2]
        if outcome is None:
            key = key or jax.random.PRNGKey(0)
            k1, k2 = jax.random.split(key)
            L = jnp.linalg.cholesky(V_eff)
            z = jnp.array([jax.random.normal(k1), jax.random.normal(k2)])
            xp = mu_m + L@z
        else:
            xp = jnp.array([jnp.real(outcome)*2, jnp.imag(outcome)*2])
        alpha = (xp[0]+1j*xp[1])/2.0
        diff = xp - mu_m
        lp = -0.5*jnp.log(jnp.linalg.det(2*jnp.pi*V_eff)) - 0.5*diff@jnp.linalg.solve(V_eff,diff)
        N = state.n_modes
        if N == 1:
            return MeasurementResult(outcome=alpha, log_prob=lp)
        rem = jnp.array([j for j in range(2*N) if j not in [int(i),int(i+1)]])
        meas_idx = jnp.array([i,i+1])
        V_AB = state.cov[jnp.ix_(rem, meas_idx)]
        V_AA = state.cov[jnp.ix_(rem, rem)]
        mu_post = state.mu[rem] + V_AB@jnp.linalg.solve(V_eff, xp-mu_m)
        cov_post = V_AA - V_AB@jnp.linalg.solve(V_eff, V_AB.T)
        post = GaussianState(mu=mu_post, cov=cov_post, hbar=state.hbar)
        return MeasurementResult(outcome=alpha, post_state=post, log_prob=lp)

class PhotonNumberMeasurement:
    def __init__(self, max_photons=10): self.max_photons = max_photons
    def apply(self, state, mode=0, key=None, outcome=None):
        from lumq.photonics.states import FockState, GaussianState
        if isinstance(state, FockState):
            axes = tuple(i for i in range(state.n_modes) if i != mode)
            probs = jnp.sum(jnp.abs(state.data)**2, axis=axes)
            key = key or jax.random.PRNGKey(0)
            n = int(outcome) if outcome is not None else int(jax.random.choice(key, jnp.arange(state.cutoff), p=probs))
            return MeasurementResult(outcome=jnp.asarray(n), log_prob=jnp.log(probs[n]+1e-300))
        if isinstance(state, GaussianState):
            n_bar = float(state.mean_photon_number(mode))
            ns = jnp.arange(self.max_photons+1, dtype=jnp.float64)
            probs = n_bar**ns / (n_bar+1)**(ns+1)
            probs = probs / probs.sum()
            key = key or jax.random.PRNGKey(0)
            n = int(outcome) if outcome is not None else int(jax.random.choice(key, jnp.arange(self.max_photons+1), p=probs))
            return MeasurementResult(outcome=jnp.asarray(n), log_prob=jnp.log(probs[n]+1e-300))
        raise TypeError(f"Unsupported state type {type(state)}")

class FockMeasurement:
    def __init__(self, target): self.target = list(target)
    def apply(self, state, mode=None, modes=None):
        from lumq.photonics.states import FockState
        if not isinstance(state, FockState): raise TypeError("FockMeasurement requires FockState.")
        amp = state.data[tuple(self.target)]
        prob = jnp.abs(amp)**2
        return MeasurementResult(outcome=jnp.array(self.target), log_prob=jnp.log(prob+1e-300))


