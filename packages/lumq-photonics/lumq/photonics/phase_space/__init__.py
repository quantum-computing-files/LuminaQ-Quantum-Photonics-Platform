"""Phase-space representations: Wigner, Husimi Q, marginals."""
from __future__ import annotations
import jax.numpy as jnp
from lumq.photonics._types import _check_mode

__all__ = ["wigner","husimi_q","marginal_x","marginal_p"]

def wigner(state, mode=0, xvec=None, pvec=None, n_points=100, x_range=5.0):
    from lumq.photonics.states import GaussianState
    xvec = jnp.linspace(-x_range, x_range, n_points) if xvec is None else jnp.asarray(xvec)
    pvec = jnp.linspace(-x_range, x_range, n_points) if pvec is None else jnp.asarray(pvec)
    X, P = jnp.meshgrid(xvec, pvec, indexing="ij")
    if isinstance(state, GaussianState):
        _check_mode(mode, state.n_modes)
        i = 2*mode; mu = state.mu[i:i+2]; V = state.cov[i:i+2,i:i+2]
        iV = jnp.linalg.inv(V)
        norm = 1.0/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(V)))
        dx, dp = X-mu[0], P-mu[1]
        qf = iV[0,0]*dx**2 + 2*iV[0,1]*dx*dp + iV[1,1]*dp**2
        return X, P, norm*jnp.exp(-0.5*qf)
    raise TypeError(f"wigner() does not support {type(state)} yet.")

def husimi_q(state, mode=0, xvec=None, pvec=None, n_points=100, x_range=5.0):
    from lumq.photonics.states import GaussianState
    xvec = jnp.linspace(-x_range, x_range, n_points) if xvec is None else jnp.asarray(xvec)
    pvec = jnp.linspace(-x_range, x_range, n_points) if pvec is None else jnp.asarray(pvec)
    X, P = jnp.meshgrid(xvec, pvec, indexing="ij")
    if isinstance(state, GaussianState):
        _check_mode(mode, state.n_modes)
        i = 2*mode; mu = state.mu[i:i+2]
        V = state.cov[i:i+2,i:i+2] + (state.hbar/2)*jnp.eye(2)
        iV = jnp.linalg.inv(V)
        dx, dp = X-mu[0], P-mu[1]
        qf = iV[0,0]*dx**2 + 2*iV[0,1]*dx*dp + iV[1,1]*dp**2
        return X, P, jnp.exp(-0.5*qf)/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(V)))
    raise TypeError(f"husimi_q() does not support {type(state)} yet.")

def marginal_x(state, mode=0, xvec=None, n_points=200, x_range=6.0):
    from lumq.photonics.states import GaussianState
    xvec = jnp.linspace(-x_range, x_range, n_points) if xvec is None else jnp.asarray(xvec)
    if isinstance(state, GaussianState):
        _check_mode(mode, state.n_modes)
        i = 2*mode; mu_x = state.mu[i]; s2 = state.cov[i,i]
        return xvec, jnp.exp(-0.5*(xvec-mu_x)**2/s2)/jnp.sqrt(2*jnp.pi*s2)
    pvec = jnp.linspace(-x_range, x_range, n_points)
    _, _, W = wigner(state, mode, xvec, pvec)
    return xvec, jnp.sum(W, axis=1)*(pvec[1]-pvec[0])

def marginal_p(state, mode=0, pvec=None, n_points=200, p_range=6.0):
    from lumq.photonics.states import GaussianState
    pvec = jnp.linspace(-p_range, p_range, n_points) if pvec is None else jnp.asarray(pvec)
    if isinstance(state, GaussianState):
        _check_mode(mode, state.n_modes)
        i = 2*mode+1; mu_p = state.mu[i]; s2 = state.cov[i,i]
        return pvec, jnp.exp(-0.5*(pvec-mu_p)**2/s2)/jnp.sqrt(2*jnp.pi*s2)
    xvec = jnp.linspace(-p_range, p_range, n_points)
    _, _, W = wigner(state, mode, xvec, pvec)
    return pvec, jnp.sum(W, axis=0)*(xvec[1]-xvec[0])
