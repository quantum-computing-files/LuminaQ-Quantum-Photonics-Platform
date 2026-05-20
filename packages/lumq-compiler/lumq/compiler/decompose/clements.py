"""Clements rectangular mesh decomposition.
Reference: Clements et al., Optica 3, 1460 (2016).
"""
from __future__ import annotations
import cmath, math
from typing import Optional
import numpy as np

__all__ = ["clements_decompose","ClementsResult","random_unitary"]

class ClementsResult:
    def __init__(self, beamsplitters, phases, n_modes, reconstruction_error=0.0):
        self.beamsplitters = beamsplitters
        self.phases = phases
        self.n_modes = n_modes
        self.reconstruction_error = reconstruction_error
    @property
    def n_beamsplitters(self): return len(self.beamsplitters)
    @property
    def circuit(self):
        from lumq.compiler.ir import CircuitMetadata, PhotonicCircuit
        c = PhotonicCircuit(n_modes=self.n_modes,
            metadata=CircuitMetadata(name=f"Clements-{self.n_modes}x{self.n_modes}",created_by="lumq-compiler"))
        for m0,m1,theta,phi in self.beamsplitters:
            if abs(phi) > 1e-12: c.ps(m0, phi)
            c.bs(m0, m1, theta)
        for mode,phi in self.phases:
            if abs(phi) > 1e-12: c.ps(mode, phi)
        return c
    def __repr__(self):
        return f"ClementsResult(n_modes={self.n_modes}, BSs={self.n_beamsplitters}, error={self.reconstruction_error:.2e})"

def clements_decompose(U, tol=1e-12):
    U = np.array(U, dtype=complex); N = U.shape[0]
    if U.ndim != 2 or U.shape[1] != N: raise ValueError(f"U must be square, got {U.shape}.")
    err = np.linalg.norm(U.conj().T @ U - np.eye(N))
    if err > 1e-6: raise ValueError(f"U is not unitary: ||U†U-I||={err:.4e}.")
    V = U.copy(); T_list = []
    for col in range(N-1):
        if col % 2 == 0:
            for row in range(N-1, col, -1):
                t,phi = _null_right(V, row, col, row-1, tol)
                T_list.append((row-1, row, t, phi, "right"))
        else:
            for row in range(col+1, N):
                t,phi = _null_left(V, col, row, col, tol)
                T_list.append((col, row, t, phi, "left"))
    diag_phases = [(k, cmath.phase(V[k,k])) for k in range(N)]
    bs_list = [(m0,m1,t,phi) for m0,m1,t,phi,_ in reversed(T_list)]
    U_rec = _reconstruct(bs_list, diag_phases, N)
    return ClementsResult(bs_list, diag_phases, N, float(np.linalg.norm(U-U_rec)))

def _null_right(V, row, col, m0, tol):
    a,b = V[row,col], V[m0,col]
    if abs(a) < tol: return 0.0, 0.0
    theta = math.atan2(abs(a),abs(b)) if abs(b)>tol else math.pi/2
    phi = cmath.phase(a)-cmath.phase(b) if abs(b)>tol else cmath.phase(a)
    ct,st,ep = math.cos(theta), math.sin(theta), cmath.exp(1j*phi)
    c0,c1 = V[:,m0].copy(), V[:,row].copy()
    V[:,m0] =  ct*c0 + st*ep.conjugate()*c1
    V[:,row] = -st*ep*c0 + ct*c1
    return theta, phi

def _null_left(V, col, row, m0, tol):
    a,b = V[col,row], V[col,m0]
    if abs(a) < tol: return 0.0, 0.0
    theta = math.atan2(abs(a),abs(b)) if abs(b)>tol else math.pi/2
    phi = cmath.phase(a)-cmath.phase(b)+math.pi if abs(b)>tol else cmath.phase(a)+math.pi
    ct,st,ep = math.cos(theta), math.sin(theta), cmath.exp(1j*phi)
    r0,r1 = V[m0,:].copy(), V[row,:].copy()
    V[m0,:]  =  ct*r0 - st*ep*r1
    V[row,:] =  st*ep.conjugate()*r0 + ct*r1
    return theta, phi

def _bs2(theta, phi):
    ct,st = math.cos(theta), math.sin(theta); ep = cmath.exp(1j*phi)
    return np.array([[ct,-st*ep.conjugate()],[st*ep,ct]])

def _reconstruct(bs_list, diag_phases, N):
    U = np.eye(N, dtype=complex)
    for m0,m1,theta,phi in bs_list:
        T = np.eye(N, dtype=complex); T2 = _bs2(theta,phi)
        for i,ri in enumerate([m0,m1]):
            for j,cj in enumerate([m0,m1]): T[ri,cj] = T2[i,j]
        U = T@U
    D = np.diag([cmath.exp(1j*phi) for _,phi in diag_phases])
    return D@U

def random_unitary(N, seed=None):
    rng = np.random.default_rng(seed)
    Z = (rng.standard_normal((N,N)) + 1j*rng.standard_normal((N,N))) / math.sqrt(2)
    Q,R = np.linalg.qr(Z)
    return Q * (np.diag(R)/np.abs(np.diag(R)))

