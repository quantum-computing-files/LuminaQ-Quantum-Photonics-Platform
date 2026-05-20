"""lumq.backends - simulators and hardware adapters v0.1.0"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4
import jax, jax.numpy as jnp

__version__ = "0.1.0"
__all__ = ["Device","DeviceCapabilities","JobResult","BackendError",
           "GaussianSimulator","GaussianSimulatorConfig",
           "FockSimulator","FockSimulatorConfig"]

@dataclass
class JobResult:
    job_id: str = field(default_factory=lambda: str(uuid4())[:8])
    backend: str = "unknown"
    shots: int = 1
    samples: Any = None
    state: Any = None
    expectation_values: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    def __repr__(self):
        return f"JobResult(backend='{self.backend}', shots={self.shots}, ev={list(self.expectation_values)})"

@dataclass
class DeviceCapabilities:
    name: str; n_modes: int
    supports_gaussian: bool = True
    supports_fock: bool = False
    supports_non_gaussian: bool = False
    native_gates: list = field(default_factory=lambda:["Beamsplitter","PhaseShifter","Squeezer","Displacer"])
    is_hardware: bool = False

class BackendError(RuntimeError): pass

class Device(ABC):
    @property
    @abstractmethod
    def capabilities(self): ...
    @abstractmethod
    def run_circuit(self, circuit, shots=1): ...
    def run(self, circuit, shots=1):
        if circuit.n_modes > self.capabilities.n_modes:
            raise BackendError(f"Circuit needs {circuit.n_modes} modes, device has {self.capabilities.n_modes}.")
        return self.run_circuit(circuit, shots=shots)

@dataclass
class GaussianSimulatorConfig:
    n_modes: int = 8; hbar: float = 2.0; seed: int = 0
    track_state: bool = True
    compute_ev: list = field(default_factory=lambda:["mean_photon"])

class GaussianSimulator(Device):
    def __init__(self, config=None):
        self.config = config or GaussianSimulatorConfig()
        self._key = jax.random.PRNGKey(self.config.seed)

    @property
    def capabilities(self):
        return DeviceCapabilities(name="GaussianSimulator", n_modes=self.config.n_modes,
            native_gates=["Beamsplitter","PhaseShifter","Squeezer","TwoModeSqueeze","Displacer","Interferometer"])

    def run_circuit(self, circuit, shots=1):
        from lumq.photonics.gates._symplectic import (
            embed_single, embed_two, S_phase_shifter, S_squeezer,
            S_beamsplitter, S_two_mode_squeeze, S_displacer_vec)
        N = circuit.n_modes; h = self.config.hbar
        mu = jnp.zeros(2*N); cov = (h/2)*jnp.eye(2*N)
        for op in circuit.ops:
            nm, md, p = op.name, op.modes, op.params
            if nm=="PhaseShifter":
                S = embed_single(S_phase_shifter(p["phi"]), md[0], N)
                mu,cov = S@mu, S@cov@S.T
            elif nm=="Squeezer":
                S = embed_single(S_squeezer(p["r"],p.get("phi",0.)), md[0], N)
                mu,cov = S@mu, S@cov@S.T
            elif nm=="Displacer":
                d = jnp.zeros(2*N).at[2*md[0]:2*md[0]+2].set(S_displacer_vec(p["alpha"],hbar=h))
                mu = mu+d
            elif nm=="Beamsplitter":
                S = embed_two(S_beamsplitter(p["theta"],p.get("phi",0.)), (md[0],md[1]), N)
                mu,cov = S@mu, S@cov@S.T
            elif nm=="TwoModeSqueeze":
                S = embed_two(S_two_mode_squeeze(p["r"],p.get("phi",0.)), (md[0],md[1]), N)
                mu,cov = S@mu, S@cov@S.T
            elif nm=="Interferometer":
                U = jnp.asarray(p["U"], dtype=jnp.complex128)
                Re,Im = jnp.real(U), jnp.imag(U); Nm = U.shape[0]
                S = jnp.zeros((2*N,2*N))
                for i in range(Nm):
                    for j in range(Nm):
                        S=S.at[2*i,2*j].set(Re[i,j]).at[2*i,2*j+1].set(-Im[i,j])
                        S=S.at[2*i+1,2*j].set(Im[i,j]).at[2*i+1,2*j+1].set(Re[i,j])
                mu,cov = S@mu, S@cov@S.T
            else:
                raise ValueError(f"GaussianSimulator: unknown gate '{nm}'. Use FockSimulator for non-Gaussian gates.")
        samples = None
        if circuit.measurements:
            rows = []
            for mop in circuit.measurements:
                mode=mop.modes[0]; phi=mop.params.get("phi",0.); i=2*mode
                c = jnp.array([jnp.cos(phi),jnp.sin(phi)])
                mu_m = jnp.dot(c,mu[i:i+2]); sig2=jnp.dot(c,cov[i:i+2,i:i+2]@c)
                self._key,sk = jax.random.split(self._key)
                rows.append(mu_m+jnp.sqrt(sig2)*jax.random.normal(sk,shape=(shots,)))
            samples = jnp.stack(rows,axis=1)
        state = None
        if self.config.track_state:
            from lumq.photonics.states import GaussianState
            state = GaussianState(mu=mu, cov=cov, hbar=h)
        ev = {}
        if "mean_photon" in self.config.compute_ev:
            ev["mean_photon"] = jnp.array([(cov[2*k,2*k]+cov[2*k+1,2*k+1]+mu[2*k]**2+mu[2*k+1]**2)/h-0.5 for k in range(N)])
        if "x_mean" in self.config.compute_ev: ev["x_mean"] = mu[0::2]
        if "p_mean" in self.config.compute_ev: ev["p_mean"] = mu[1::2]
        return JobResult(backend="GaussianSimulator",shots=shots,samples=samples,
                         state=state,expectation_values=ev,metadata={"n_modes":N,"n_ops":len(circuit.ops)})

@dataclass
class FockSimulatorConfig:
    n_modes: int = 4; cutoff: int = 10; hbar: float = 2.0; seed: int = 0

class FockSimulator(Device):
    def __init__(self, config=None):
        self.config = config or FockSimulatorConfig()
        self._key = jax.random.PRNGKey(self.config.seed)

    @property
    def capabilities(self):
        return DeviceCapabilities(name="FockSimulator", n_modes=self.config.n_modes,
            supports_fock=True, supports_non_gaussian=True,
            native_gates=["Beamsplitter","PhaseShifter","Squeezer","TwoModeSqueeze","Displacer","KerrGate","CubicPhase"])

    def run_circuit(self, circuit, shots=1):
        from lumq.photonics.states import FockState
        from lumq.photonics.gates import (Beamsplitter, Displacer, KerrGate,
                                           PhaseShifter, Squeezer, TwoModeSqueeze)
        N = circuit.n_modes; cut = self.config.cutoff
        state = FockState.vacuum(n_modes=N, cutoff=cut)
        GATE = {"PhaseShifter": lambda p: PhaseShifter(p["phi"]),
                "Squeezer":     lambda p: Squeezer(p["r"],p.get("phi",0.)),
                "Displacer":    lambda p: Displacer(p["alpha"]),
                "Beamsplitter": lambda p: Beamsplitter(p["theta"],p.get("phi",0.)),
                "TwoModeSqueeze":lambda p: TwoModeSqueeze(p["r"],p.get("phi",0.)),
                "KerrGate":     lambda p: KerrGate(p["kappa"])}
        for op in circuit.ops:
            if op.name not in GATE: raise ValueError(f"FockSimulator: gate '{op.name}' not supported.")
            g = GATE[op.name](op.params)
            state = g.apply(state,mode=op.modes[0]) if len(op.modes)==1 else g.apply(state,modes=tuple(op.modes))
        samples = None
        if circuit.measurements:
            rows = []
            for mop in circuit.measurements:
                mode=mop.modes[0]
                axes=tuple(i for i in range(N) if i!=mode)
                probs=jnp.sum(jnp.abs(state.data)**2,axis=axes)
                self._key,sk=jax.random.split(self._key)
                rows.append(jax.random.choice(sk,jnp.arange(cut),shape=(shots,),p=probs))
            samples = jnp.stack(rows,axis=1)
        ns = jnp.arange(cut,dtype=jnp.float64)
        n_bar = []
        for k in range(N):
            axes=tuple(i for i in range(N) if i!=k)
            n_bar.append(jnp.dot(jnp.sum(jnp.abs(state.data)**2,axis=axes),ns))
        return JobResult(backend="FockSimulator",shots=shots,samples=samples,state=state,
                         expectation_values={"mean_photon":jnp.array(n_bar)},
                         metadata={"n_modes":N,"cutoff":cut})
