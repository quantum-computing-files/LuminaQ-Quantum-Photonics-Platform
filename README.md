# LuminaQ-Quantum-Photonics-Platform

# LuminaQ Platform

**An open-source photonic quantum computing software stack.**

[![CI](https://github.com/nunofernandes-plight/LuminaQ_Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/nunofernandes-plight/LuminaQ_Platform/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quantum Computing Files](https://img.shields.io/badge/Substack-Quantum%20Computing%20Files-orange)](https://quantumcomputingfiles.substack.com)

---

## What is LuminaQ?

LuminaQ is a photonic quantum computing platform built from the ground up for **continuous-variable (CV) quantum optics**. It provides a complete software stack — from quantum state physics to a browser-based circuit interface — with a design philosophy borrowed from the best of PennyLane and Strawberry Fields, re-implemented on JAX for end-to-end differentiability.

The project is documented publicly on [Quantum Computing Files](https://quantumcomputingfiles.substack.com), a Substack publication covering photonic and quantum computing research since 2021.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LuminaQ UI  (TypeScript · React · Vite · Tailwind)     │  ← src/
│  Circuit canvas · Results viewer · Algorithm explorer   │
└────────────────────────┬────────────────────────────────┘
                         │ REST / WebSocket
┌────────────────────────▼────────────────────────────────┐
│  lumq-api  (FastAPI · Pydantic · WebSocket)             │
│  Job queue · Phase-space endpoints · Compiler API       │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
┌───────▼──────┐ ┌───────▼──────┐ ┌───────▼──────┐
│lumq-algorithms│ │lumq-compiler │ │lumq-backends │
│GBS · VQE-CV  │ │IR · Clements │ │Gaussian sim  │
│QML · Sensing │ │Passes · Rsrc │ │Fock sim      │
└───────┬───────┘ └───────┬──────┘ └───────┬──────┘
        │                 │                │
        └────────────┬────┴────────────────┘
                     │
        ┌────────────▼────────────────────────────────────┐
        │  lumq-photonics                                 │
        │  States · Gates · Measurements · Phase space    │
        └─────────────────────────────────────────────────┘
```

---

## Packages

### `lumq-photonics` — physics core
The foundation of the entire stack. All numerics use **JAX** — every gate, state, and measurement is differentiable via `jax.grad` and JIT-compilable.

| Module | Contents |
|---|---|
| `states` | `GaussianState` (covariance matrix, hbar=2), `FockState` (dense tensor), `StateVector` |
| `gates` | `Beamsplitter`, `Squeezer`, `PhaseShifter`, `TwoModeSqueeze`, `Displacer`, `Interferometer`, `KerrGate`, `CubicPhase` |
| `measurements` | `HomodyneMeasurement`, `HeterodyneMeasurement`, `PhotonNumberMeasurement`, `FockMeasurement` |
| `phase_space` | `wigner()`, `husimi_q()`, `marginal_x()`, `marginal_p()` |

### `lumq-compiler` — circuit IR and optimisation
A hardware-agnostic circuit representation with a pass pipeline and the **Clements rectangular mesh decomposer**.

| Component | Description |
|---|---|
| `PhotonicCircuit` | Fluent builder API, JSON serialisation, ASCII draw |
| `clements_decompose` | Any N×N unitary → optimal beamsplitter mesh. Verified error < 1e-9 |
| `PassPipeline` | `RemoveIdentityGates`, `MergePhaseShifters`, `DecomposeTwoMode` |
| `ResourceEstimator` | Gate counts, circuit depth, squeezing budget in dB |
| `Compiler` | One-call compile: `compiled, report = Compiler().compile(circuit)` |

### `lumq-backends` — simulators
| Backend | Description |
|---|---|
| `GaussianSimulator` | Exact covariance matrix engine. O(N²) per gate. Differentiable. |
| `FockSimulator` | Dense Fock tensor. Supports non-Gaussian gates (Kerr, CubicPhase). |

Both implement the `Device` abstract interface — algorithm code runs unchanged on any backend.

### `lumq-api` — REST + WebSocket gateway
FastAPI application bridging the Python stack to the TypeScript UI.

| Endpoint | Description |
|---|---|
| `POST /jobs/submit` | Compile + simulate a circuit asynchronously |
| `GET /jobs/{id}` | Poll job status and result |
| `POST /phase-space/wigner` | Wigner W(x,p) grid for heatmap rendering |
| `POST /phase-space/marginal` | P(x) and P(p) marginals |
| `POST /compiler/clements` | Decompose a unitary matrix |
| `POST /compiler/resource` | Resource estimate without running |
| `ws://host/ws` | Real-time job events (queued → running → completed) |
| `GET /docs` | Swagger UI — all endpoints documented and testable |

### `lumq-algorithms` — high-level algorithms
#### Gaussian Boson Sampling (GBS)
The first algorithm layer. Encodes graph problems into photonic circuits and samples from the output photon-number distribution.

```python
from lumq.algorithms.gbs import gbs_circuit_from_graph, GBSSampler, gbs_to_graph_features
import numpy as np

# Encode a graph
A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
circuit, r, U = gbs_circuit_from_graph(A, scale=0.4)

# Sample
result = GBSSampler(seed=42).sample(circuit, shots=1000)
print(result)   # GBSResult(n_modes=4, shots=1000, mean_total_photons=1.84, collision_rate=0.12)

# Graph features
features = gbs_to_graph_features(result, A)
print(f"Max clique found: {features['max_clique_size']}")
```

The hafnian — the computational kernel of GBS — is implemented in pure JAX (differentiable, O(N² · 2^N)):

```python
from lumq.algorithms.gbs import hafnian
import jax.numpy as jnp

A = jnp.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]], dtype=jnp.complex128)
print(hafnian(A))   # exact hafnian value
```

**Planned algorithms:** VQE-CV, continuous-variable QML, quantum sensing / Heisenberg-limited metrology.

---

## Quick start

### Prerequisites
- Python 3.10 or 3.11
- Node.js 18+ (for the UI)

### Install

```bash
git clone https://github.com/nunofernandes-plight/LuminaQ_Platform.git
cd LuminaQ_Platform

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install "jax[cpu]" numpy scipy matplotlib pydantic
pip install fastapi "uvicorn[standard]" websockets pydantic-settings python-multipart anyio
pip install pytest pytest-asyncio httpx

pip install -e packages/lumq-photonics
pip install -e packages/lumq-compiler
pip install -e packages/lumq-backends
pip install -e packages/lumq-api
pip install -e packages/lumq-algorithms
```

### Run the tests

```bash
pytest packages/ -v --tb=short
```

### Start the API server

```bash
uvicorn lumq.api.app:app --reload --port 8000
```

Swagger UI → http://localhost:8000/docs

### Start the full stack with Docker

```bash
docker-compose up --build
```

UI → http://localhost:5173  
API → http://localhost:8000

---

## Example: variational circuit with autodiff

Because everything is JAX-native, gradients flow through the entire stack with no extra instrumentation:

```python
import jax
import jax.numpy as jnp
from lumq.photonics.states import GaussianState
from lumq.photonics.gates import Squeezer, Beamsplitter

def mean_photon(params):
    r, theta = params[0], params[1]
    state = GaussianState.vacuum(n_modes=2)
    state = Squeezer(r=r).apply(state, mode=0)
    state = Beamsplitter(theta=theta).apply(state, modes=(0, 1))
    return float(state.mean_photon_number(0))

# Gradient of <n̂> w.r.t. squeezing and beamsplitter angle
grad = jax.grad(mean_photon)(jnp.array([1.0, 0.5]))
print(f"d<n>/dr = {grad[0]:.4f},  d<n>/dθ = {grad[1]:.4f}")
```

---

## Example: compile and simulate a GBS circuit

```python
from lumq.compiler import Compiler, PhotonicCircuit
from lumq.backends import GaussianSimulator

# Build a 4-mode GBS circuit
circ = PhotonicCircuit(n_modes=4)
circ.sq(0, r=1.0).sq(1, r=0.8).sq(2, r=1.0).sq(3, r=0.8)
circ.bs(0, 1, theta=0.785).bs(1, 2, theta=0.524).bs(2, 3, theta=0.785)
circ.meas_homodyne(0).meas_homodyne(1)

print(circ.draw())

# Compile (expand interferometers, optimise passes)
compiled, report = Compiler().compile(circ)
print(report)

# Simulate
result = GaussianSimulator().run(compiled, shots=500)
print(result)
print("Mean photons per mode:", result.expectation_values["mean_photon"])
```

---

## Project structure

```
LuminaQ_Platform/
├── src/                          ← TypeScript/React UI (Bolt, Vite, Tailwind, Supabase)
├── packages/
│   ├── lumq-photonics/           ← physics core
│   │   └── lumq/photonics/
│   │       ├── states/           ← GaussianState, FockState, StateVector
│   │       ├── gates/            ← gate library + symplectic primitives
│   │       ├── measurements/     ← homodyne, heterodyne, PNR, Fock
│   │       └── phase_space/      ← Wigner, Husimi Q, marginals
│   ├── lumq-compiler/
│   │   └── lumq/compiler/
│   │       ├── ir/               ← PhotonicCircuit, GateOp, MeasOp
│   │       └── decompose/        ← Clements mesh decomposer
│   ├── lumq-backends/
│   │   └── lumq/backends/        ← GaussianSimulator, FockSimulator, Device
│   ├── lumq-api/
│   │   └── lumq/api/
│   │       ├── app.py            ← FastAPI application
│   │       ├── schemas/          ← Pydantic v2 models
│   │       └── bridge/           ← schema ↔ native object conversion
│   └── lumq-algorithms/
│       └── lumq/algorithms/
│           └── gbs/              ← hafnian, circuit, sampling, graph encoding
├── pyproject.toml                ← monorepo root config
├── docker-compose.yml
├── INSTALL.md
└── .github/workflows/ci.yml
```

---

## Technical conventions

| Convention | Value |
|---|---|
| hbar | 2 (standard in CV-QC literature) |
| Vacuum variance | hbar/2 = 1 |
| Quadrature ordering | interleaved: (x₀, p₀, x₁, p₁, …) |
| Numerical backend | JAX (float64 / complex128) |
| Pydantic version | v2 |
| Python minimum | 3.10 |

---

## Roadmap

- [x] `lumq-photonics` — Gaussian and Fock state core
- [x] `lumq-compiler` — circuit IR, Clements decomposer, pass pipeline
- [x] `lumq-backends` — Gaussian and Fock simulators
- [x] `lumq-api` — FastAPI gateway
- [x] `lumq-algorithms` — GBS (hafnian, circuit builder, sampler, graph encoding)
- [ ] `lumqClient.ts` — TypeScript client connecting UI to API
- [ ] `lumq-algorithms` — VQE-CV (variational quantum eigensolver)
- [ ] `lumq-algorithms` — CV-QML (quantum machine learning)
- [ ] `lumq-algorithms` — quantum sensing / Heisenberg-limited metrology
- [ ] `lumq-backends` — Fock MPS simulator (large mode counts)
- [ ] `lumq-backends` — hardware adapters (Xanadu X-series, Quandela)
- [ ] Rust hafnian extension (`lumq_hafnian`) for N > 20

---

## Related work and acknowledgements

LuminaQ is inspired by and complementary to:

- **[PennyLane](https://pennylane.ai)** (Xanadu) — the differentiable quantum computing framework that pioneered hardware-agnostic quantum programming. LuminaQ is CV-native and JAX-first where PennyLane is now general-purpose.
- **[Strawberry Fields](https://strawberryfields.ai)** (Xanadu) — photonic quantum software for CV circuits.
- *Serafini, Quantum Continuous Variables* (CRC Press, 2017) — primary physics reference.
- *Weedbrook et al., Rev. Mod. Phys. 84 (2012)* — CV quantum information foundations.
- *Clements et al., Optica 3, 1460 (2016)* — the rectangular mesh decomposition algorithm.

---

## Publication

This project is documented through the **[Quantum Computing Files](https://quantumcomputingfiles.substack.com)** Substack publication — covering photonic and quantum computing research, software architecture, and the LuminaQ development journey.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Nuno Edgar Fernandes**  
Physics Engineer · Optoelectronics & Photonics  
Torres Vedras, Portugal  
[Quantum Computing Files](https://quantumcomputingfiles.substack.com) · [GitHub](https://github.com/nunofernandes-plight)


