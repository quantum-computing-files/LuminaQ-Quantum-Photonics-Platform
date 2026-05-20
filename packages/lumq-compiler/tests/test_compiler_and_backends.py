"""
tests/test_compiler_and_backends.py
====================================
Tests for lumq-compiler (IR, passes, Clements) and lumq-backends (simulators).
Run: pytest -v packages/lumq-compiler/tests/ packages/lumq-backends/tests/
"""

import math
import numpy as np
import pytest


# ============================================================
# PhotonicCircuit IR
# ============================================================


class TestPhotonicCircuit:
    def test_basic_construction(self):
        from lumq.compiler.ir import PhotonicCircuit
        circ = PhotonicCircuit(n_modes=4)
        circ.sq(0, r=1.0).bs(0, 1, theta=0.785).meas_homodyne(0)
        assert circ.n_modes == 4
        assert circ.gate_count == 2
        assert len(circ.measurements) == 1

    def test_depth(self):
        from lumq.compiler.ir import PhotonicCircuit
        circ = PhotonicCircuit(n_modes=3)
        circ.sq(0, r=1.0)
        circ.sq(1, r=1.0)
        circ.bs(0, 1, theta=0.5)
        assert circ.depth == 2   # sq0 and sq1 are parallel (depth=1), then bs (depth=2)

    def test_serialisation_roundtrip(self):
        from lumq.compiler.ir import PhotonicCircuit
        circ = PhotonicCircuit(n_modes=3)
        circ.sq(0, r=1.2, phi=0.3)
        circ.d(1, alpha=1.0 + 0.5j)
        circ.bs(0, 1, theta=0.785)
        circ.tms(1, 2, r=0.8)
        circ.meas_homodyne(0)
        circ.meas_heterodyne(1)

        json_str = circ.to_json()
        circ2 = PhotonicCircuit.from_json(json_str)

        assert circ2.n_modes == circ.n_modes
        assert circ2.gate_count == circ.gate_count
        assert len(circ2.measurements) == len(circ.measurements)
        # Complex alpha roundtrips
        alpha_orig = circ.ops[1].params["alpha"]
        alpha_rt   = circ2.ops[1].params["alpha"]
        assert abs(alpha_orig - alpha_rt) < 1e-10

    def test_invalid_mode_raises(self):
        from lumq.compiler.ir import PhotonicCircuit
        circ = PhotonicCircuit(n_modes=2)
        with pytest.raises(ValueError):
            circ.sq(5, r=1.0)

    def test_has_non_gaussian(self):
        from lumq.compiler.ir import PhotonicCircuit
        circ = PhotonicCircuit(n_modes=2)
        circ.sq(0, r=0.5)
        assert not circ.has_non_gaussian
        circ.kerr(0, kappa=0.1)
        assert circ.has_non_gaussian

    def test_draw_returns_string(self):
        from lumq.compiler.ir import PhotonicCircuit
        circ = PhotonicCircuit(n_modes=2)
        circ.sq(0, r=1.0).bs(0, 1, theta=0.5)
        s = circ.draw()
        assert isinstance(s, str)
        assert "q0" in s and "q1" in s


# ============================================================
# Clements decomposer
# ============================================================


class TestClements:
    def test_2x2_identity(self):
        from lumq.compiler.decompose.clements import clements_decompose
        U = np.eye(2, dtype=complex)
        result = clements_decompose(U)
        assert result.reconstruction_error < 1e-10

    def test_2x2_hadamard(self):
        from lumq.compiler.decompose.clements import clements_decompose
        U = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
        result = clements_decompose(U)
        assert result.reconstruction_error < 1e-10
        assert result.n_beamsplitters >= 1

    def test_4x4_haar_random(self):
        from lumq.compiler.decompose.clements import clements_decompose, random_unitary
        for seed in range(5):
            U = random_unitary(4, seed=seed)
            result = clements_decompose(U)
            assert result.reconstruction_error < 1e-9, \
                f"Seed {seed}: error = {result.reconstruction_error:.3e}"

    def test_8x8_haar_random(self):
        from lumq.compiler.decompose.clements import clements_decompose, random_unitary
        U = random_unitary(8, seed=99)
        result = clements_decompose(U)
        assert result.reconstruction_error < 1e-8

    def test_circuit_from_clements(self):
        from lumq.compiler.decompose.clements import clements_decompose, random_unitary
        U = random_unitary(3, seed=7)
        result = clements_decompose(U)
        circ = result.circuit
        assert circ.n_modes == 3
        assert circ.gate_count > 0

    def test_non_unitary_raises(self):
        from lumq.compiler.decompose.clements import clements_decompose
        with pytest.raises(ValueError, match="not unitary"):
            clements_decompose(np.array([[2.0, 0], [0, 0.5]]))


# ============================================================
# Compiler passes
# ============================================================


class TestPasses:
    def test_remove_identity_squeezer(self):
        from lumq.compiler.ir import PhotonicCircuit
        from lumq.compiler.passes import RemoveIdentityGates
        circ = PhotonicCircuit(n_modes=2)
        circ.sq(0, r=0.0)    # identity
        circ.sq(1, r=1.0)    # real gate
        result = RemoveIdentityGates()(circ)
        assert result.gate_count == 1
        assert result.ops[0].params["r"] == pytest.approx(1.0)

    def test_merge_phase_shifters(self):
        from lumq.compiler.ir import PhotonicCircuit
        from lumq.compiler.passes import MergePhaseShifters
        circ = PhotonicCircuit(n_modes=1)
        circ.ps(0, phi=0.3)
        circ.ps(0, phi=0.7)
        result = MergePhaseShifters()(circ)
        assert result.gate_count == 1
        assert result.ops[0].params["phi"] == pytest.approx(1.0)

    def test_merge_does_not_merge_across_other_gates(self):
        from lumq.compiler.ir import PhotonicCircuit
        from lumq.compiler.passes import MergePhaseShifters
        circ = PhotonicCircuit(n_modes=2)
        circ.ps(0, phi=0.3)
        circ.bs(0, 1, theta=0.5)   # barrier
        circ.ps(0, phi=0.7)
        result = MergePhaseShifters()(circ)
        assert result.gate_count == 3  # not merged

    def test_decompose_two_mode(self):
        from lumq.compiler.ir import PhotonicCircuit
        from lumq.compiler.passes import DecomposeTwoMode
        circ = PhotonicCircuit(n_modes=2)
        circ.tms(0, 1, r=1.0)
        result = DecomposeTwoMode()(circ)
        # TMS → 4 gates (2 BS + 2 Sq)
        assert result.gate_count == 4
        names = {op.name for op in result.ops}
        assert "Beamsplitter" in names
        assert "Squeezer" in names
        assert "TwoModeSqueeze" not in names

    def test_resource_estimator(self):
        from lumq.compiler.ir import PhotonicCircuit
        from lumq.compiler.passes import ResourceEstimator
        circ = PhotonicCircuit(n_modes=4)
        circ.sq(0, r=1.5).sq(1, r=1.0)
        circ.bs(0, 1, theta=0.785)
        circ.bs(1, 2, theta=0.524)
        estimator = ResourceEstimator()
        _, report = estimator(circ)
        assert report.n_squeezers == 2
        assert report.n_beamsplitters == 2
        assert report.squeezing_budget_db > 0
        assert not report.has_non_gaussian

    def test_default_pipeline(self):
        from lumq.compiler.ir import PhotonicCircuit
        from lumq.compiler.passes import PassPipeline
        circ = PhotonicCircuit(n_modes=2)
        circ.ps(0, phi=0.0)   # identity — should be removed
        circ.sq(0, r=1.0)
        circ.ps(0, phi=0.3)
        circ.ps(0, phi=0.7)   # should merge with above
        result = PassPipeline.default().run(circ)
        assert result.gate_count == 2  # Sq + merged PS


# ============================================================
# Compiler (end-to-end)
# ============================================================


class TestCompiler:
    def test_compile_basic_circuit(self):
        from lumq.compiler import Compiler, PhotonicCircuit
        circ = PhotonicCircuit(n_modes=3)
        circ.sq(0, r=1.0).sq(1, r=0.8).bs(0, 1, theta=0.785).bs(1, 2, theta=0.524)
        compiler = Compiler(decompose_interferometers=False)
        compiled, report = compiler.compile(circ)
        assert compiled.gate_count <= circ.gate_count
        assert report.n_modes == 3

    def test_compile_with_interferometer_expansion(self):
        from lumq.compiler import Compiler, PhotonicCircuit
        from lumq.compiler.decompose.clements import random_unitary
        U = random_unitary(3, seed=5)
        circ = PhotonicCircuit(n_modes=3)
        circ.interferometer(U)
        compiler = Compiler(decompose_interferometers=True)
        compiled, report = compiler.compile(circ)
        # Interferometer should be expanded — no Interferometer ops remain
        assert all(op.name != "Interferometer" for op in compiled.ops)
        assert report.n_beamsplitters > 0


# ============================================================
# GaussianSimulator
# ============================================================


class TestGaussianSimulator:
    def test_vacuum_mean_photon_zero(self):
        from lumq.backends.simulators.gaussian import GaussianSimulator
        from lumq.compiler.ir import PhotonicCircuit
        sim  = GaussianSimulator()
        circ = PhotonicCircuit(n_modes=3)   # no gates = vacuum
        result = sim.run(circ, shots=1)
        n_bar = result.expectation_values["mean_photon"]
        assert float(n_bar.max()) == pytest.approx(0.0, abs=1e-10)

    def test_squeezed_state_mean_photon(self):
        from lumq.backends.simulators.gaussian import GaussianSimulator
        from lumq.compiler.ir import PhotonicCircuit
        r = 1.0
        sim  = GaussianSimulator()
        circ = PhotonicCircuit(n_modes=1)
        circ.sq(0, r=r)
        result = sim.run(circ)
        n_bar = float(result.expectation_values["mean_photon"][0])
        # <n> = sinh²(r)
        expected = math.sinh(r) ** 2
        assert n_bar == pytest.approx(expected, rel=1e-5)

    def test_beamsplitter_energy_conservation(self):
        from lumq.backends.simulators.gaussian import GaussianSimulator
        from lumq.compiler.ir import PhotonicCircuit
        import math
        sim  = GaussianSimulator()
        circ = PhotonicCircuit(n_modes=2)
        circ.sq(0, r=1.5)
        circ.bs(0, 1, theta=math.pi / 4)
        result = sim.run(circ)
        n_total_before = math.sinh(1.5) ** 2
        n_total_after  = float(result.expectation_values["mean_photon"].sum())
        assert n_total_after == pytest.approx(n_total_before, rel=1e-5)

    def test_homodyne_sampling(self):
        from lumq.backends.simulators.gaussian import GaussianSimulator, GaussianSimulatorConfig
        from lumq.compiler.ir import PhotonicCircuit
        sim  = GaussianSimulator(GaussianSimulatorConfig(seed=42))
        circ = PhotonicCircuit(n_modes=1)
        circ.meas_homodyne(0)   # vacuum homodyne
        result = sim.run(circ, shots=500)
        assert result.samples is not None
        assert result.samples.shape == (500, 1)
        # Vacuum homodyne: mean ≈ 0, var ≈ hbar/2 = 1.0
        mean = float(result.samples.mean())
        var  = float(result.samples.var())
        assert abs(mean) < 0.15
        assert abs(var - 1.0) < 0.2

    def test_result_has_state(self):
        from lumq.backends.simulators.gaussian import GaussianSimulator
        from lumq.compiler.ir import PhotonicCircuit
        sim  = GaussianSimulator()
        circ = PhotonicCircuit(n_modes=2)
        circ.sq(0, r=0.8).tms(0, 1, r=0.5)
        result = sim.run(circ)
        assert result.state is not None
        assert result.state.n_modes == 2

    def test_end_to_end_with_compiler(self):
        """Full pipeline: build circuit → compile → simulate."""
        from lumq.backends.simulators.gaussian import GaussianSimulator
        from lumq.compiler import Compiler, PhotonicCircuit
        from lumq.compiler.decompose.clements import random_unitary
        import numpy as np

        N = 4
        U = random_unitary(N, seed=1)
        circ = PhotonicCircuit(n_modes=N)
        for k in range(N):
            circ.sq(k, r=0.8)
        circ.interferometer(U)
        for k in range(N):
            circ.meas_homodyne(k)

        compiler = Compiler(decompose_interferometers=True)
        compiled, report = compiler.compile(circ)

        sim    = GaussianSimulator()
        result = sim.run(compiled, shots=200)

        assert result.samples.shape == (200, N)
        assert report.n_beamsplitters > 0


# ============================================================
# FockSimulator (basic)
# ============================================================


class TestFockSimulator:
    def test_vacuum_norm(self):
        from lumq.backends.simulators.fock import FockSimulator, FockSimulatorConfig
        from lumq.compiler.ir import PhotonicCircuit
        sim  = FockSimulator(FockSimulatorConfig(n_modes=2, cutoff=6))
        circ = PhotonicCircuit(n_modes=2)
        result = sim.run(circ)
        import jax.numpy as jnp
        norm = float(jnp.sum(jnp.abs(result.state.data) ** 2))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_kerr_gate_runs(self):
        from lumq.backends.simulators.fock import FockSimulator, FockSimulatorConfig
        from lumq.compiler.ir import PhotonicCircuit
        sim  = FockSimulator(FockSimulatorConfig(n_modes=1, cutoff=8))
        circ = PhotonicCircuit(n_modes=1)
        circ.kerr(0, kappa=0.1)
        result = sim.run(circ)
        assert result.state is not None
