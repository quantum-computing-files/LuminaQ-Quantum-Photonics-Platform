"""lumq-backends tests."""
import math, pytest
import jax.numpy as jnp

class TestGaussianSimulator:
    def test_vacuum_zero_photons(self):
        from lumq.backends import GaussianSimulator
        from lumq.compiler.ir import PhotonicCircuit
        r = GaussianSimulator().run(PhotonicCircuit(n_modes=2),shots=1)
        assert float(r.expectation_values["mean_photon"].max())==pytest.approx(0.0,abs=1e-10)

    def test_squeezed_mean_photon(self):
        from lumq.backends import GaussianSimulator
        from lumq.compiler.ir import PhotonicCircuit
        r = 1.0
        result = GaussianSimulator().run(PhotonicCircuit(n_modes=1).sq(0,r=r))
        assert float(result.expectation_values["mean_photon"][0])==pytest.approx(math.sinh(r)**2,rel=1e-4)

    def test_homodyne_shape(self):
        from lumq.backends import GaussianSimulator, GaussianSimulatorConfig
        from lumq.compiler.ir import PhotonicCircuit
        c = PhotonicCircuit(n_modes=1).meas_homodyne(0)
        r = GaussianSimulator(GaussianSimulatorConfig(seed=0)).run(c,shots=200)
        assert r.samples.shape==(200,1)

    def test_energy_conservation(self):
        from lumq.backends import GaussianSimulator
        from lumq.compiler.ir import PhotonicCircuit
        c = PhotonicCircuit(n_modes=2).sq(0,r=1.5).bs(0,1,theta=math.pi/4)
        r = GaussianSimulator().run(c)
        assert float(r.expectation_values["mean_photon"].sum())==pytest.approx(math.sinh(1.5)**2,rel=1e-4)

class TestFockSimulator:
    def test_vacuum_norm(self):
        from lumq.backends import FockSimulator, FockSimulatorConfig
        from lumq.compiler.ir import PhotonicCircuit
        r = FockSimulator(FockSimulatorConfig(n_modes=2,cutoff=6)).run(PhotonicCircuit(n_modes=2))
        norm = float(jnp.sum(jnp.abs(r.state.data)**2))
        assert norm==pytest.approx(1.0,abs=1e-6)

    def test_kerr(self):
        from lumq.backends import FockSimulator, FockSimulatorConfig
        from lumq.compiler.ir import PhotonicCircuit
        c = PhotonicCircuit(n_modes=1).kerr(0,kappa=0.1)
        r = FockSimulator(FockSimulatorConfig(n_modes=1,cutoff=8)).run(c)
        assert r.state is not None
