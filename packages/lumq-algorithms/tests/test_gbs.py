"""lumq-algorithms GBS tests."""
import math
import numpy as np
import pytest
import jax.numpy as jnp


class TestHafnian:
    def test_empty_matrix(self):
        from lumq.algorithms.gbs.hafnian import hafnian
        assert float(jnp.abs(hafnian(jnp.zeros((0,0),dtype=jnp.complex128)) - 1.0)) < 1e-10

    def test_2x2(self):
        from lumq.algorithms.gbs.hafnian import hafnian
        A = jnp.array([[0,2],[2,0]], dtype=jnp.complex128)
        assert float(jnp.abs(hafnian(A) - 2.0)) < 1e-10

    def test_4x4_known(self):
        # haf([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]) = 1
        from lumq.algorithms.gbs.hafnian import hafnian
        A = jnp.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=jnp.complex128)
        assert float(jnp.abs(hafnian(A) - 1.0)) < 1e-8

    def test_odd_dimension_raises(self):
        from lumq.algorithms.gbs.hafnian import hafnian
        with pytest.raises(ValueError):
            hafnian(jnp.ones((3,3), dtype=jnp.complex128))

    def test_batch(self):
        from lumq.algorithms.gbs.hafnian import hafnian_batch
        A = jnp.array([[0,2],[2,0]], dtype=jnp.complex128)
        batch = jnp.stack([A, A, A])
        results = hafnian_batch(batch)
        assert results.shape == (3,)
        assert float(jnp.abs(results[0] - 2.0)) < 1e-10


class TestGBSCircuit:
    def test_gbs_circuit_structure(self):
        from lumq.algorithms.gbs.circuit import gbs_circuit
        N = 4
        r = np.array([0.8, 0.8, 0.6, 0.6])
        U = np.eye(N, dtype=complex)
        c = gbs_circuit(r, U)
        assert c.n_modes == N
        assert c.gate_count >= N  # N squeezers + interferometer
        assert len(c.measurements) == N

    def test_gbs_circuit_from_graph(self):
        from lumq.algorithms.gbs.circuit import gbs_circuit_from_graph
        A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
        circuit, r, U = gbs_circuit_from_graph(A, scale=0.4)
        assert circuit.n_modes == 4
        assert len(r) == 4
        assert U.shape == (4, 4)
        assert np.all(r >= 0)

    def test_wrong_shape_raises(self):
        from lumq.algorithms.gbs.circuit import gbs_circuit
        with pytest.raises(ValueError):
            gbs_circuit(np.array([1.0, 1.0]), np.eye(3))


class TestGBSSampler:
    def test_sample_shape(self):
        from lumq.algorithms.gbs.circuit import gbs_circuit_from_graph
        from lumq.algorithms.gbs.sampling import GBSSampler
        A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
        circuit, r, U = gbs_circuit_from_graph(A, scale=0.3)
        sampler = GBSSampler(seed=42)
        result = sampler.sample(circuit, shots=50)
        assert result.samples.shape == (50, 4)
        assert result.shots == 50
        assert result.n_modes == 4

    def test_result_properties(self):
        from lumq.algorithms.gbs.circuit import gbs_circuit_from_graph
        from lumq.algorithms.gbs.sampling import GBSSampler
        A = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        circuit, r, U = gbs_circuit_from_graph(A, scale=0.4)
        sampler = GBSSampler(seed=0)
        result = sampler.sample(circuit, shots=100)
        assert result.clicks.shape == (100, 3)
        assert result.mean_photon_number.shape == (3,)
        assert 0.0 <= result.collision_rate <= 1.0

    def test_click_patterns(self):
        from lumq.algorithms.gbs.circuit import gbs_circuit_from_graph
        from lumq.algorithms.gbs.sampling import GBSSampler
        A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
        circuit, r, U = gbs_circuit_from_graph(A, scale=0.3)
        result = GBSSampler(seed=7).sample(circuit, shots=200)
        patterns, counts = result.click_patterns
        assert counts.sum() == 200


class TestGraphEncoding:
    def test_adjacency_to_gbs(self):
        from lumq.algorithms.gbs.graph import adjacency_to_gbs
        A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
        circuit, r, U, info = adjacency_to_gbs(A, scale=0.4)
        assert info["n_nodes"] == 4
        assert info["n_edges"] == 5
        assert info["mean_photon_est"] > 0

    def test_gbs_to_graph_features(self):
        from lumq.algorithms.gbs.circuit import gbs_circuit_from_graph
        from lumq.algorithms.gbs.sampling import GBSSampler
        from lumq.algorithms.gbs.graph import gbs_to_graph_features
        A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
        circuit, r, U = gbs_circuit_from_graph(A, scale=0.3)
        result = GBSSampler(seed=1).sample(circuit, shots=100)
        features = gbs_to_graph_features(result, A)
        assert "max_clique_size" in features
        assert "density_scores" in features
        assert len(features["top_patterns"]) > 0


