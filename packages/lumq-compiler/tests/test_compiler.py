"""lumq-compiler tests."""
import math, numpy as np, pytest

class TestCircuitIR:
    def test_build(self):
        from lumq.compiler.ir import PhotonicCircuit
        c = PhotonicCircuit(n_modes=3)
        c.sq(0,r=1.0).bs(0,1,theta=0.785).meas_homodyne(0)
        assert c.gate_count==2 and c.depth==2

    def test_json_roundtrip(self):
        from lumq.compiler.ir import PhotonicCircuit
        c = PhotonicCircuit(n_modes=2)
        c.sq(0,r=1.2).d(1,alpha=1.0+0.5j).bs(0,1,theta=0.5)
        c2 = PhotonicCircuit.from_json(c.to_json())
        assert c2.gate_count==c.gate_count
        assert abs(c2.ops[1].params["alpha"]-(1.0+0.5j))<1e-10

    def test_invalid_mode(self):
        from lumq.compiler.ir import PhotonicCircuit
        with pytest.raises(ValueError):
            PhotonicCircuit(n_modes=2).sq(5,r=1.0)

class TestClements:
    def test_2x2(self):
        from lumq.compiler.decompose.clements import clements_decompose
        U = np.array([[1,1],[1,-1]],dtype=complex)/math.sqrt(2)
        assert clements_decompose(U).reconstruction_error < 1e-10

    def test_4x4_random(self):
        from lumq.compiler.decompose.clements import clements_decompose, random_unitary
        for seed in range(4):
            r = clements_decompose(random_unitary(4,seed=seed))
            assert r.reconstruction_error < 1e-9

    def test_non_unitary_raises(self):
        from lumq.compiler.decompose.clements import clements_decompose
        with pytest.raises(ValueError, match="not unitary"):
            clements_decompose(np.array([[2.,0],[0,.5]]))

class TestPasses:
    def test_remove_identity(self):
        from lumq.compiler import PhotonicCircuit, RemoveIdentityGates
        c = PhotonicCircuit(n_modes=2).sq(0,r=0.0).sq(1,r=1.0)
        assert RemoveIdentityGates()(c).gate_count==1

    def test_merge_ps(self):
        from lumq.compiler import PhotonicCircuit, MergePhaseShifters
        c = PhotonicCircuit(n_modes=1).ps(0,phi=0.3).ps(0,phi=0.7)
        result = MergePhaseShifters()(c)
        assert result.gate_count==1
        assert result.ops[0].params["phi"]==pytest.approx(1.0)

    def test_decompose_tms(self):
        from lumq.compiler import PhotonicCircuit, DecomposeTwoMode
        c = PhotonicCircuit(n_modes=2).tms(0,1,r=1.0)
        result = DecomposeTwoMode()(c)
        assert result.gate_count==4
        assert "TwoModeSqueeze" not in {op.name for op in result.ops}

