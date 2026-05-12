"""
examples/gbs_demo.py
====================
GBS demo: encode a graph and sample photon-number patterns.

Run: python examples/gbs_demo.py
"""
import numpy as np

# Graph: 4-node cycle + diagonals (K4 minus one edge)
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
], dtype=float)

print("Graph adjacency matrix:")
print(A)
print()

from lumq.algorithms.gbs.circuit import gbs_circuit_from_graph
from lumq.algorithms.gbs.sampling import GBSSampler
from lumq.algorithms.gbs.graph import gbs_to_graph_features

circuit, r, U = gbs_circuit_from_graph(A, scale=0.4)
print(f"GBS circuit: {circuit.n_modes} modes")
print(f"Squeezing params: {np.round(r, 3)}")
print(f"Mean photon number (est.): {np.sum(np.sinh(r)**2):.3f}")
print()

sampler = GBSSampler(seed=42)
result = sampler.sample(circuit, shots=500)

print(f"Sampling result: {result}")
print()

# Top click patterns
patterns, counts = result.click_patterns
print("Top-5 click patterns (nodes clicked | count):")
for pat, cnt in zip(patterns[:5], counts[:5]):
    nodes = [i for i,v in enumerate(pat) if v]
    print(f"  nodes={nodes:20s}  count={cnt}")

print()

# Graph features
features = gbs_to_graph_features(result, A)
print(f"Max clique found: size {features['max_clique_size']}")
if features["clique_candidates"]:
    print(f"Clique candidates: {features['clique_candidates'][:3]}")


