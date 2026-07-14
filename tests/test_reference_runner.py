from __future__ import annotations

import numpy as np

from noisevault.benchmarks import ghz, mirror
from noisevault.runners.reference import run_reference
from noisevault.select import best_connected_subset


def test_reference_probabilities_are_normalized(demo_snapshot):
    for circuit in (ghz(2), mirror(3, 2, 123)):
        physical = best_connected_subset(demo_snapshot, circuit.num_qubits)
        result = run_reference(demo_snapshot, circuit, physical)
        assert np.isclose(result.probabilities.sum(), 1.0)
        assert np.all(np.isfinite(result.probabilities))
        assert np.all(result.probabilities >= 0)
