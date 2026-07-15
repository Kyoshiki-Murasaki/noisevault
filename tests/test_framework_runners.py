from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from noisevault.benchmarks import ghz
from noisevault.metrics import total_variation_distance
from noisevault.runners.cirq_runner import run_cirq
from noisevault.runners.pennylane_runner import run_pennylane
from noisevault.runners.qiskit_runner import run_qiskit
from noisevault.runners.reference import run_reference
from noisevault.select import best_connected_subset


@pytest.mark.parametrize(
    ("module", "runner"),
    [
        ("qiskit_aer", run_qiskit),
        ("cirq", run_cirq),
        ("pennylane", run_pennylane),
    ],
)
def test_framework_matches_reference(module, runner, demo_snapshot):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    for size in (2, 3):
        circuit = ghz(size)
        physical = best_connected_subset(demo_snapshot, size)
        reference = run_reference(demo_snapshot, circuit, physical)
        actual = runner(demo_snapshot, circuit, physical)
        assert np.isclose(actual.probabilities.sum(), 1.0)
        assert total_variation_distance(reference.probabilities, actual.probabilities) <= 1e-7
