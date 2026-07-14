from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pytest

from noisevault.metrics import total_variation_distance


def test_committed_results_are_internally_consistent(root):
    path = root / "experiments/results/pilot_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for record in data["simulations"]:
        probabilities = np.asarray(record["probabilities"], dtype=float)
        assert np.all(np.isfinite(probabilities))
        assert np.all(probabilities >= -1e-12)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-10)
        grouped[record["circuit_name"]][record["framework"]] = probabilities

    values = []
    for frameworks in grouped.values():
        if not all(name in frameworks for name in ("qiskit", "cirq", "pennylane")):
            continue
        for first, second in (("qiskit", "cirq"), ("qiskit", "pennylane"), ("cirq", "pennylane")):
            values.append(total_variation_distance(frameworks[first], frameworks[second]))

    assert values
    assert max(values) == pytest.approx(data["experiment_b"]["max_pairwise_tvd"], abs=1e-15)
    assert float(np.mean(values)) == pytest.approx(
        data["experiment_b"]["mean_pairwise_tvd"], abs=1e-15
    )
    report = (root / "experiments/PILOT_RESULTS.md").read_text(encoding="utf-8")
    assert data["summary"]["status"] in report
    assert data["experiment_d"]["pair_classification"] in report
