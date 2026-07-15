from __future__ import annotations

import pytest

from noisevault.models import GateCalibration
from noisevault.select import best_connected_subset, connected


def test_connected_subset(demo_snapshot):
    for size in range(2, 6):
        subset = best_connected_subset(demo_snapshot, size)
        assert len(subset) == size
        assert connected(demo_snapshot, subset)
        assert all(
            any(
                gate.operational
                and gate.name in {"cx", "ecr", "cz"}
                and tuple(gate.qubits) == edge
                for gate in demo_snapshot.gates
            )
            for edge in zip(subset, subset[1:], strict=False)
        )


def test_directed_star_without_path_is_rejected(demo_snapshot):
    snapshot = demo_snapshot.model_copy(deep=True)
    snapshot.num_qubits = 3
    snapshot.qubits = snapshot.qubits[:3]
    snapshot.coupling_map = [[0, 1], [0, 2]]
    snapshot.gates = [gate for gate in snapshot.gates if len(gate.qubits) == 1 and gate.qubits[0] < 3]
    snapshot.gates.extend(
        [
            GateCalibration(name="cx", qubits=[0, 1], error=0.01, duration_ns=300),
            GateCalibration(name="cx", qubits=[0, 2], error=0.01, duration_ns=300),
        ]
    )
    with pytest.raises(ValueError, match="directed, calibrated entangling path"):
        best_connected_subset(snapshot, 3)
