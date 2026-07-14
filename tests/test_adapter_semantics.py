from __future__ import annotations

import numpy as np
import pytest

from noisevault.adapters.qiskit_adapter import (
    _canonical_to_qiskit_operator,
    _sequence_to_quantum_error,
    to_qiskit_aer,
)
from noisevault.channels import channel_sequence_for_operation, find_gate_calibration
from noisevault.readout import confusion_matrix
from noisevault.runners.qiskit_runner import _canonicalize_qiskit_diagonal
from noisevault.snapshot_io import load_snapshot


def test_qiskit_readout_boundary_transposes_canonical_matrix(demo_snapshot):
    conversion = to_qiskit_aer(demo_snapshot, [0, 1], include_readout=True)
    readout_records = [
        record
        for record in conversion.noise_model.to_dict()["errors"]
        if record["type"] == "roerror"
    ]
    assert len(readout_records) == 2
    qubit = demo_snapshot.qubits[0]
    expected = confusion_matrix(
        qubit.prob_meas0_prep1, qubit.prob_meas1_prep0
    ).T
    assert np.allclose(readout_records[0]["probabilities"], expected)
    assert np.allclose(expected.sum(axis=1), 1.0)


def test_qiskit_multiqubit_kraus_subsystem_order():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    from qiskit_aer import AerSimulator

    identity = np.eye(2, dtype=complex)
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    canonical_x_on_wire_zero = np.kron(x_gate, identity)
    qiskit_operator = _canonical_to_qiskit_operator(canonical_x_on_wire_zero)

    circuit = QuantumCircuit(2)
    circuit.append(Kraus([qiskit_operator]).to_instruction(), [0, 1])
    circuit.save_density_matrix()
    result = AerSimulator(method="density_matrix").run(circuit).result()
    diagonal = np.diag(np.asarray(result.data(0)["density_matrix"], dtype=complex)).real
    canonical = _canonicalize_qiskit_diagonal(diagonal, 2)
    assert canonical == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-12)


def test_composed_channel_targets_archived_gate_infidelity(root):
    from qiskit.quantum_info import average_gate_fidelity

    snapshot = load_snapshot(
        root / "snapshots/qiskit_fake/fake_manila/2024-05-27T15-27-23-03-00.json"
    )
    calibration = find_gate_calibration(snapshot, "cx", (0, 1))
    sequence = channel_sequence_for_operation(snapshot, "cx", (0, 1), (0, 1))
    error = _sequence_to_quantum_error(sequence, (0, 1))
    reconstructed = 1.0 - float(average_gate_fidelity(error.to_quantumchannel()))
    assert reconstructed == pytest.approx(calibration.error, abs=1e-12)

