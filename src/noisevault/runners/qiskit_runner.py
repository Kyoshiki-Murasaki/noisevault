from __future__ import annotations

import numpy as np

from ..benchmarks import CircuitSpec
from ..channels import channel_sequence_for_operation
from ..readout import apply_readout_error
from .common import SimulationResult


def _append_ideal(qc, operation) -> None:
    if operation.name == "h":
        qc.h(operation.wires[0])
    elif operation.name == "x":
        qc.x(operation.wires[0])
    elif operation.name == "rx":
        qc.rx(operation.params[0], operation.wires[0])
    elif operation.name == "rz":
        qc.rz(operation.params[0], operation.wires[0])
    elif operation.name == "cx":
        qc.cx(*operation.wires)
    else:
        raise ValueError(f"Unsupported operation {operation.name}")


def _canonicalize_qiskit_diagonal(diagonal: np.ndarray, num_qubits: int) -> np.ndarray:
    canonical = np.zeros_like(diagonal, dtype=float)
    for canonical_index in range(2**num_qubits):
        bits = [(canonical_index >> (num_qubits - 1 - wire)) & 1 for wire in range(num_qubits)]
        qiskit_index = sum(bit << wire for wire, bit in enumerate(bits))
        canonical[canonical_index] = float(diagonal[qiskit_index])
    return canonical


def run_qiskit(snapshot, circuit: CircuitSpec, physical_qubits: list[int]) -> SimulationResult:
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Kraus
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise RuntimeError("Qiskit pilot dependencies are not installed.") from exc

    qc = QuantumCircuit(circuit.num_qubits)
    for operation in circuit.operations:
        _append_ideal(qc, operation)
        physical_wires = tuple(physical_qubits[wire] for wire in operation.wires)
        for channel in channel_sequence_for_operation(
            snapshot, operation.name, operation.wires, physical_wires
        ):
            instruction = Kraus(list(channel.kraus)).to_instruction()
            qc.append(instruction, list(channel.wires))
    qc.save_density_matrix()
    result = AerSimulator(method="density_matrix").run(qc).result()
    rho = np.asarray(result.data(0)["density_matrix"], dtype=complex)
    diagonal = np.real_if_close(np.diag(rho)).real
    probs = _canonicalize_qiskit_diagonal(diagonal, circuit.num_qubits)
    probs = apply_readout_error(probs, snapshot, physical_qubits)
    return SimulationResult(
        framework="qiskit",
        circuit_name=circuit.name,
        probabilities=probs,
        metadata={"physical_qubits": physical_qubits, "density_trace": float(np.trace(rho).real)},
    )
