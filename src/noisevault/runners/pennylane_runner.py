from __future__ import annotations

import numpy as np

from ..benchmarks import CircuitSpec
from ..channels import channel_sequence_for_operation
from ..readout import apply_readout_error
from .common import SimulationResult


def run_pennylane(snapshot, circuit: CircuitSpec, physical_qubits: list[int]) -> SimulationResult:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise RuntimeError("PennyLane pilot dependencies are not installed.") from exc

    device = qml.device("default.mixed", wires=circuit.num_qubits, shots=None)

    @qml.qnode(device)
    def qnode():
        for operation in circuit.operations:
            if operation.name == "h":
                qml.Hadamard(wires=operation.wires[0])
            elif operation.name == "x":
                qml.PauliX(wires=operation.wires[0])
            elif operation.name == "rx":
                qml.RX(operation.params[0], wires=operation.wires[0])
            elif operation.name == "rz":
                qml.RZ(operation.params[0], wires=operation.wires[0])
            elif operation.name == "cx":
                qml.CNOT(wires=list(operation.wires))
            else:
                raise ValueError(f"Unsupported operation {operation.name}")

            physical_wires = tuple(physical_qubits[wire] for wire in operation.wires)
            for channel in channel_sequence_for_operation(
                snapshot, operation.name, operation.wires, physical_wires
            ):
                qml.QubitChannel(list(channel.kraus), wires=list(channel.wires))
        return qml.probs(wires=range(circuit.num_qubits))

    probs = np.asarray(qnode(), dtype=float)
    probs = apply_readout_error(probs, snapshot, physical_qubits)
    return SimulationResult(
        framework="pennylane",
        circuit_name=circuit.name,
        probabilities=probs,
        metadata={"physical_qubits": physical_qubits},
    )
