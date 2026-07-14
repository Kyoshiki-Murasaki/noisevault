from __future__ import annotations

import numpy as np

from ..adapters.pennylane_adapter import to_pennylane
from ..benchmarks import CircuitSpec
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

        return qml.probs(wires=range(circuit.num_qubits))

    conversion = to_pennylane(snapshot, physical_qubits)
    noisy_qnode = qml.add_noise(qnode, conversion.noise_model)
    probs = np.asarray(noisy_qnode(), dtype=float)
    probs = apply_readout_error(probs, snapshot, physical_qubits)
    return SimulationResult(
        framework="pennylane",
        circuit_name=circuit.name,
        probabilities=probs,
        metadata={
            "physical_qubits": physical_qubits,
            "converter": "noisevault.adapters.to_pennylane",
        },
    )
