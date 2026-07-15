from __future__ import annotations

import numpy as np

from ..adapters.cirq_adapter import to_cirq
from ..benchmarks import CircuitSpec
from ..readout import apply_readout_error
from .common import SimulationResult


def run_cirq(snapshot, circuit: CircuitSpec, physical_qubits: list[int]) -> SimulationResult:
    try:
        import cirq
    except ImportError as exc:
        raise RuntimeError("Cirq pilot dependencies are not installed.") from exc

    qubits = cirq.LineQubit.range(circuit.num_qubits)
    moments = []
    for operation in circuit.operations:
        if operation.name == "h":
            moments.append(cirq.H(qubits[operation.wires[0]]))
        elif operation.name == "x":
            moments.append(cirq.X(qubits[operation.wires[0]]))
        elif operation.name == "rx":
            moments.append(cirq.rx(operation.params[0])(qubits[operation.wires[0]]))
        elif operation.name == "rz":
            moments.append(cirq.rz(operation.params[0])(qubits[operation.wires[0]]))
        elif operation.name == "cx":
            moments.append(cirq.CNOT(qubits[operation.wires[0]], qubits[operation.wires[1]]))
        else:
            raise ValueError(f"Unsupported operation {operation.name}")

    conversion = to_cirq(snapshot, physical_qubits)
    cirq_circuit = cirq.Circuit(moments).with_noise(conversion.noise_model)
    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
    simulation = simulator.simulate(cirq_circuit, qubit_order=qubits)
    rho = np.asarray(simulation.final_density_matrix, dtype=complex)
    probs = np.real_if_close(np.diag(rho)).real
    probs = apply_readout_error(probs, snapshot, physical_qubits)
    return SimulationResult(
        framework="cirq",
        circuit_name=circuit.name,
        probabilities=probs,
        metadata={
            "physical_qubits": physical_qubits,
            "density_trace": float(np.trace(rho).real),
            "converter": "noisevault.adapters.to_cirq",
        },
    )
