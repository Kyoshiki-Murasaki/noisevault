from __future__ import annotations

import numpy as np

from ..benchmarks import CircuitSpec, Operation
from ..channels import channel_sequence_for_operation
from ..readout import apply_readout_error
from .common import SimulationResult


def _bits(index: int, n: int) -> list[int]:
    return [(index >> (n - 1 - wire)) & 1 for wire in range(n)]


def _index(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def _expand_operator(operator: np.ndarray, wires: tuple[int, ...], n: int) -> np.ndarray:
    operator = np.asarray(operator, dtype=complex)
    local_dim = 2 ** len(wires)
    if operator.shape != (local_dim, local_dim):
        raise ValueError(f"Operator shape {operator.shape} incompatible with wires {wires}.")
    full_dim = 2**n
    full = np.zeros((full_dim, full_dim), dtype=complex)
    for input_index in range(full_dim):
        input_bits = _bits(input_index, n)
        local_input = _index([input_bits[wire] for wire in wires])
        for local_output in range(local_dim):
            amplitude = operator[local_output, local_input]
            if abs(amplitude) < 1e-15:
                continue
            output_bits = input_bits.copy()
            replacement = _bits(local_output, len(wires))
            for wire, bit in zip(wires, replacement, strict=True):
                output_bits[wire] = bit
            full[_index(output_bits), input_index] += amplitude
    return full


def _operation_matrix(operation: Operation) -> np.ndarray:
    if operation.name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if operation.name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if operation.name == "rx":
        theta = operation.params[0]
        return np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )
    if operation.name == "rz":
        theta = operation.params[0]
        return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)
    if operation.name == "cx":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        )
    raise ValueError(f"Unsupported operation {operation.name}")


def run_reference(snapshot, circuit: CircuitSpec, physical_qubits: list[int]) -> SimulationResult:
    if len(physical_qubits) != circuit.num_qubits:
        raise ValueError("physical_qubits length must match circuit.num_qubits")
    dimension = 2**circuit.num_qubits
    rho = np.zeros((dimension, dimension), dtype=complex)
    rho[0, 0] = 1.0

    for operation in circuit.operations:
        ideal = _expand_operator(_operation_matrix(operation), operation.wires, circuit.num_qubits)
        rho = ideal @ rho @ ideal.conj().T
        physical_wires = tuple(physical_qubits[wire] for wire in operation.wires)
        for channel in channel_sequence_for_operation(
            snapshot, operation.name, operation.wires, physical_wires
        ):
            updated = np.zeros_like(rho)
            for kraus in channel.kraus:
                full = _expand_operator(kraus, channel.wires, circuit.num_qubits)
                updated += full @ rho @ full.conj().T
            rho = updated

    probs = np.real_if_close(np.diag(rho)).real
    probs = apply_readout_error(probs, snapshot, physical_qubits)
    return SimulationResult(
        framework="reference",
        circuit_name=circuit.name,
        probabilities=probs,
        metadata={"physical_qubits": physical_qubits, "density_trace": float(np.trace(rho).real)},
    )
