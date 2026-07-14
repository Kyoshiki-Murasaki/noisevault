from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

import numpy as np

from .models import DeviceNoiseSnapshot, GateCalibration

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass(frozen=True)
class ChannelSpec:
    kind: str
    wires: tuple[int, ...]
    kraus: tuple[np.ndarray, ...]
    metadata: dict[str, float | str]


def compose_kraus(after: Iterable[np.ndarray], before: Iterable[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(a) @ np.asarray(b) for a in after for b in before]


def tensor_kraus(channels: list[list[np.ndarray]]) -> list[np.ndarray]:
    result = [np.array([[1.0 + 0.0j]])]
    for channel in channels:
        result = [np.kron(left, right) for left in result for right in channel]
    return result


def thermal_relaxation_kraus(t1_us: float | None, t2_us: float | None, duration_ns: float) -> list[np.ndarray]:
    if duration_ns <= 0 or t1_us is None or t2_us is None:
        return [I2.copy()]
    t1_ns = max(float(t1_us) * 1000.0, 1e-12)
    t2_ns = min(max(float(t2_us) * 1000.0, 1e-12), 2.0 * t1_ns)
    duration_ns = float(duration_ns)

    gamma_amp = float(np.clip(1.0 - np.exp(-duration_ns / t1_ns), 0.0, 1.0))
    amplitude = [
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma_amp)]], dtype=complex),
        np.array([[0.0, np.sqrt(gamma_amp)], [0.0, 0.0]], dtype=complex),
    ]

    pure_rate = max(0.0, 1.0 / t2_ns - 1.0 / (2.0 * t1_ns))
    coherence_factor = float(np.exp(-duration_ns * pure_rate))
    phase_flip_probability = float(np.clip((1.0 - coherence_factor) / 2.0, 0.0, 0.5))
    phase = [
        np.sqrt(1.0 - phase_flip_probability) * I2,
        np.sqrt(phase_flip_probability) * Z,
    ]
    return compose_kraus(phase, amplitude)


def _pauli_basis(num_qubits: int) -> list[np.ndarray]:
    basis = [I2, X, Y, Z]
    return [
        np.array([[1.0 + 0.0j]]) if num_qubits == 0 else _kron_many(items)
        for items in product(basis, repeat=num_qubits)
    ]


def _kron_many(items: Iterable[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for item in items:
        result = np.kron(result, item)
    return result


def depolarizing_kraus(avg_gate_error: float | None, num_qubits: int) -> list[np.ndarray]:
    if avg_gate_error is None or avg_gate_error <= 0:
        return [np.eye(2**num_qubits, dtype=complex)]
    dimension = 2**num_qubits
    max_infidelity = (dimension - 1.0) / dimension
    clipped_error = float(np.clip(avg_gate_error, 0.0, max_infidelity))
    lam = clipped_error * dimension / (dimension - 1.0)
    paulis = _pauli_basis(num_qubits)
    pauli_count = len(paulis)
    weights = [lam / pauli_count] * pauli_count
    weights[0] += 1.0 - lam
    return [np.sqrt(max(weight, 0.0)) * pauli for weight, pauli in zip(weights, paulis, strict=True)]


_GATE_ALIASES: dict[str, tuple[str, ...]] = {
    "h": ("h", "sx", "x"),
    "x": ("x", "sx"),
    "rx": ("rx", "sx", "x"),
    "rz": ("rz",),
    "cx": ("cx", "ecr", "cz"),
}


def find_gate_calibration(
    snapshot: DeviceNoiseSnapshot,
    operation_name: str,
    physical_wires: tuple[int, ...],
) -> GateCalibration | None:
    aliases = _GATE_ALIASES.get(operation_name, (operation_name,))
    exact: list[GateCalibration] = []
    reverse: list[GateCalibration] = []
    for gate in snapshot.gates:
        if gate.name not in aliases or not gate.operational:
            continue
        if tuple(gate.qubits) == physical_wires:
            exact.append(gate)
        elif len(physical_wires) == 2 and tuple(gate.qubits) == tuple(reversed(physical_wires)):
            reverse.append(gate)
    candidates = exact or reverse
    if not candidates:
        return None
    return min(candidates, key=lambda gate: float("inf") if gate.error is None else gate.error)


def channel_sequence_for_operation(
    snapshot: DeviceNoiseSnapshot,
    operation_name: str,
    logical_wires: tuple[int, ...],
    physical_wires: tuple[int, ...],
) -> list[ChannelSpec]:
    gate = find_gate_calibration(snapshot, operation_name, physical_wires)
    default_duration = 60.0 if len(logical_wires) == 1 else 600.0
    duration_ns = default_duration if gate is None or gate.duration_ns is None else gate.duration_ns
    gate_error = None if gate is None else gate.error
    channels: list[ChannelSpec] = []

    depolarizing = depolarizing_kraus(gate_error, len(logical_wires))
    if len(depolarizing) > 1:
        channels.append(
            ChannelSpec(
                kind="depolarizing",
                wires=logical_wires,
                kraus=tuple(depolarizing),
                metadata={"avg_gate_error": float(gate_error or 0.0)},
            )
        )

    by_index = {qubit.index: qubit for qubit in snapshot.qubits}
    for logical, physical in zip(logical_wires, physical_wires, strict=True):
        qubit = by_index[physical]
        thermal = thermal_relaxation_kraus(qubit.t1_us, qubit.t2_us, duration_ns)
        if len(thermal) > 1:
            channels.append(
                ChannelSpec(
                    kind="thermal_relaxation",
                    wires=(logical,),
                    kraus=tuple(thermal),
                    metadata={
                        "t1_us": float(qubit.t1_us or 0.0),
                        "t2_us": float(qubit.t2_us or 0.0),
                        "duration_ns": float(duration_ns),
                    },
                )
            )
    return channels
