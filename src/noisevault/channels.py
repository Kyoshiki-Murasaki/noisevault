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
    metadata: dict[str, float | str | bool | None]


class CalibrationUnavailableError(ValueError):
    """Raised when a requested operation has no direction-aligned calibration."""


def compose_kraus(after: Iterable[np.ndarray], before: Iterable[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(a) @ np.asarray(b) for a in after for b in before]


def tensor_kraus(channels: list[list[np.ndarray]]) -> list[np.ndarray]:
    result = [np.array([[1.0 + 0.0j]])]
    for channel in channels:
        result = [np.kron(left, right) for left in result for right in channel]
    return result


def thermal_relaxation_kraus(
    t1_us: float | None, t2_us: float | None, duration_ns: float
) -> list[np.ndarray]:
    if duration_ns <= 0 or (t1_us is None and t2_us is None):
        return [I2.copy()]
    t1_ns = np.inf if t1_us is None else max(float(t1_us) * 1000.0, 1e-12)
    if t2_us is None:
        t2_ns = np.inf if not np.isfinite(t1_ns) else 2.0 * t1_ns
    else:
        t2_ns = max(float(t2_us) * 1000.0, 1e-12)
        if np.isfinite(t1_ns):
            t2_ns = min(t2_ns, 2.0 * t1_ns)
    duration_ns = float(duration_ns)

    gamma_amp = (
        0.0
        if not np.isfinite(t1_ns)
        else float(np.clip(1.0 - np.exp(-duration_ns / t1_ns), 0.0, 1.0))
    )
    amplitude = [
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma_amp)]], dtype=complex),
        np.array([[0.0, np.sqrt(gamma_amp)], [0.0, 0.0]], dtype=complex),
    ]

    inverse_t1 = 0.0 if not np.isfinite(t1_ns) else 1.0 / t1_ns
    inverse_t2 = 0.0 if not np.isfinite(t2_ns) else 1.0 / t2_ns
    pure_rate = max(0.0, inverse_t2 - inverse_t1 / 2.0)
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
    # The most general n-qubit depolarizing channel remains CPTP through
    # lambda=4**n/(4**n-1), corresponding to average infidelity d/(d+1).
    max_infidelity = dimension / (dimension + 1.0)
    clipped_error = float(np.clip(avg_gate_error, 0.0, max_infidelity))
    lam = clipped_error * dimension / (dimension - 1.0)
    lam = min(lam, 4**num_qubits / (4**num_qubits - 1.0))
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
    # Alias order is semantic precedence: an exact H calibration must not lose to
    # a numerically smaller SX/X fallback. Two-qubit qargs are directional and are
    # never silently borrowed from the reversed operation.
    for alias in aliases:
        candidates = [
            gate
            for gate in snapshot.gates
            if gate.name == alias
            and gate.operational
            and tuple(gate.qubits) == physical_wires
        ]
        if candidates:
            return min(
                candidates,
                key=lambda gate: float("inf") if gate.error is None else gate.error,
            )
    return None


def average_gate_fidelity_from_kraus(kraus: Iterable[np.ndarray]) -> float:
    """Return average gate fidelity to identity for a trace-preserving channel."""
    operators = [np.asarray(operator, dtype=complex) for operator in kraus]
    if not operators:
        raise ValueError("At least one Kraus operator is required.")
    dimension = operators[0].shape[0]
    if any(operator.shape != (dimension, dimension) for operator in operators):
        raise ValueError("Kraus operators must be square and have one common dimension.")
    entanglement_fidelity = sum(abs(np.trace(operator)) ** 2 for operator in operators)
    entanglement_fidelity /= dimension**2
    return float((dimension * entanglement_fidelity.real + 1.0) / (dimension + 1.0))


def _residual_depolarizing_infidelity(
    reported_error: float | None,
    relaxation_kraus: list[np.ndarray],
    num_qubits: int,
) -> tuple[float | None, float, float | None]:
    """Match Aer's residual-depolarization convention for total gate infidelity."""
    relaxation_fidelity = average_gate_fidelity_from_kraus(relaxation_kraus)
    relaxation_infidelity = max(0.0, 1.0 - relaxation_fidelity)
    if reported_error is None or reported_error <= relaxation_infidelity:
        return None, relaxation_infidelity, None

    dimension = 2**num_qubits
    target_error = min(float(reported_error), dimension / (dimension + 1.0))
    denominator = dimension * relaxation_fidelity - 1.0
    if denominator <= 0:
        return None, relaxation_infidelity, None
    strength = dimension * (target_error - relaxation_infidelity) / denominator
    strength = float(
        np.clip(strength, 0.0, 4**num_qubits / (4**num_qubits - 1.0))
    )
    residual_infidelity = strength * (dimension - 1.0) / dimension
    return residual_infidelity, relaxation_infidelity, strength


def channel_sequence_for_operation(
    snapshot: DeviceNoiseSnapshot,
    operation_name: str,
    logical_wires: tuple[int, ...],
    physical_wires: tuple[int, ...],
) -> list[ChannelSpec]:
    gate = find_gate_calibration(snapshot, operation_name, physical_wires)
    if gate is None and len(logical_wires) > 1:
        raise CalibrationUnavailableError(
            f"No direction-aligned calibration for {operation_name}{physical_wires}."
        )
    default_duration = 60.0 if len(logical_wires) == 1 else 600.0
    if gate is None:
        duration_ns = default_duration
        duration_source = "pilot_default_missing_calibration"
    elif gate.duration_ns is None:
        duration_ns = default_duration
        duration_source = "pilot_default_missing_duration"
    else:
        duration_ns = gate.duration_ns
        duration_source = "snapshot"
    gate_error = None if gate is None else gate.error
    duration_ns = float(duration_ns)
    thermal_specs: list[ChannelSpec] = []
    thermal_by_wire: list[list[np.ndarray]] = []

    by_index = {qubit.index: qubit for qubit in snapshot.qubits}
    for logical, physical in zip(logical_wires, physical_wires, strict=True):
        qubit = by_index[physical]
        thermal = thermal_relaxation_kraus(qubit.t1_us, qubit.t2_us, duration_ns)
        thermal_by_wire.append(thermal)
        if len(thermal) > 1:
            thermal_specs.append(
                ChannelSpec(
                    kind="thermal_relaxation",
                    wires=(logical,),
                    kraus=tuple(thermal),
                    metadata={
                        "t1_us": qubit.t1_us,
                        "t2_us": qubit.t2_us,
                        "duration_ns": duration_ns,
                        "duration_source": duration_source,
                    },
                )
            )

    relaxation_kraus = tensor_kraus(thermal_by_wire)
    residual_error, relaxation_error, depolarizing_strength = (
        _residual_depolarizing_infidelity(
            gate_error, relaxation_kraus, len(logical_wires)
        )
    )
    channels: list[ChannelSpec] = []
    depolarizing = depolarizing_kraus(residual_error, len(logical_wires))
    if len(depolarizing) > 1:
        channels.append(
            ChannelSpec(
                kind="depolarizing",
                wires=logical_wires,
                kraus=tuple(depolarizing),
                metadata={
                    "reported_total_avg_gate_error": gate_error,
                    "relaxation_avg_gate_error": relaxation_error,
                    "residual_depolarizing_avg_gate_error": residual_error,
                    "depolarizing_strength": depolarizing_strength,
                    "calibration_gate": None if gate is None else gate.name,
                    "duration_source": duration_source,
                },
            )
        )
    return channels + thermal_specs
