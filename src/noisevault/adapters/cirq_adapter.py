from __future__ import annotations

from dataclasses import dataclass

from ..channels import channel_sequence_for_operation
from ..models import DeviceNoiseSnapshot


@dataclass
class CirqConversion:
    noise_model: object
    logical_to_physical: dict[int, int]
    notes: list[str]


def to_cirq(snapshot: DeviceNoiseSnapshot, physical_qubits: list[int]) -> CirqConversion:
    try:
        import cirq
    except ImportError as exc:
        raise RuntimeError("Install noisevault[pilot] to use the Cirq converter.") from exc

    logical_to_physical = {logical: physical for logical, physical in enumerate(physical_qubits)}

    class NoiseVaultCirqNoiseModel(cirq.NoiseModel):
        def noisy_operation(self, operation):
            gate = operation.gate
            if isinstance(gate, cirq.HPowGate):
                name = "h"
            elif isinstance(gate, cirq.CXPowGate):
                name = "cx"
            elif isinstance(gate, cirq.XPowGate):
                name = "rx" if getattr(gate, "global_shift", 0.0) != 0.0 else "x"
            elif isinstance(gate, cirq.ZPowGate):
                name = "rz"
            else:
                return operation
            logical_wires = tuple(int(qubit.x) for qubit in operation.qubits)
            physical_wires = tuple(logical_to_physical[wire] for wire in logical_wires)
            channels = channel_sequence_for_operation(
                snapshot, name, logical_wires, physical_wires
            )
            additions = [operation]
            for channel in channels:
                gate = cirq.KrausChannel(kraus_ops=list(channel.kraus))
                additions.append(gate.on(*(cirq.LineQubit(wire) for wire in channel.wires)))
            return additions

    notes = [
        "Noise is inserted after each ideal operation using canonical Kraus channels.",
        "Readout confusion is applied in common classical post-processing.",
    ]
    return CirqConversion(NoiseVaultCirqNoiseModel(), logical_to_physical, notes)
