from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

from . import gates
from .channels import ChannelSpec, GateChannels, gate_channels, thermal_relaxation_kraus
from .errors import LayoutError, MissingCalibrationError, qubit_loci
from .report import Report
from .table import GateNoise, NoiseTable, Unavailable

UnknownGates = Literal["typical", "error"]
TYPICAL_FIX = "compile to native gates for realistic gate counts, or pass unknown_gates='error'"


# The gates each registry gate equals, in the order a profile without its own calibration
# takes theirs: a fixed gate equals its rotation at one angle, up to global phase. A z-family
# fixed gate, and u1, equal p exactly, so p comes before rz.
_FALLBACKS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        **dict.fromkeys(("x", "sx", "sxdg"), ("rx",)),
        "y": ("ry",),
        **dict.fromkeys(("z", "s", "sdg", "t", "tdg", "u1"), ("p", "rz")),
        "p": ("rz",),
        "zz": ("rzz",),
    }
)


def native_name(name: str, defined: Container[str], rotation: str | None = None) -> str:
    """The gate whose calibration an operation equal to registry gate ``name`` uses.

    ``defined`` holds a profile's gate names. The profile's ``name`` comes first, then the
    first defined gate it equals: ``rx`` for ``sx``; ``p``, then ``rz``, for ``s`` and the other
    z-family gates. ``rotation`` names that gate for a native outside this table (``rxx`` or
    ``ryy`` for an ``ms``). With none defined, the operation keeps its own name, except that a
    native with parameters gives way to ``rotation``, so errors and reports name the gate the
    circuit wrote.
    """
    if name in defined:
        return name
    equal = _FALLBACKS.get(name) or ((rotation,) if rotation else ())
    found = next((gate for gate in equal if gate in defined), None)
    if found is not None:
        return found
    info = gates.lookup(name)
    return rotation if rotation and info and info.params else name


def resolve_op(
    table: NoiseTable,
    name: str,
    qubits: Sequence[int],
    *,
    unknown_gates: UnknownGates,
    report: Report,
) -> GateChannels:
    """Channels for gate ``name`` on physical ``qubits``; records what it did in ``report``.

    An ideal gate gets no channels and a calibrated one its own. A disabled gate raises
    DisabledGateError. A gate with no calibration there (not defined, not a native, no error
    metric, or a pair the profile does not allow) raises MissingCalibrationError when
    ``unknown_gates`` is ``"error"``; with ``"typical"`` it gets the noise of the typical native
    gate of its arity, reported and warned about once per gate name. Under either setting, a
    gate with no calibration there raises MissingCalibrationError if it needs several native
    entanglers or acts on more than two qubits.
    """
    if unknown_gates not in ("typical", "error"):
        raise ValueError(f"unknown_gates={unknown_gates!r}: choose 'typical' or 'error'")
    qubits = tuple(qubits)
    _check_qubits(table, name, qubits)
    info = gates.lookup(name)
    if info is not None and info.unitary is None:
        raise ValueError(f"{name} is not a unitary gate; resolve_op only handles gates")

    found = table.gate(name, qubits)
    if isinstance(found, GateNoise) and found.state != "uncalibrated":
        return _channels(table, found, report)  # a disabled gate raises here
    if isinstance(found, Unavailable) and found.kind == "bad_target":
        raise ValueError(found.reason)
    why = found.reason if isinstance(found, Unavailable) else f"{name} has no error metric"
    where = f"{name} on {qubit_loci(qubits)}"
    if (info is not None and info.multi_entangler) or len(qubits) > 2:
        raise MissingCalibrationError(
            f"{where}: {name} needs more than one native entangling gate, so no single"
            " calibration describes it",
            hint="decompose it into the profile's native gates first",
        )
    if unknown_gates == "error":
        raise MissingCalibrationError(
            f"{where}: {why}",
            hint="compile to the profile's native gates, or pass unknown_gates='typical' to use"
            " the typical native gate's noise",
        )
    typical = table.typical(len(qubits), qubits)
    if isinstance(typical, Unavailable):
        raise MissingCalibrationError(f"{where}: {why}, and {typical.reason}")
    report.count("typical_noise_used", name)
    report.approximate(
        f"gate {name}", f"noise of the typical {len(qubits)}-qubit native gate", TYPICAL_FIX
    )
    report.warn_once(
        f"typical_noise_used:{name}",
        f"{where}: {why}; using the noise of {typical.gate} instead. To fix: {TYPICAL_FIX}",
    )
    return _channels(table, typical, report)


def _check_qubits(table: NoiseTable, name: str, qubits: tuple[int, ...]) -> None:
    if len(set(qubits)) != len(qubits):
        raise LayoutError(f"{name} acts on {qubit_loci(qubits)}; its targets must be distinct")
    for q in qubits:
        if not 0 <= q < table.num_qubits:
            raise LayoutError(
                f"{name} acts on physical qubit {q}, but the device has qubits"
                f" 0..{table.num_qubits - 1}",
                hint="fix the layout",
            )
        if table.qubit(q).disabled:
            raise LayoutError(
                f"{name} acts on physical qubit {q}, which the profile marks disabled",
                hint="map the circuit elsewhere (profile.suggest_layout(n) proposes a usable"
                " chain)",
            )


def _channels(table: NoiseTable, gate: GateNoise, report: Report) -> GateChannels:
    built = gate_channels(gate, [table.qubit(q) for q in gate.qubits])
    report.record_channels(built)
    return built


def idle_channel(
    table: NoiseTable, qubit: int, duration_ns: float, report: Report, *, label: str = "delay"
) -> ChannelSpec | None:
    noise = table.qubit(qubit)
    if noise.relaxation_unknown:
        report.mark_unknown(f"T1 and T2 of qubit {qubit} (no {label} relaxation)")
        return None
    if noise.t2_clamped:
        report.record_t2_clamp(qubit)
    if duration_ns <= 0:
        return None
    kraus = thermal_relaxation_kraus(
        noise.t1_ns, noise.t2_ns, duration_ns, noise.dephasing_rate_per_s
    )
    return ChannelSpec("thermal_relaxation", (qubit,), tuple(kraus))
