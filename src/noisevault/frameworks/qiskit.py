"""Qiskit export: an AerSimulator with a device Target, a physical-qubit noise model and a report.

``to_qiskit(profile)`` returns a :class:`NoiseVaultSimulator`. Its Target lists every native
gate on every allowed locus with the error and duration the simulator applies, so
``transpile(circuit, sim)`` routes around disabled gates and places circuits noise-aware. Its
NoiseModel is keyed on physical qubits and the Target's instruction names, with the channels
the shared conversion rules assign (so noise follows the qubits a circuit actually lands on).

Two profile natives have no Qiskit instruction of their own and are exported under the
parametric gate that contains them: ``zz`` (exp(-i pi/4 ZZ)) as ``rzz`` and ``ms`` (IonQ
Molmer-Sorensen, MS(0, 0) = RXX(pi/2)) as ``rxx``. Aer keys noise on instruction names, so the
alias gets the native's noise at any angle; the report says so. When the profile also defines
the Qiskit gate itself, that definition wins.

Google's ``sqrt_iswap`` has no Qiskit gate either, and no parametric one transpile can target,
so it is exported as :class:`SqrtISwapGate`. Importing this module adds a cx -> sqrt_iswap rule
to Qiskit's session equivalence library, which is how ``transpile`` reaches that gate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from itertools import permutations, product
from math import pi
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import NoiseVaultError, install_hint

try:
    import qiskit
    import qiskit_aer
except ImportError as exc:
    raise ImportError(
        f"the Qiskit export needs Qiskit and Qiskit Aer: {install_hint('qiskit')}"
    ) from exc

from qiskit.circuit import Delay, Gate, Measure, Parameter, QuantumCircuit, Reset
from qiskit.circuit import library as qiskit_gates
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import CXGate, PauliGate, UnitaryGate, XXPlusYYGate
from qiskit.providers import QubitProperties
from qiskit.quantum_info import Kraus
from qiskit.transpiler import InstructionProperties, PassManager, Target
from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration
from qiskit_aer.library import (
    SetDensityMatrix,
    SetMatrixProductState,
    SetStabilizer,
    SetStatevector,
    SetSuperOp,
    SetUnitary,
)
from qiskit_aer.library.save_instructions.save_data import SaveData
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, kraus_error, pauli_error
from qiskit_aer.noise.passes import LocalNoisePass

from .. import gates
from ..channels import ChannelSpec, GateChannels, readout_matrix, thermal_relaxation_kraus
from ..conversion import UnknownGates, resolve_op
from ..report import Report
from ..table import GateNoise, NoiseTable, QubitNoise, Unavailable

if TYPE_CHECKING:
    from ..profile import Profile

# Canonical natives with no Qiskit instruction, exported as the gate that contains them.
ALIASES = {"zz": "rzz", "ms": "rxx"}


class SqrtISwapGate(Gate):
    """Google's ``sqrt_iswap`` (iSWAP**0.5), which is Qiskit's ``XXPlusYYGate(-pi/2, 0)``."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__("sqrt_iswap", 2, [], label=label)

    def _define(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.append(XXPlusYYGate(-pi / 2, 0), [0, 1])
        self.definition = circuit

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        # Symmetric, so the registry's big-endian matrix is also Qiskit's little-endian one.
        return np.asarray(gates.GATES["sqrt_iswap"].unitary(), dtype=dtype)


def _cx_via_sqrt_iswap() -> QuantumCircuit:
    circuit = QuantumCircuit(2, global_phase=pi)
    circuit.u(3 * pi / 4, -pi, pi / 2, 1)
    circuit.u(pi / 2, pi / 2, -pi / 4, 0)
    circuit.append(SqrtISwapGate(), [0, 1])
    circuit.u(pi, -3 * pi / 2, pi / 2, 0)
    circuit.append(SqrtISwapGate(), [0, 1])
    circuit.u(pi / 2, -pi / 4, pi / 2, 0)
    circuit.u(3 * pi / 4, 3 * pi / 2, pi, 1)
    return circuit


SessionEquivalenceLibrary.add_equivalence(CXGate(), _cx_via_sqrt_iswap())
# Registry natives Qiskit has no gate for, exported as this module's own.
_OWN_GATES: dict[str, type[Gate]] = {"sqrt_iswap": SqrtISwapGate}
# Simulator directives that act on no device resource.
_DIRECTIVES = (
    SaveData,
    SetDensityMatrix,
    SetMatrixProductState,
    SetStabilizer,
    SetStatevector,
    SetSuperOp,
    SetUnitary,
)
_TRANSPILE_FIX = (
    "transpile it for this simulator first: `from qiskit import transpile;"
    " sim.run(transpile(circuit, sim))`"
)
_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


class CircuitNotNativeError(NoiseVaultError, ValueError):
    """A circuit uses an instruction or a qubit pair the exported device does not provide."""


class UnsupportedDevice(NoiseVaultError, ValueError):
    """The profile leaves no one- or two-qubit native that Qiskit can compile circuits to."""


@dataclass(frozen=True)
class Export:
    """One profile native as a Qiskit instruction."""

    canonical: str
    gate: qiskit.circuit.Gate  # parametric where the gate has parameters

    @property
    def name(self) -> str:
        return self.gate.name


@dataclass(frozen=True)
class Placement:
    """A native on one allowed physical locus with the channels the simulator applies there."""

    export: Export
    qargs: tuple[int, ...]
    built: GateChannels
    events: tuple[tuple[str, str], ...]  # report events one application of it counts


class NoiseVaultSimulator(AerSimulator):
    """An AerSimulator for one profile: ``.target``, ``.noise_model``, ``.report``, ``.profile``.

    ``run`` accepts circuits already transpiled for this simulator (``transpile(circuit,
    sim)``): every instruction must be a native on a locus the Target allows. Delays get
    thermal relaxation (plus the profile's dephasing rate) on the delayed qubit.
    """

    def __init__(
        self,
        *,
        profile: Profile,
        report: Report,
        target: Target,
        noise_model: NoiseModel,
        placements: Sequence[Placement],
        **options: Any,
    ) -> None:
        configuration = AerBackendConfiguration(
            backend_name=f"noisevault_{profile.id}",
            backend_version=qiskit_aer.__version__,
            n_qubits=target.num_qubits,
            basis_gates=sorted(target.operation_names),
            gates=[],
            max_shots=int(1e6),
            coupling_map=None,
            description=target.description,
        )
        super().__init__(
            configuration=configuration, target=target, noise_model=noise_model, **options
        )
        self.profile = profile
        self.report = report
        self._events = {(p.export.name, p.qargs): p.events for p in placements}
        self._passes = PassManager(
            [
                LocalNoisePass(_delay_relaxation(profile.table, report), op_types=Delay),
                *_aer_unitary_passes(target),
            ]
        )

    @property
    def noise_model(self) -> NoiseModel:
        return self.options.noise_model

    def run(self, circuits: Any, parameter_binds: Any = None, **run_options: Any) -> Any:
        single = isinstance(circuits, QuantumCircuit)
        batch = [circuits] if single else list(circuits)
        for circuit in batch:
            self._check(circuit)
        for circuit in batch:
            self._count_events(circuit)
        prepared = [self._passes.run(circuit) for circuit in batch]
        return super().run(prepared[0] if single else prepared, parameter_binds, **run_options)

    def _check(self, circuit: QuantumCircuit) -> None:
        target = self.target
        if circuit.num_qubits > target.num_qubits:
            raise CircuitNotNativeError(
                f"circuit {circuit.name!r} has {circuit.num_qubits} qubits but {self.profile.id}"
                f" has {target.num_qubits}; {_TRANSPILE_FIX}"
            )
        for instruction in circuit.data:
            op = instruction.operation
            if op.name == "barrier" or isinstance(op, _DIRECTIVES):
                continue
            qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)
            if not target.instruction_supported(op.name, qargs):
                raise CircuitNotNativeError(
                    f"circuit {circuit.name!r}: {op.name} on qubits {list(qargs)} is not"
                    f" available on {self.profile.id} ({self._why(op.name, qargs)}); "
                    + _TRANSPILE_FIX
                )

    def _why(self, name: str, qargs: tuple[int, ...]) -> str:
        target = self.target
        if name not in target.operation_names:
            natives = sorted(set(target.operation_names) - {"measure", "reset", "delay"})
            return f"not a native instruction; natives are {', '.join(natives)}"
        if any(self.profile.table.qubit(q).disabled for q in qargs):
            return "a qubit there is disabled in the profile"
        return f"the device does not provide {name} on that locus"

    def _count_events(self, circuit: QuantumCircuit) -> None:
        for instruction in circuit.data:
            qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)
            for event, key in self._events.get((instruction.operation.name, qargs), ()):
                self.report.count(event, key)


def to_qiskit(
    profile: Profile, *, unknown_gates: UnknownGates = "typical", readout: bool = True
) -> NoiseVaultSimulator:
    """A noisy AerSimulator for ``profile``; transpile circuits for it before ``run``.

    The Target leaves out disabled qubits and gates the way Qiskit models faulty ones. Qiskit's
    ``optimization_level=0`` still puts circuit qubit i on physical qubit i, and levels 1-3 do
    not check that a chosen qubit has the 1-qubit gates a circuit needs. On a profile with
    disabled parts, transpile with ``initial_layout=list(profile.suggest_layout(n).values())``:
    it picks connected qubits that each have every 1-qubit native usable anywhere on the
    device, and warns when no chain of ``n`` such qubits exists.

    ``unknown_gates`` applies to natives with no error metric on a locus: ``"typical"``
    (default) gives them the typical native's noise (reported and warned), ``"error"`` leaves
    those loci out of the Target so transpile never uses them. ``readout=False`` leaves
    measurements noiseless. Raises UnsupportedDevice when no one- or no two-qubit native is
    left for Qiskit to compile to.
    """
    if not isinstance(readout, bool):
        raise ValueError(
            f"readout={readout!r}: pass True or False; Qiskit readout is exact"
            " (per-qubit P(1|0) and P(0|1))"
        )
    if unknown_gates not in ("typical", "error"):
        raise ValueError(f"unknown_gates={unknown_gates!r}: choose 'typical' or 'error'")
    report = Report.start(
        profile,
        "qiskit",
        f"{qiskit.__version__} (qiskit-aer {qiskit_aer.__version__})",
        unknown_gates=unknown_gates,
        readout=readout,
    )
    report.record_effects(profile.effects)
    table = profile.table
    enabled = [q for q in range(table.num_qubits) if not table.qubit(q).disabled]
    omitted: dict[str, str] = {}  # canonical native -> why the export leaves it out
    exports = _exports(profile, report, omitted)
    placements = _placements(table, exports, enabled, unknown_gates, report, omitted)
    for name, why in omitted.items():
        report.omit(f"native {name}: {why}")
    _require_natives(profile, placements, omitted, enabled)
    report.events.clear()  # locus bookkeeping; events count applications as circuits run
    target = _target(profile, placements, enabled)
    noise_model = _noise_model(table, placements, enabled, target, readout)
    _report_fixed(table, enabled, readout, report)
    return NoiseVaultSimulator(
        profile=profile,
        report=report,
        target=target,
        noise_model=noise_model,
        placements=placements,
    )


# natives ------------------------------------------------------------------------------------


def _exports(profile: Profile, report: Report, omitted: dict[str, str]) -> list[Export]:
    """Qiskit instructions for the profile's unitary natives, aliases after real gates."""
    table = profile.table
    names = [n for n in profile.gates if _is_unitary(n)]
    by_name: dict[str, Export] = {}
    for canonical in sorted(names, key=lambda n: n in ALIASES):
        arity = table.arity(canonical)
        gate = _qiskit_gate(canonical)
        if gate is None or arity is None or arity > 2:
            omitted[canonical] = "no Qiskit instruction for it in this export"
            continue
        if gate.name in by_name:
            omitted[canonical] = f"the profile's {gate.name} is exported instead"
            continue
        if canonical in ALIASES:
            report.approximate(
                f"gate {canonical}",
                f"exported as Qiskit {gate.name}",
                f"{gate.name} at any angle gets the calibrated noise of {canonical}",
            )
        if canonical == "sqrt_iswap":
            report.approximate(
                "gate count of transpiled circuits",
                "transpile reaches sqrt_iswap only through cx, two sqrt_iswap per cx",
                "a general two-qubit block can take 6 sqrt_iswap where 3 suffice; build"
                " circuits in sqrt_iswap directly, or use profile.to_cirq() to compile for Google",
            )
        by_name[gate.name] = Export(canonical, gate)
    return list(by_name.values())


def _is_unitary(name: str) -> bool:
    info = gates.lookup(name)
    return info is None or info.unitary is not None


def gate_class(name: str) -> type[Gate] | None:
    """The Qiskit gate class of registry gate ``name``: Qiskit's own, else this module's
    (:class:`SqrtISwapGate`), else None."""
    if name in _OWN_GATES:
        return _OWN_GATES[name]
    info = gates.lookup(name)
    if info is None or info.qiskit_class is None:
        return None
    return getattr(qiskit_gates, info.qiskit_class)


def _qiskit_gate(canonical: str) -> Gate | None:
    name = ALIASES.get(canonical, canonical)
    cls = gate_class(name)
    return None if cls is None else cls(*(Parameter(p) for p in gates.GATES[name].params))


def _placements(
    table: NoiseTable,
    exports: Iterable[Export],
    enabled: Sequence[int],
    unknown_gates: UnknownGates,
    report: Report,
    omitted: dict[str, str],
) -> list[Placement]:
    """Every native on every locus that gets noise; uncalibrated loci drop out in error mode."""
    loci = _loci(table, enabled)
    memo: dict[tuple, Placement] = {}
    out = []
    for export in exports:
        candidates = loci[export.gate.num_qubits]
        uncalibrated, unavailable = [], []
        for qargs in candidates:
            found = table.gate(export.canonical, qargs)
            if isinstance(found, Unavailable):
                unavailable.append(found.reason)
                continue
            if found.state == "disabled":
                continue
            if found.state == "uncalibrated" and unknown_gates == "error":
                uncalibrated.append(qargs)
                continue
            out.append(_placement(table, export, qargs, found, unknown_gates, report, memo))
        if uncalibrated:
            shown = [qargs[0] if len(qargs) == 1 else qargs for qargs in uncalibrated]
            omitted[export.canonical] = (
                f"no error metric on qubits {_span(shown)} and unknown_gates='error', so"
                " transpile does not use it there (unknown_gates='typical' gives it the"
                " typical native's noise)"
            )
        elif unavailable and len(unavailable) == len(candidates):
            omitted[export.canonical] = unavailable[0]
    return out


def _placement(
    table: NoiseTable,
    export: Export,
    qargs: tuple[int, ...],
    found: GateNoise,
    unknown_gates: UnknownGates,
    report: Report,
    memo: dict[tuple, Placement],
) -> Placement:
    """resolve_op on one locus, reusing the channels of a locus with identical physics.

    All-to-all devices have n(n-1) ordered pairs per native, mostly with the same noise; building
    each pair's channels is what made large exports slow.
    """
    source = found if found.state != "uncalibrated" else table.typical(len(qargs), qargs)
    key = None
    if isinstance(source, GateNoise):
        operands = tuple(_qubit_physics(table.qubit(q)) for q in qargs)
        key = (export.canonical, _gate_physics(source), operands)
        hit = memo.get(key)
        if hit is not None:
            built = _relabel(hit.built, source, qargs)
            report.record_channels(built)
            return Placement(export, qargs, built, hit.events)
    before = _event_totals(report)
    built = resolve_op(table, export.canonical, qargs, unknown_gates=unknown_gates, report=report)
    events = tuple((_event_totals(report) - before).elements())
    placement = Placement(export, qargs, built, events)
    if key is not None:
        memo[key] = placement
    return placement


def _gate_physics(gate: GateNoise) -> tuple:
    """What gate_channels and the report read from a gate, apart from its qubits."""
    return (gate.gate, gate.state, gate.avg_infidelity, gate.pauli, gate.duration_ns, gate.origin)


def _qubit_physics(qubit: QubitNoise) -> tuple:
    return (qubit.t1_ns, qubit.t2_ns, qubit.dephasing_rate_per_s)


def _relabel(built: GateChannels, gate: GateNoise, qargs: tuple[int, ...]) -> GateChannels:
    """``built`` moved from its gate's qubits onto ``qargs``, operand by operand."""
    moved = dict(zip(built.gate.qubits, qargs, strict=True))
    channels = tuple(replace(c, wires=tuple(moved[w] for w in c.wires)) for c in built.channels)
    clamped = tuple(moved[q] for q in built.t2_clamped)
    return replace(built, gate=gate, channels=channels, t2_clamped=clamped)


def _require_natives(
    profile: Profile,
    placements: Sequence[Placement],
    omitted: dict[str, str],
    enabled: Sequence[int],
) -> None:
    """Refuse a simulator that could not run any circuit needing a gate of some arity."""
    table = profile.table
    for arity, word in ((1, "one"), (2, "two")):
        defined = [n for n in profile.gates if _is_unitary(n) and table.arity(n) == arity]
        if arity > len(enabled) or not defined:
            continue
        if any(p.export.gate.num_qubits == arity for p in placements):
            continue
        loci = _loci(table, enabled)[arity]
        refusal = f"{profile.id} has no {word}-qubit native gate this Qiskit export can compile to"
        if not loci:
            raise UnsupportedDevice(
                f"{refusal}: its connectivity allows no pair of enabled qubits, so"
                f" profile.to_cirq() cannot run a {word}-qubit gate either"
            )
        why = "; ".join(f"{n}: {omitted.get(n, 'disabled on every locus')}" for n in defined)
        if any(table.allowed(n, qargs) for n in defined for qargs in loci):
            advice = (
                "Simulate it with profile.to_cirq() instead, or give the profile a calibrated"
                f" {word}-qubit native that Qiskit provides"
            )
        else:
            unit = "enabled qubit" if arity == 1 else "pair of enabled qubits"
            advice = (
                f"The profile allows no {word}-qubit native on any {unit}, so profile.to_cirq()"
                " cannot run one either"
            )
        raise UnsupportedDevice(f"{refusal} ({why}). {advice}")


def _loci(table: NoiseTable, enabled: Sequence[int]) -> dict[int, Sequence[tuple[int, ...]]]:
    return {1: [(q,) for q in enabled], 2: _pairs(table, enabled)}


def _pairs(table: NoiseTable, enabled: Sequence[int]) -> list[tuple[int, int]]:
    """Ordered pairs that may carry a 2-qubit gate; the table decides which really do."""
    if table.all_to_all:
        return list(permutations(enabled, 2))
    usable = set(enabled)
    listed = [pair for pair in table.listed_pairs() if usable.issuperset(pair)]
    return sorted(p for a, b in listed for p in ((a, b), (b, a)))


def _event_totals(report: Report) -> Counter[tuple[str, str]]:
    return Counter({(e, k): n for e, counts in report.events.items() for k, n in counts.items()})


# target -------------------------------------------------------------------------------------


def _target(profile: Profile, placements: Sequence[Placement], enabled: Sequence[int]) -> Target:
    table = profile.table
    target = Target(
        description=f"NoiseVault {profile.id} (nv:{profile.fingerprint[:12]})",
        num_qubits=table.num_qubits,
        qubit_properties=[_qubit_properties(table, q) for q in range(table.num_qubits)],
    )
    by_gate: dict[str, tuple[qiskit.circuit.Gate, dict]] = {}
    for p in placements:
        props = by_gate.setdefault(p.export.name, (p.export.gate, {}))[1]
        props[p.qargs] = InstructionProperties(error=p.built.achieved, duration=_seconds(p.built))
    for gate, props in by_gate.values():
        target.add_instruction(gate, props)
    measure = {(q,): _measure_properties(profile, q) for q in enabled}
    target.add_instruction(Measure(), measure)
    target.add_instruction(Reset(), {(q,): None for q in enabled})
    target.add_instruction(Delay(Parameter("t")), {(q,): None for q in enabled})
    return target


def _seconds(built: GateChannels) -> float | None:
    if built.gate.duration_ns is None:
        return 0.0 if built.gate.state == "ideal" else None  # virtual gates take no time
    return built.gate.duration_ns * 1e-9


def _qubit_properties(table: NoiseTable, q: int) -> QubitProperties:
    noise = table.qubit(q)
    t1 = None if noise.t1_ns is None else noise.t1_ns * 1e-9
    t2 = None if noise.t2_ns is None else noise.t2_ns * 1e-9
    return QubitProperties(t1=t1, t2=t2)


def _measure_properties(profile: Profile, q: int) -> InstructionProperties:
    pair = profile.table.qubit(q).readout
    record = next((r for r in profile.qubits if r.index == q and r.readout), None)
    spec = record.readout if record else profile.readout
    duration = spec.duration_ns if spec and spec.duration_ns is not None else None
    return InstructionProperties(
        error=None if pair is None else (pair[0] + pair[1]) / 2,
        duration=None if duration is None else duration * 1e-9,
    )


# noise model --------------------------------------------------------------------------------


def _noise_model(
    table: NoiseTable,
    placements: Sequence[Placement],
    enabled: Sequence[int],
    target: Target,
    readout: bool,
) -> NoiseModel:
    model = NoiseModel(basis_gates=sorted({*target.operation_names, "unitary"}))
    cache: dict[tuple, QuantumError] = {}
    errors: dict[str, dict[tuple[int, ...], QuantumError]] = {}
    for p in placements:
        if p.built.channels:
            error = gate_error(p.built.channels, p.qargs, cache)
            errors.setdefault(p.export.name, {})[p.qargs] = error
    everywhere = {1: len(enabled), 2: len(enabled) * (len(enabled) - 1)}
    for name, by_qargs in errors.items():
        distinct = {id(error) for error in by_qargs.values()}
        if len(distinct) == 1 and len(by_qargs) == everywhere[len(next(iter(by_qargs)))]:
            # A uniform device: one entry instead of one per ordered pair.
            model.add_all_qubit_quantum_error(next(iter(by_qargs.values())), name)
            continue
        for qargs, error in by_qargs.items():
            model.add_quantum_error(error, name, list(qargs))
    for q in enabled:
        noise = table.qubit(q)
        matrix = readout_matrix(noise)
        if readout and matrix is not None:
            # Aer rows are the prepared state: the transpose of M[measured, prepared].
            model.add_readout_error(ReadoutError(matrix.T), [q])
        if noise.prep_error:
            flip = noise.prep_error
            model.add_quantum_error(pauli_error([("X", flip), ("I", 1 - flip)]), "reset", [q])
    return model


def gate_error(
    channels: Sequence[ChannelSpec],
    qargs: Sequence[int],
    cache: dict[tuple, QuantumError] | None = None,
) -> QuantumError:
    """One QuantumError on ``qargs`` (Qiskit order) applying ``channels`` in order.

    Pauli mixtures become separate noise circuits, so Aer samples them instead of applying
    Kraus operators; any other channel is one Kraus instruction in every circuit.
    """
    position = {q: i for i, q in enumerate(qargs)}
    key = tuple(
        (tuple(position[w] for w in c.wires), tuple(k.tobytes() for k in c.kraus)) for c in channels
    )
    if cache is not None and key in cache:
        return cache[key]
    terms = [(QuantumCircuit(len(qargs)), 1.0)]
    for channel in channels:
        places = [position[w] for w in channel.wires]
        paulis = _pauli_terms(channel.kraus)
        if paulis is None:
            kraus = Kraus([little_endian(k) for k in channel.kraus]).to_instruction()
            for circuit, _ in terms:
                circuit.append(kraus, places)
            continue
        terms = [
            (_with_pauli(circuit, label, places), p * q)
            for circuit, p in terms
            for label, q in paulis
            if q > 0
        ]
    error = QuantumError(terms)
    if cache is not None:
        cache[key] = error
    return error


def _with_pauli(circuit: QuantumCircuit, label: str, places: list[int]) -> QuantumCircuit:
    out = circuit.copy()
    out.append(PauliGate(label), places)
    return out


def _pauli_terms(kraus: Sequence[np.ndarray]) -> list[tuple[str, float]] | None:
    """Qiskit-labelled (label, probability) terms if every Kraus operator is a scaled Pauli."""
    n = int(np.log2(kraus[0].shape[0]))
    labels, basis = _pauli_basis(n)
    terms = []
    for op in kraus:
        coefficients = np.einsum("pij,ji->p", basis, op) / 2**n
        best = int(np.argmax(np.abs(coefficients)))
        if not np.allclose(op, coefficients[best] * basis[best], rtol=0, atol=1e-14):
            return None
        terms.append((labels[best], float(abs(coefficients[best]) ** 2)))
    return terms


@cache
def _pauli_basis(n: int) -> tuple[list[str], np.ndarray]:
    """Pauli strings as Qiskit labels (read right to left) and their big-endian matrices."""
    labels, matrices = [], []
    for letters in product("IXYZ", repeat=n):
        matrix = np.eye(1, dtype=complex)
        for c in letters:
            matrix = np.kron(matrix, _PAULI[c])
        labels.append("".join(letters)[::-1])
        matrices.append(matrix)
    return labels, np.array(matrices)


def little_endian(op: np.ndarray) -> np.ndarray:
    """A big-endian operator on wires (w0, w1, ...) as Qiskit's matrix over qubits in that order."""
    n = int(np.log2(op.shape[0]))
    axes = [*reversed(range(n)), *reversed(range(n, 2 * n))]
    return np.asarray(op, dtype=complex).reshape((2,) * (2 * n)).transpose(axes).reshape(op.shape)


# circuit passes -----------------------------------------------------------------------------


def _delay_relaxation(
    table: NoiseTable, report: Report
) -> Callable[[Delay, Sequence[int]], QuantumError | None]:
    """LocalNoisePass callback: thermal relaxation and dephasing over a delay's duration."""

    def relax(op: Delay, qubits: Sequence[int]) -> QuantumError | None:
        (q,) = qubits
        noise = table.qubit(q)
        duration = _delay_ns(op, q)
        if noise.relaxation_unknown:
            report.mark_unknown(f"T1 and T2 of qubit {q} (no delay relaxation)")
            return None
        kraus = thermal_relaxation_kraus(
            noise.t1_ns, noise.t2_ns, duration, noise.dephasing_rate_per_s
        )
        return None if len(kraus) == 1 else kraus_error(kraus)

    return relax


_TIME_UNITS_NS = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0, "ps": 1e-3}


def _delay_ns(op: Delay, q: int) -> float:
    if isinstance(op.duration, qiskit.circuit.ParameterExpression):
        raise CircuitNotNativeError(
            f"delay on qubit {q} has the unbound duration {op.duration}; delays relax before"
            " parameter_binds apply, so bind delay durations before run:"
            " sim.run(circuit.assign_parameters({...}))"
        )
    if op.unit not in _TIME_UNITS_NS:
        raise CircuitNotNativeError(
            f"delay on qubit {q} has duration {op.duration} {op.unit}; the profile has no sample"
            " time, so give delays a time unit (s, ms, us, ns, ps), e.g."
            " qc.delay(100, q, unit='ns')"
        )
    return float(op.duration) * _TIME_UNITS_NS[op.unit]


def _aer_unitary_passes(target: Target) -> list[LocalNoisePass]:
    """Natives no Aer method executes by name run as labelled unitaries, which keep their noise.

    Only for gates outside every method (iswap): Aer 0.17 mis-simulates labelled unitaries
    whose matrix is a non-symmetric permutation with phases (ecr, cy), so gates a method merely
    lacks are left for Aer to reject with its own message.
    """
    supported = {*AerSimulator().configuration().basis_gates, "measure", "reset", "delay"}
    missing = tuple(
        target.operation_from_name(name).base_class
        for name in target.operation_names
        if name not in supported
    )
    if not missing:
        return []

    def as_unitary(op: qiskit.circuit.Gate, qubits: Sequence[int]) -> UnitaryGate:
        return UnitaryGate(op.to_matrix(), label=op.name)

    return [LocalNoisePass(as_unitary, op_types=missing, method="replace")]


# report -------------------------------------------------------------------------------------


def _report_fixed(table: NoiseTable, enabled: Sequence[int], readout: bool, report: Report) -> None:
    qubits = [table.qubit(q) for q in enabled]
    report.mark_exact(
        "gate noise: channels per exported native and physical locus (Aer QuantumError)"
    )
    no_readout = [q.index for q in qubits if q.readout is None]
    if not readout:
        report.omit("readout error (readout=False)")
    elif len(no_readout) < len(qubits):
        report.mark_exact("readout: P(1|0) and P(0|1) per qubit, as given (Aer ReadoutError)")
    if readout and no_readout:
        report.mark_unknown(f"readout error of qubits {_span(no_readout)}")
    no_prep = [q.index for q in qubits if q.prep_error is None]
    if len(no_prep) < len(qubits):
        report.mark_exact("reset: bit flip with the preparation error after each reset")
        report.approximate("initial state", "ideal |0>", "preparation error applies after reset")
    if no_prep:
        report.mark_unknown(f"preparation (reset) error of qubits {_span(no_prep)}")
    if not all(q.relaxation_unknown for q in qubits):
        report.mark_exact("delay: thermal relaxation and dephasing over its duration")
    for q in qubits:
        if q.t2_clamped:
            report.record_t2_clamp(q.index)
    report.omit(
        "idle time outside explicit delays (insert delays with"
        " transpile(circuit, sim, scheduling_method='alap'))"
    )


def _span(indices: Sequence[object]) -> str:
    shown = ", ".join(map(str, indices[:8]))
    return f"[{shown}]" if len(indices) <= 8 else f"[{shown}, ...] ({len(indices)} qubits)"
