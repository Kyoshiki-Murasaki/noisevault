"""PennyLane export: a ``qml.NoiseModel`` that carries its conversion report.

Use it with ``qml.add_noise(qnode, model)`` on ``default.mixed``. Every gate gets the channels the
shared conversion rules assign to it on its physical qubits, as ``qml.QubitChannel`` operations
after the gate. Readout confusion goes right before each computational-basis measurement, on every
wire the circuit's operations or measurements use so that all such measurements share one
simulation, and before each Pauli measurement in its measured basis. A measurement without wires
(``qml.probs()``, ``qml.sample()``, ``qml.counts()``) reads every device wire; it gets readout
confusion on the wires the circuit uses, because a noise model never sees the device's wires.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable, Container, Hashable, Mapping, Sequence
from math import pi
from types import FrameType
from typing import Any, NamedTuple

import numpy as np

from ..errors import LayoutError, install_hint

try:
    import pennylane as qml
except ImportError as exc:
    raise ImportError(f"the PennyLane export needs PennyLane: {install_hint('pennylane')}") from exc

from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)
from pennylane.operation import Channel, Operation, Operator, StatePrepBase
from pennylane.ops.op_math import Adjoint, CompositeOp, Conditional, SymbolicOp

from .. import gates
from ..channels import readout_matrix
from ..conversion import UnknownGates, native_name, resolve_op
from ..layout import normalize_layout
from ..profile import Profile
from ..report import Report

Layout = Mapping[Hashable, int] | Sequence[int]
Kraus = tuple[np.ndarray, ...]
PhysicalChannel = tuple[Kraus, tuple[int, ...]]  # Kraus operators (big-endian) and their qubits
EventCounts = tuple[tuple[str, str, int], ...]  # (event, key, count) added to the report
CacheEntry = tuple[tuple[PhysicalChannel, ...], EventCounts]


class Charge(NamedTuple):
    gate: str
    wires: tuple[Hashable, ...]


_CANONICAL = {info.pennylane: info.name for info in gates.GATES.values() if info.pennylane}
# Registry gates PennyLane has only as another operation at a fixed angle: operation name ->
# (gate, the angle read from the operation's parameters, the values that make it that gate).
# Angles compare modulo 2 pi, where both operations repeat up to a global phase.
_AT_ANGLE: dict[str, tuple[str, Callable[[Sequence[Any]], Any], tuple[float, ...]]] = {
    "IsingZZ": ("zz", lambda p: p[0], (pi / 2,)),
    "IsingXX": ("ms", lambda p: p[0], (pi / 2, -pi / 2)),  # ms(0, 0) and ms(pi, 0)
    "IsingYY": ("ms", lambda p: p[0], (pi / 2, -pi / 2)),  # ms(pi/2, pi/2) and ms(pi/2, -pi/2)
    "Rot": ("r", lambda p: p[0] + p[2], (0.0,)),  # Rot(a, theta, -a) = r(theta, pi/2 - a)
}
_BUILDERS: dict[str, Callable[..., Operator]] = {
    "zz": lambda wires: qml.IsingZZ(pi / 2, wires=wires),
    "r": lambda theta, phi, wires: qml.Rot(pi / 2 - phi, theta, phi - pi / 2, wires=wires),
}
_ANGLE_TOL = 1e-9
_NOT_GATES = frozenset({"Barrier", "Snapshot", "GlobalPhase", "WireCut"})
_READOUT_MEASUREMENTS = (ExpectationMP, VarianceMP, ProbabilityMP, SampleMP, CountsMP)
_ROTATED_PAULIS = {"X": qml.PauliX, "Y": qml.PauliY}
_ADD_NOISE = ("pennylane.noise.add_noise", "add_noise")  # module and name of its tape transform
_NO_BASIS_FIX = (
    "measure Pauli words or computational-basis probabilities; for a Hamiltonian, wrap the"
    " QNode in qml.transforms.split_non_commuting before qml.add_noise"
)


class NoiseVaultPennyLaneModel(qml.NoiseModel):
    """A PennyLane noise model built from a NoiseVault profile.

    ``.report`` says what the conversion reproduced, approximated or omitted and keeps
    counting events as circuits run; ``.profile`` is the source profile.
    """

    def __init__(
        self,
        profile: Profile,
        *,
        layout: Layout | None = None,
        unknown_gates: UnknownGates = "typical",
        readout: bool = True,
    ) -> None:
        if unknown_gates not in ("typical", "error"):
            raise ValueError(f"unknown_gates={unknown_gates!r}: choose 'typical' or 'error'")
        self.profile = profile
        self.report = Report.start(
            profile,
            "pennylane",
            qml.__version__,
            layout=layout,
            unknown_gates=unknown_gates,
            readout=readout,
        )
        self.report.record_effects(profile.effects)
        _describe(self.report, readout)
        self._unknown_gates: UnknownGates = unknown_gates
        self._explicit_layout = layout is not None
        self._list_layout = self._explicit_layout and not isinstance(layout, Mapping)
        self._layout: dict[Hashable, int] = (
            {} if layout is None else normalize_layout(_labels(layout), layout, profile)
        )
        self._cache: dict[tuple[str, tuple[int, ...]], CacheEntry] = {}
        gate_map = {
            qml.BooleanFn(_is_operation, "NoiseVaultWireCheck"): self._check_wires,
            qml.BooleanFn(_is_gate, "NoiseVaultGate"): self._gate_noise,
            qml.BooleanFn(_is_reset, "NoiseVaultReset"): self._reset_noise,
        }
        meas_map = {qml.BooleanFn(_reads_out, "NoiseVaultReadout"): self._readout_noise}
        super().__init__(gate_map, meas_map=meas_map if readout else None)

    @property
    def model_map(self) -> dict:
        # add_noise reads this for every tape, so a tape with no operation and no readout
        # callback (measurements only, or readout=False) still has its wires checked.
        if _add_noise_frame() is not None:
            self._noised_tape()
        return super().model_map

    def _noised_tape(self) -> qml.tape.QuantumScript:
        """The tape qml.add_noise is noising, after checking every wire it uses against the layout.

        Noise functions see one operation or measurement, but readout needs the whole tape and only
        add_noise's frame holds it. Each call finds it there instead of keeping it on the model,
        because a composed model is a new plain qml.NoiseModel that shares only these functions.
        """
        frame = _add_noise_frame()
        if frame is None:
            raise RuntimeError(
                f"NoiseVault noise ran outside qml.add_noise (PennyLane {qml.__version__}),"
                " where it cannot see the circuit's wires; use qml.add_noise(qnode, model)"
            )
        # Its first two parameters: the tape and the model it applies. That model, not this one,
        # decides readout, because composing can remove or add a measurement map.
        tape, model = (frame.f_locals[name] for name in frame.f_code.co_varnames[:2])
        if model.meas_map and tape.shots.has_partitioned_shots:
            raise ValueError(
                "qml.add_noise keeps only part of a shot vector's results when readout noise is"
                " on; run each shot count separately or pass readout=False"
            )
        for wire in tape.wires:
            self.physical_qubit(wire)
        return tape

    def _check_wires(self, _: Operator, **__: Any) -> None:
        self._noised_tape()

    def physical_qubit(self, wire: Hashable) -> int:
        """The device qubit a circuit wire maps to; integer wire ``i`` is qubit ``i`` by default."""
        if wire not in self._layout:
            if self._explicit_layout:
                raise LayoutError(f"wire {wire!r} is not in the layout; {self._layout_fix(wire)}")
            self._layout.update(normalize_layout([wire], None, self.profile))
        return self._layout[wire]

    def _layout_fix(self, wire: Hashable) -> str:
        if self._list_layout:
            return f"a list layout covers wires 0 to {len(self._layout) - 1}; extend the list"
        return f"add it: layout={{..., {wire!r}: <physical qubit>}}"

    def _gate_noise(self, op: Operator, **_: Any) -> None:
        gate = _unconditional(op)
        if gate is not op:
            self.report.approximate(
                "conditional gates",
                "gate noise applied whether or not the condition holds",
                "default.mixed cannot condition a channel on a mid-circuit measurement",
            )
        noise: list[tuple[Kraus, list[Hashable]]] = []
        noisy_wires: set[Hashable] = set()
        moved = False
        for name, wires in _charges(gate, self.profile.gates):
            moved |= not noisy_wires.isdisjoint(wires)
            physical = tuple(self.physical_qubit(w) for w in wires)
            wire_of = dict(zip(physical, wires, strict=True))
            for kraus, qubits in self._channels(name, physical):
                targets = [wire_of[q] for q in qubits]
                noise.append((kraus, targets))
                noisy_wires.update(targets)
        if moved:
            self.report.approximate(
                "operator arithmetic",
                "the noise of each gate it decomposes into applied after the whole operator",
                "apply the gates one by one to put each gate's noise right after it",
            )
        for kraus, wires in noise:
            qml.QubitChannel(list(kraus), wires=wires)

    def _reset_noise(self, op: MidMeasureMP, **_: Any) -> None:
        qubit = self.physical_qubit(op.wires[0])
        error = self.profile.table.qubit(qubit).prep_error
        if error is None:
            self.report.mark_unknown(f"reset error on qubit {qubit}")
        elif error > 0:
            qml.BitFlip(error, wires=op.wires)

    def _channels(self, name: str, physical: tuple[int, ...]) -> tuple[PhysicalChannel, ...]:
        """Channels of one gate, cached; a cache hit replays the report events it produced.

        Channels do not depend on gate angles, so the key leaves them out and a trained
        circuit keeps hitting the cache as its parameters change.
        """
        key = (name, physical)
        if key in self._cache:
            channels, events = self._cache[key]
            for event, what, n in events:
                self.report.count(event, what, n)
            return channels
        before = {event: Counter(counts) for event, counts in self.report.events.items()}
        built = resolve_op(
            self.profile.table,
            name,
            physical,
            unknown_gates=self._unknown_gates,
            report=self.report,
        )
        channels = tuple((c.kraus, c.wires) for c in built.channels)
        self._cache[key] = (channels, _added_events(before, self.report.events))
        return channels

    def _readout_noise(self, mp: Any, **_: Any) -> None:
        if not mp.wires:
            self.report.approximate(
                "readout of measurements without wires",
                "applied to every wire the circuit's operations or measurements use",
                "a device wire the circuit never uses reads out without error; pass wires="
                " to give it readout error",
            )
        basis = _measured_basis(mp.obs)
        if basis is None:
            self.report.omit("readout on observables not measured in one product basis")
            self.report.warn_once(
                f"readout_skipped:{mp.obs}",
                f"no readout noise applied to {mp}: its observable is not measured in one"
                f" product basis. To fix: {_NO_BASIS_FIX}",
            )
            return
        if basis:
            self.report.approximate(
                "measurement basis change",
                "ideal rotation around the readout confusion",
                "add the rotation to the circuit to give it gate noise",
            )
        tape_wires = list(self._noised_tape().wires)
        for wire in mp.wires or tape_wires:
            if self._readout_matrix(wire) is None:
                self.report.mark_unknown(f"readout on qubit {self.physical_qubit(wire)}")
        # Confusion on an unmeasured wire leaves the measured marginals alone, and identical
        # noise lets add_noise keep all computational-basis measurements on one tape.
        wires = mp.wires if basis else tape_wires
        with qml.QueuingManager.stop_recording():
            undo = [qml.adjoint(gate, lazy=False) for gate in reversed(basis)]
        for gate in basis:
            qml.apply(gate)
        for wire in wires:
            matrix = self._readout_matrix(wire)
            if matrix is not None and not np.array_equal(matrix, np.eye(2)):
                qml.QubitChannel(confusion_kraus(matrix), wires=[wire])
        for gate in undo:
            qml.apply(gate)

    def _readout_matrix(self, wire: Hashable) -> np.ndarray | None:
        return readout_matrix(self.profile.table.qubit(self.physical_qubit(wire)))


def to_pennylane(
    profile: Profile,
    *,
    layout: Layout | None = None,
    unknown_gates: UnknownGates = "typical",
    readout: bool = True,
) -> NoiseVaultPennyLaneModel:
    """A noise model for ``qml.add_noise(qnode, model)`` on ``default.mixed``.

    ``layout`` maps circuit wires to physical qubits (a mapping, or a sequence where wire ``i``
    maps to ``layout[i]``); integer wires map to the same qubit by default, other wire labels
    need a layout. ``unknown_gates`` decides what gates the profile does not calibrate get:
    ``"typical"`` noise (warned and reported) or an error. ``readout=False`` leaves out
    readout errors.

    ``qml.add_noise`` at its default ``level="user"`` decomposes ``qml.adjoint`` gates and
    templates first, so they are noised gate by gate; pass ``level="top"`` to noise
    ``Adjoint(SX)``, ``Adjoint(S)`` and ``Adjoint(T)`` as the profile's sxdg, sdg and tdg.
    Operator arithmetic, such as ``qml.prod`` or ``@``, gets the noise of the gates it
    decomposes into, after the whole operator. Arithmetic with no such decomposition, such as
    ``qml.sum``, raises ValueError.

    ``qml.IsingZZ(pi/2)`` gets the noise of a profile's ``zz``, ``qml.IsingXX(+-pi/2)`` and
    ``qml.IsingYY(+-pi/2)`` that of its ``ms``, and ``qml.Rot(a, theta, -a)`` that of its ``r``
    (:func:`operation_for` builds zz and r); see :func:`gate_name`. Traced angles, as under
    ``jax.jit``, cannot be compared, so those operations then get ``rzz``, ``rxx`` or ``ryy``
    noise or the typical-noise rule. A broadcast whose angles call for different gates' noise
    raises ValueError; apply ``qml.transforms.broadcast_expand`` before ``qml.add_noise``.
    """
    return NoiseVaultPennyLaneModel(
        profile, layout=layout, unknown_gates=unknown_gates, readout=readout
    )


def gate_name(op: Operator, defined: Container[str] = ()) -> str:
    """The registry name of ``op`` under :func:`~noisevault.conversion.native_name`.

    An operation that equals a registry gate at its angle is named after it (``IsingZZ(pi/2)``
    as ``zz``, ``IsingXX(pi/2)`` and ``IsingYY(pi/2)`` as ``ms``, ``Rot(a, theta, -a)`` as
    ``r``), and a fixed gate on a profile with only the rotation it equals after the rotation
    (``SX`` as ``rx``); ``defined`` is a profile's gate names.
    """
    own = _CANONICAL.get(op.name, op.name)
    names = {
        native_name(own, defined) if native is None else native_name(native, defined, own)
        for native in _natives_at_angle(op)
    }
    if len(names) > 1:
        raise ValueError(
            f"{op.name} is broadcast over angles that get the noise of different gates"
            f" ({', '.join(sorted(names))}), and one operation takes one noise channel; expand"
            " the broadcast first: qml.add_noise(qml.transforms.broadcast_expand(qnode), model)"
        )
    return names.pop()


def operation_for(name: str) -> Callable[..., Operator] | None:
    """The PennyLane operation for registry gate ``name``, called with the gate's parameters
    and ``wires=``; None when PennyLane has none. ``operation_for("r")(theta, phi, wires=0)``
    is a ``qml.Rot`` with the unitary of ``r``, which the noise model recognizes as ``r``."""
    if name in _BUILDERS:
        return _BUILDERS[name]
    info = gates.lookup(name)
    return getattr(qml, info.pennylane, None) if info is not None and info.pennylane else None


def confusion_kraus(matrix: np.ndarray) -> list[np.ndarray]:
    """Kraus operators sqrt(M[j, i]) |j><i| that turn populations p into M p, for any column-
    stochastic M (including P(1|0) + P(0|1) > 1, which a generalized amplitude damping cannot)."""
    return [
        np.sqrt(matrix[j, i]) * np.outer(np.eye(2)[j], np.eye(2)[i])
        for i in range(2)
        for j in range(2)
        if matrix[j, i] > 0
    ]


def _describe(report: Report, readout: bool) -> None:
    report.mark_exact("gate errors")
    if readout:
        report.mark_exact("readout errors")
    else:
        report.omit("readout errors (readout=False)")
    report.approximate(
        "initial state",
        "ideal |0...0>",
        "state preparation operations (BasisState, StatePrep) are noiseless",
    )
    report.omit("idle time between gates (PennyLane circuits are not scheduled)")
    report.omit("readout on mid-circuit measurements")
    report.approximate(
        "adjoint gates and templates",
        "noised through their decomposition at qml.add_noise's default level='user'",
        "pass level='top' to qml.add_noise to noise Adjoint(SX), Adjoint(S), Adjoint(T) whole",
    )


def _natives_at_angle(op: Operator) -> set[str | None]:
    """The native each angle of ``op`` equals, None for an angle that equals none; a broadcast
    operation has one angle per element."""
    if op.name not in _AT_ANGLE:
        return {None}
    native, angle_of, values = _AT_ANGLE[op.name]
    angle = angle_of(op.parameters)
    if qml.math.is_abstract(angle):
        return {None}
    return {
        native if any(_same_angle(a, value) for value in values) else None
        for a in np.ravel(qml.math.toarray(angle))
    }


def _same_angle(a: float, b: float) -> bool:
    gap = (float(a) - b) % (2 * pi)
    return min(gap, 2 * pi - gap) < _ANGLE_TOL


def _add_noise_frame() -> FrameType | None:
    frame = sys._getframe(1)
    while frame and (frame.f_globals.get("__name__"), frame.f_code.co_name) != _ADD_NOISE:
        frame = frame.f_back
    return frame


def _labels(layout: Layout) -> list[Hashable]:
    return list(layout) if isinstance(layout, Mapping) else list(range(len(layout)))


def _unconditional(op: Operator) -> Operator:
    return op.base if isinstance(op, Conditional) else op


def _is_operation(_: Operator) -> bool:
    return True


def _is_gate(op: Operator) -> bool:
    op = _unconditional(op)
    return (
        isinstance(op, Operation | CompositeOp | SymbolicOp)
        and not isinstance(op, Channel | StatePrepBase)
        and op.name not in _NOT_GATES
    )


def _is_arithmetic(op: Operator) -> bool:
    """Operator arithmetic such as ``qml.prod``; a symbolic Operation such as CNOT is a gate."""
    return isinstance(op, CompositeOp | SymbolicOp) and not isinstance(op, Operation)


def _charges(op: Operator, defined: Container[str]) -> list[Charge]:
    charges: list[Charge] = []
    for gate in _decomposed(op) if _is_arithmetic(op) else [op]:
        name = gate_name(gate, defined)
        if isinstance(gate, qml.Identity):
            charges += [Charge(name, (wire,)) for wire in gate.wires]
        else:
            charges.append(Charge(name, tuple(gate.wires)))
    return charges


def _decomposed(op: Operator) -> list[Operator]:
    if not op.has_decomposition:
        raise ValueError(
            f"{op} has no decomposition into gates, so it gets no gate noise and default.mixed"
            " cannot run it; apply a unitary operator as"
            " qml.QubitUnitary(qml.matrix(op), wires=...)"
        )
    with qml.QueuingManager.stop_recording():
        parts = op.decomposition()
    return [
        gate
        for part in parts
        if _is_gate(part)
        for gate in (_decomposed(part) if _splits_at_user_level(part) else [part])
    ]


def _splits_at_user_level(op: Operator) -> bool:
    return _is_arithmetic(op) or (isinstance(op, Adjoint) and op.has_decomposition)


def _is_reset(op: Operator) -> bool:
    return isinstance(op, MidMeasureMP) and op.reset


def _reads_out(mp: Any) -> bool:
    return isinstance(mp, _READOUT_MEASUREMENTS) and getattr(mp, "mv", None) is None


def _measured_basis(obs: Operator | None) -> tuple[Operator, ...] | None:
    """Single-qubit rotations into the measured basis: () for the computational basis, None
    when the observable is not measured in one product basis (Pauli terms that disagree on a
    wire, or an entangled eigenbasis)."""
    if obs is None:
        return ()
    if obs.pauli_rep is not None:
        return _pauli_basis(obs.pauli_rep)
    try:
        with qml.QueuingManager.stop_recording():
            basis = tuple(obs.diagonalizing_gates())
    except qml.exceptions.DiagGatesUndefinedError:
        return None
    return basis if all(len(gate.wires) == 1 for gate in basis) else None


def _pauli_basis(words: qml.pauli.PauliSentence) -> tuple[Operator, ...] | None:
    """Rotations that make every +1 eigenstate read 0, from the Pauli letter of each wire.

    Taken from the letters rather than the observable's own diagonalizing gates, which for a
    sum follow eigh order and can map a -1 eigenstate to |0>.
    """
    letters: dict[Hashable, str] = {}
    for word in words:
        for wire, letter in word.items():
            if letters.setdefault(wire, letter) != letter:
                return None
    with qml.QueuingManager.stop_recording():
        return tuple(
            gate
            for wire, letter in letters.items()
            if letter != "Z"
            for gate in _ROTATED_PAULIS[letter]([wire]).diagonalizing_gates()
        )


def _added_events(before: dict[str, Counter[str]], after: dict[str, Counter[str]]) -> EventCounts:
    return tuple(
        (event, what, n)
        for event, counts in after.items()
        for what, n in (counts - before.get(event, Counter())).items()
    )
