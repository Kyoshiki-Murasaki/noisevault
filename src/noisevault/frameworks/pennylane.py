"""PennyLane export: a ``qml.NoiseModel`` that carries its conversion report.

Use it with ``qml.add_noise(qnode, model)`` on ``default.mixed``. Every gate gets the channels the
shared conversion rules assign to it on its physical qubits, as ``qml.QubitChannel`` operations
after the gate. Readout confusion goes right before each computational-basis measurement, on every
wire the circuit touches so that all such measurements share one simulation, and before each
Pauli measurement in its measured basis. A measurement without wires (``qml.probs()``,
``qml.sample()``, ``qml.counts()``) reads every device wire; it gets readout confusion on the
wires the circuit's operations touch, because a noise model never sees the device's wires.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the PennyLane export needs PennyLane: pip install 'noisevault[pennylane]'"
    ) from exc

from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)
from pennylane.operation import Channel, Operation, Operator, StatePrepBase
from pennylane.ops.op_math import Conditional

from .. import gates
from ..channels import readout_matrix
from ..conversion import UnknownGates, resolve_op
from ..errors import LayoutError
from ..layout import normalize_layout
from ..profile import Profile
from ..report import Report

Layout = Mapping[Hashable, int] | Sequence[int]
Kraus = tuple[np.ndarray, ...]
PhysicalChannel = tuple[Kraus, tuple[int, ...]]  # Kraus operators (big-endian) and their qubits
EventCounts = tuple[tuple[str, str, int], ...]  # (event, key, count) added to the report
CacheEntry = tuple[tuple[PhysicalChannel, ...], EventCounts]

_CANONICAL = {info.pennylane: info.name for info in gates.GATES.values() if info.pennylane}
_NOT_GATES = frozenset({"Barrier", "Snapshot", "GlobalPhase", "WireCut"})
_READOUT_MEASUREMENTS = (ExpectationMP, VarianceMP, ProbabilityMP, SampleMP, CountsMP)
_ROTATED_PAULIS = {"X": qml.PauliX, "Y": qml.PauliY}
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
        self._tape_wires: dict[Hashable, None] = {}  # ordered set of wires the current tape uses
        gate_map = {
            qml.BooleanFn(_any_op, "NoiseVaultWires"): self._touch,
            qml.BooleanFn(_is_gate, "NoiseVaultGate"): self._gate_noise,
            qml.BooleanFn(_is_reset, "NoiseVaultReset"): self._reset_noise,
        }
        meas_map = {qml.BooleanFn(_reads_out, "NoiseVaultReadout"): self._readout_noise}
        super().__init__(gate_map, meas_map=meas_map if readout else None)

    @property
    def model_map(self) -> dict:
        # qml.add_noise reads model_map once at the start of every tape, before any noise function
        # runs: the one point where the wires of the previous tape can be forgotten.
        self._tape_wires = {}
        return super().model_map

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

    def _touch(self, op: Operator, **_: Any) -> None:
        """Record the wires of every operation; add_noise runs this before any measurement."""
        self._tape_wires.update(dict.fromkeys(op.wires))

    def _gate_noise(self, op: Operator, **_: Any) -> None:
        gate = _unconditional(op)
        if gate is not op:
            self.report.approximate(
                "conditional gates",
                "gate noise applied whether or not the condition holds",
                "default.mixed cannot condition a channel on a mid-circuit measurement",
            )
        physical = tuple(self.physical_qubit(w) for w in op.wires)
        wire_of = dict(zip(physical, op.wires, strict=True))
        for kraus, qubits in self._channels(_CANONICAL.get(gate.name, gate.name), physical):
            qml.QubitChannel(list(kraus), wires=[wire_of[q] for q in qubits])

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
                "applied to every wire the circuit's operations touch",
                "a device wire no operation touches reads out without error; pass wires="
                " to give it readout error",
            )
        self._tape_wires.update(dict.fromkeys(mp.wires))
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
        for wire in mp.wires or self._tape_wires:
            if self._readout_matrix(wire) is None:
                self.report.mark_unknown(f"readout on qubit {self.physical_qubit(wire)}")
        # Confusion on an unmeasured wire leaves the measured marginals alone, and identical
        # noise lets add_noise keep all computational-basis measurements on one tape.
        wires = mp.wires if basis else list(self._tape_wires)
        with qml.QueuingManager.stop_recording():
            undo = [qml.adjoint(gate, lazy=False) for gate in reversed(basis)]
        for gate in basis:
            qml.apply(gate)
        for wire in wires:
            matrix = self._readout_matrix(wire)
            if matrix is not None and not np.array_equal(matrix, np.eye(2)):
                qml.QubitChannel(confusion_kraus(matrix), wires=wire)
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
    """
    return NoiseVaultPennyLaneModel(
        profile, layout=layout, unknown_gates=unknown_gates, readout=readout
    )


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


def _labels(layout: Layout) -> list[Hashable]:
    return list(layout) if isinstance(layout, Mapping) else list(range(len(layout)))


def _unconditional(op: Operator) -> Operator:
    return op.base if isinstance(op, Conditional) else op


def _any_op(op: Operator) -> bool:
    return True


def _is_gate(op: Operator) -> bool:
    op = _unconditional(op)
    return (
        isinstance(op, Operation)
        and not isinstance(op, Channel | StatePrepBase)
        and op.name not in _NOT_GATES
    )


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
            for gate in _ROTATED_PAULIS[letter](wire).diagonalizing_gates()
        )


def _added_events(before: dict[str, Counter[str]], after: dict[str, Counter[str]]) -> EventCounts:
    return tuple(
        (event, what, n)
        for event, counts in after.items()
        for what, n in (counts - before.get(event, Counter())).items()
    )
