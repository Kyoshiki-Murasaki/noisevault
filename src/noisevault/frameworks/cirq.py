"""Cirq export: a ``cirq.NoiseModel`` that adds a profile's noise to every operation.

Each gate is followed by the channels the shared conversion rules give it, as
``cirq.KrausChannel`` on the same qubits. Mid-circuit measurements get the readout assignment
error as a ``MeasurementGate`` confusion map and terminal ones as a channel just before them
(see :class:`NoiseVaultNoiseModel`), resets get the preparation error, and ``WaitGate`` gets
thermal relaxation. ``LineQubit(i)`` is device qubit ``i`` unless ``layout`` says otherwise;
``GridQubit(r, c)`` is the qubit at coords ``(r, c)`` when the profile records coords.
"""

from __future__ import annotations

import numbers
import re
from collections import defaultdict
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, get_args

import numpy as np

from ..errors import LayoutError, install_hint

try:
    import cirq
except ImportError as exc:
    raise ImportError(f"to_cirq needs Cirq; install it with: {install_hint('cirq')}") from exc

from .. import gates
from ..channels import readout_matrix, thermal_relaxation_kraus
from ..conversion import UnknownGates, native_name, resolve_op
from ..layout import normalize_layout
from ..profile import Profile
from ..report import Report

CirqLayout = Mapping[Any, int] | Sequence[int]
_ANGLE_TOL = 1e-9


@dataclass(frozen=True)
class _PowFamily:
    """Canonical names of a Cirq EigenGate class by exponent, compared modulo ``period``.

    Exponents that name no gate map to ``other``; when ``other`` is None the gate is unknown
    and is reported as ``base**exponent``. A named exponent takes its name under
    :func:`~noisevault.conversion.native_name`, with ``other`` as the rotation it equals.
    """

    base: str
    period: float
    names: tuple[tuple[float, str], ...]
    other: str | None

    def name_for(self, exponent: Any, defined: Container[str]) -> str:
        if isinstance(exponent, numbers.Real):
            for value, name in self.names:
                gap = (float(exponent) - value) % self.period
                if min(gap, self.period - gap) < _ANGLE_TOL:
                    return native_name(name, defined, self.other)
        if self.other is not None:
            return self.other
        shown = f"{exponent:.6g}" if isinstance(exponent, numbers.Real) else str(exponent)
        return f"{self.base}**{shown}"


# Up to a global phase these gates repeat with period 2 in the exponent (iSWAP with 4), so
# inverses such as X**-1, S**-1 or CZ**-1 from cirq.inverse map to their gates. MS(pi, 0) is
# XX**-0.5 and MS(pi/2, +-pi/2) is YY**+-0.5, so both signs are the native MS; MSGate is an
# XXPowGate and is named the same way.
_XX = _PowFamily("rxx", 2, ((0.5, "ms"), (-0.5, "ms")), "rxx")
_POW: dict[type, _PowFamily] = {
    cirq.XPowGate: _PowFamily("x", 2, ((1, "x"), (0.5, "sx"), (-0.5, "sxdg")), "rx"),
    cirq.YPowGate: _PowFamily("y", 2, ((1, "y"),), "ry"),
    cirq.ZPowGate: _PowFamily(
        "z", 2, ((1, "z"), (0.5, "s"), (-0.5, "sdg"), (0.25, "t"), (-0.25, "tdg")), "rz"
    ),
    cirq.HPowGate: _PowFamily("h", 2, ((1, "h"),), None),
    cirq.CXPowGate: _PowFamily("cx", 2, ((1, "cx"),), None),
    cirq.CZPowGate: _PowFamily("cz", 2, ((1, "cz"),), None),
    cirq.SwapPowGate: _PowFamily("swap", 2, ((1, "swap"),), None),
    # ISWAP**-0.5 is the same coupler pulse with the opposite sign; Google calibrates the pair
    # as one gate and cirq_google charges it the sqrt_iswap error.
    cirq.ISwapPowGate: _PowFamily(
        "iswap", 4, ((1, "iswap"), (0.5, "sqrt_iswap"), (-0.5, "sqrt_iswap")), None
    ),
    cirq.ZZPowGate: _PowFamily("rzz", 2, ((0.5, "zz"),), "rzz"),
    cirq.XXPowGate: _XX,
    cirq.MSGate: _XX,
    cirq.YYPowGate: _PowFamily("ryy", 2, ((0.5, "ms"), (-0.5, "ms")), "ryy"),
    cirq.CCXPowGate: _PowFamily("ccx", 2, ((1, "ccx"),), None),
}


@cirq.value_equality
class ECRGate(cirq.Gate):
    """IBM's echoed cross-resonance gate ``ecr``, with the gate registry's unitary.

    Cirq has no ECR of its own; the noise model gives this gate a profile's ``ecr`` noise.
    """

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return gates.GATES["ecr"].unitary()

    def _value_equality_values_(self) -> tuple:
        return ()

    def _circuit_diagram_info_(self, args: Any) -> tuple[str, str]:
        return ("ECR", "ECR")

    def __repr__(self) -> str:
        return "noisevault.frameworks.cirq.ECRGate()"


def _registry_by_class() -> dict[str, str]:
    """Registry gates whose Cirq class alone names them (e.g. Rx -> rx)."""
    names: dict[str, list[str]] = defaultdict(list)
    for info in gates.GATES.values():
        if info.cirq and info.unitary is not None:
            names[info.cirq].append(info.name)
    return {cls: found[0] for cls, found in names.items() if len(found) == 1}


_BY_CLASS = _registry_by_class()
_OWN_GATES: dict[type, str] = {ECRGate: "ecr"}
_RESOLVE_FIRST = (
    "resolve its parameters before adding noise: cirq.resolve_parameters(circuit,"
    " params).with_noise(model), or give the parameterized circuit and its sweep to a simulator"
    " built with noise=model, which resolves them first"
)


def gate_name(gate: cirq.Gate, defined: Container[str] = ()) -> str:
    """The canonical NoiseVault name of a Cirq unitary gate.

    The most specific class decides: this module's own gates (:class:`ECRGate`), an exponent
    table for the power gates, else the gate registry's Cirq column. A power gate at a fixed
    angle (``ZZ**0.5`` is ``zz``, ``X`` is ``x``, ``XX**0.5`` and ``YY**0.5`` are ``ms``) takes
    its rotation's name instead (``rzz``, ``rx``, ``rxx``, ``ryy``) when ``defined``, a
    profile's gate names, has the rotation but not the fixed gate (see
    :func:`~noisevault.conversion.native_name`). Any other gate is named after its class in
    snake case without the ``Gate`` suffix (``FSimGate`` -> ``fsim``, ``MatrixGate`` ->
    ``matrix``), so a profile can calibrate it under that name; otherwise it gets the
    typical-noise rule. A power gate whose name depends on an unresolved exponent
    (``X**t`` is ``x`` at t=1) raises ValueError.
    """
    for cls in type(gate).__mro__:
        if cls in _OWN_GATES:
            return _OWN_GATES[cls]
        family = _POW.get(cls)
        if family is not None:
            if family.names and cirq.is_parameterized(gate):
                raise ValueError(f"the gate of {gate!r} depends on its exponent; {_RESOLVE_FIRST}")
            return family.name_for(gate.exponent, defined)  # type: ignore[attr-defined]
        if cls.__name__ in _BY_CLASS:
            return _BY_CLASS[cls.__name__]
    stem = re.sub(r"(Pow)?Gate$", "", type(gate).__name__) or type(gate).__name__
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stem).lower()
    return type(gate).__name__ if name in gates.GATES else name


class _QubitMap:
    """Cirq qubits to device qubits: the layout, else LineQubit index or GridQubit coords."""

    def __init__(self, profile: Profile, layout: CirqLayout | None) -> None:
        self._profile = profile
        self._explicit = layout is not None
        self._coords = {tuple(r.coords): r.index for r in profile.qubits if r.coords is not None}
        self._known: dict[cirq.Qid, int] = {}
        if layout is not None:
            keyed = _keyed_layout(layout)
            self._known = dict(normalize_layout(list(keyed), keyed, profile))  # type: ignore[arg-type]

    def physical(self, qids: Sequence[cirq.Qid]) -> tuple[int, ...]:
        """Device qubits of ``qids``, which must map to distinct ones."""
        indices = tuple(self._one(qid) for qid in qids)
        owner: dict[int, cirq.Qid] = {}
        for qid, index in zip(qids, indices, strict=True):
            first = owner.setdefault(index, qid)
            if first != qid:
                raise LayoutError(
                    f"{first!r} and {qid!r} both map to device qubit {index}; pass layout= to"
                    " place them explicitly"
                )
        return indices

    def _one(self, qid: cirq.Qid) -> int:
        if qid in self._known:
            return self._known[qid]
        if self._explicit:
            raise LayoutError(
                f"layout has no device qubit for {qid!r}; map every circuit qubit in layout="
            )
        if qid.dimension != 2:
            raise LayoutError(f"{qid!r} has dimension {qid.dimension}; profiles describe qubits")
        (index,) = normalize_layout([qid], {qid: self._default_index(qid)}, self._profile).values()
        self._known[qid] = index
        return index

    def _default_index(self, qid: cirq.Qid) -> int:
        if isinstance(qid, cirq.LineQubit):
            return qid.x
        fix = f"pass layout={{{qid!r}: <device qubit>, ...}} covering every circuit qubit"
        if isinstance(qid, cirq.GridQubit) and self._coords:
            index = self._coords.get((qid.row, qid.col))
            if index is None:
                some = ", ".join(f"GridQubit{c}" for c in list(self._coords)[:3])
                raise LayoutError(
                    f"{self._profile.id} has no qubit at coords ({qid.row}, {qid.col}); use the"
                    f" device's coords (e.g. {some}) or {fix}"
                )
            return index
        why = "records no qubit coords" if isinstance(qid, cirq.GridQubit) else "cannot place it"
        raise LayoutError(
            f"{qid!r} needs a layout: {self._profile.id} {why} and only LineQubit(i) maps to"
            f" device qubit i by default; {fix}"
        )


def _keyed_layout(layout: CirqLayout) -> dict[cirq.Qid, int]:
    """A layout keyed by Cirq qubits; a sequence or integer keys mean LineQubit(i)."""
    if not isinstance(layout, Mapping):
        return {cirq.LineQubit(i): p for i, p in enumerate(layout)}
    keyed: dict[cirq.Qid, int] = {}
    for key, index in layout.items():
        if isinstance(key, int) and not isinstance(key, bool):
            key = cirq.LineQubit(key)
        if not isinstance(key, cirq.Qid):
            raise LayoutError(f"layout key {key!r} is not a Cirq qubit or an integer")
        keyed[key] = index
    return keyed


class NoiseVaultNoiseModel(cirq.NoiseModel):
    """A profile's noise as a Cirq noise model; ``.report`` says what was reproduced.

    Use it with ``cirq.DensityMatrixSimulator(noise=model)`` (exact) or
    ``cirq.Simulator(noise=model)`` (sampled trajectories). The report keeps collecting events,
    such as gates that got typical noise, as circuits are simulated.

    Readout error is classical: a mid-circuit measurement reports a flipped bit and leaves the
    qubit in its true state (a ``confusion_map``). A terminal measurement instead gets the same
    assignment probabilities as a channel just before it, because Cirq samples terminal
    measurements from the original circuit's gates and would drop a confusion map. Sampled
    results are exact either way, but the state ``simulate()`` returns after a terminal
    measurement includes those flips; inspect states with ``readout=False`` or without the
    final measurements.

    Report events count conversions, not shots or runs: Cirq converts a circuit once per run or
    per part of a run, and a repeat of the circuit converted last reuses that conversion.
    """

    def __init__(
        self,
        profile: Profile,
        report: Report,
        qubits: _QubitMap,
        *,
        unknown_gates: UnknownGates,
        readout: bool,
    ) -> None:
        self.profile = profile
        self.report = report
        self._table = profile.table
        self._qubits = qubits
        self._unknown_gates = unknown_gates
        self._readout = readout
        self._last: tuple[tuple[cirq.Moment, ...], list[cirq.OP_TREE]] | None = None

    def __repr__(self) -> str:
        return f"NoiseVaultNoiseModel({self.profile.id}, nv:{self.profile.fingerprint[:12]})"

    def noisy_moments(
        self, moments: Iterable[cirq.Moment], system_qubits: Sequence[cirq.Qid]
    ) -> Sequence[cirq.OP_TREE]:
        moments = tuple(moments)
        # Cirq's per-shot paths (trajectories, mid-circuit measurements) ask for the same
        # moments on every repetition; reusing the last conversion keeps them fast and keeps
        # report events from counting shots.
        if self._last is not None and self._last[0] == moments:
            return self._last[1]
        self._qubits.physical(system_qubits)
        later: set[cirq.Qid] = set()
        terminal: set[tuple[int, cirq.Operation]] = set()
        for i in reversed(range(len(moments))):
            for op in moments[i]:
                if cirq.is_measurement(op) and later.isdisjoint(op.qubits):
                    terminal.add((i, op))
            later.update(moments[i].qubits)
        noisy: list[cirq.OP_TREE] = [
            [self._noisy(op, terminal=(i, op) in terminal) for op in moment]
            for i, moment in enumerate(moments)
        ]
        self._last = (moments, noisy)
        return noisy

    def noisy_operation(self, operation: cirq.Operation) -> cirq.OP_TREE:
        """The operation followed by its noise; a measurement gets a ``confusion_map``."""
        return self._noisy(operation, terminal=False)

    def _noisy(self, operation: cirq.Operation, *, terminal: bool) -> cirq.OP_TREE:
        gate = operation.gate
        if not operation.qubits:
            return operation
        if isinstance(operation, cirq.ClassicallyControlledOperation):
            raise ValueError(
                f"{operation!r} is classically controlled, which the Cirq export does not"
                " support yet; replace the feed-forward with a quantum-controlled gate and"
                " measure at the end, or simulate the parts before and after it separately"
            )
        if gate is None:
            raise ValueError(
                f"{operation!r} is not a plain gate operation; flatten it first, e.g."
                " cirq.Circuit(cirq.decompose(circuit, keep=lambda op: op.gate is not None))"
            )
        physical = self._qubits.physical(operation.qubits)
        if isinstance(gate, cirq.MeasurementGate):
            return self._measure(operation, gate, physical, terminal=terminal)
        if cirq.is_measurement(gate):
            return self._other_measurement(operation)
        if isinstance(gate, cirq.ResetChannel):
            return self._reset(operation, physical)
        if isinstance(gate, cirq.WaitGate):
            return self._wait(operation, gate, physical)
        if isinstance(gate, cirq.IdentityGate):
            pairs = zip(operation.qubits, physical, strict=True)
            return [operation, *(op for q, p in pairs for op in self._noise("id", [q], [p]))]
        if isinstance(gate, cirq.PhasedXZGate) and "phased_xz" not in self.profile.gates:
            return self._phased_xz(gate, operation.qubits, physical)
        if not (cirq.has_unitary(gate) or cirq.is_parameterized(gate)):
            self.report.count("circuit_channel_kept", type(gate).__name__)
            return operation
        name = gate_name(gate, self.profile.gates)
        return [operation, *self._noise(name, operation.qubits, physical)]

    def _noise(
        self, name: str, qids: Sequence[cirq.Qid], physical: Sequence[int]
    ) -> list[cirq.Operation]:
        built = resolve_op(
            self._table, name, physical, unknown_gates=self._unknown_gates, report=self.report
        )
        qid_of = dict(zip(physical, qids, strict=True))
        return [
            cirq.KrausChannel(list(channel.kraus)).on(*(qid_of[w] for w in channel.wires))
            for channel in built.channels
        ]

    def _phased_xz(
        self, gate: cirq.PhasedXZGate, qids: Sequence[cirq.Qid], physical: tuple[int, ...]
    ) -> list[cirq.Operation]:
        """PhasedXZ is exactly the r gate (PhasedXPow) then a Z rotation, each with its noise.

        Google calibrates PhasedXZ as r with a virtual Z, so it needs no typical noise there.
        Only a profile without its own ``phased_xz`` gets here.
        """
        x_part = cirq.PhasedXPowGate(
            phase_exponent=gate.axis_phase_exponent, exponent=gate.x_exponent
        )
        out = [x_part.on(*qids), *self._noise("r", qids, physical)]
        if cirq.is_parameterized(gate.z_exponent) or gate.z_exponent != 0:
            out += [(cirq.Z**gate.z_exponent).on(*qids), *self._noise("rz", qids, physical)]
        return out

    def _other_measurement(self, operation: cirq.Operation) -> cirq.Operation:
        if self._readout:
            raise ValueError(
                f"{operation!r} is not a cirq.measure, so NoiseVault cannot add its readout"
                " error; rotate into the Z basis and use cirq.measure, or pass"
                " readout=False to keep it noiseless"
            )
        return operation

    def _measure(
        self,
        operation: cirq.Operation,
        gate: cirq.MeasurementGate,
        physical: tuple[int, ...],
        *,
        terminal: bool,
    ) -> cirq.OP_TREE:
        if not self._readout:
            return operation
        if gate.confusion_map:
            raise ValueError(
                f"{operation!r} already has a confusion_map; remove it, or pass readout=False to"
                " keep your own readout model"
            )
        matrices = {}
        for position, index in enumerate(physical):
            matrix = readout_matrix(self._table.qubit(index))
            if matrix is None:
                self.report.mark_unknown(f"readout of qubit {index}")
            else:
                matrices[position] = matrix
        if not matrices:
            return operation
        if terminal:
            self.report.approximate(
                "state after a terminal measurement",
                "includes the readout flips",
                "sampled results are exact; inspect states with readout=False",
            )
            flips = [
                cirq.KrausChannel(_assignment_kraus(m)).on(operation.qubits[position])
                for position, m in matrices.items()
            ]
            return [*flips, operation]
        noisy = cirq.MeasurementGate(
            gate.num_qubits(),
            key=gate.mkey,
            invert_mask=gate.invert_mask,
            qid_shape=cirq.qid_shape(gate),
            confusion_map={(position,): m.T for position, m in matrices.items()},  # rows: truth
        )
        return noisy.on(*operation.qubits).with_tags(*operation.tags)

    def _reset(self, operation: cirq.Operation, physical: tuple[int, ...]) -> cirq.OP_TREE:
        error = self._table.qubit(physical[0]).prep_error
        if error is None:
            self.report.mark_unknown(f"preparation error of qubit {physical[0]}")
        if not error:
            return operation
        return [operation, cirq.bit_flip(error).on(*operation.qubits)]

    def _wait(
        self, operation: cirq.Operation, gate: cirq.WaitGate, physical: tuple[int, ...]
    ) -> list[cirq.Operation]:
        if cirq.is_parameterized(gate):
            raise ValueError(f"{operation!r} has an unresolved duration; {_RESOLVE_FIRST}")
        duration = float(gate.duration.total_nanos())
        out = [operation]
        for qid, index in zip(operation.qubits, physical, strict=True):
            q = self._table.qubit(index)
            if q.t1_ns is None and q.t2_ns is None and not q.dephasing_rate_per_s:
                self.report.mark_unknown(f"T1 and T2 of qubit {index} (no WaitGate relaxation)")
                continue
            if q.t2_clamped:
                self.report.record_t2_clamp(index)
            kraus = thermal_relaxation_kraus(q.t1_ns, q.t2_ns, duration, q.dephasing_rate_per_s)
            if len(kraus) > 1:
                out.append(cirq.KrausChannel(kraus).on(qid))
        return out


def _assignment_kraus(matrix: np.ndarray) -> list[np.ndarray]:
    """Kraus operators sqrt(M[m, p]) |m><p|: a Z measurement after them reports m with M[m, p]."""
    ops = []
    for measured, prepared in product(range(2), repeat=2):
        op = np.zeros((2, 2), dtype=complex)
        op[measured, prepared] = np.sqrt(matrix[measured, prepared])
        ops.append(op)
    return ops


def to_cirq(
    profile: Profile,
    *,
    layout: CirqLayout | None = None,
    unknown_gates: UnknownGates = "typical",
    readout: bool = True,
) -> NoiseVaultNoiseModel:
    """A Cirq noise model of ``profile``, carrying ``.report`` and ``.profile``.

    ``layout`` maps circuit qubits (Cirq qubits, or integers meaning ``LineQubit(i)``, or a
    sequence indexed by LineQubit) to device qubits. Without it ``LineQubit(i)`` is qubit ``i``
    and ``GridQubit(r, c)`` is the qubit at coords ``(r, c)`` if the profile records coords.
    ``unknown_gates="typical"`` gives gates the profile does not calibrate the noise of its
    typical native gate (reported and warned about once per gate); ``"error"`` raises
    instead. ``readout=False`` leaves measurements noiseless.
    """
    if unknown_gates not in get_args(UnknownGates):
        raise ValueError(f"unknown_gates={unknown_gates!r}: choose 'typical' or 'error'")
    if not isinstance(readout, bool):
        raise TypeError(
            f"readout={readout!r}: pass True to add readout error or False to leave"
            " measurements noiseless"
        )
    report = Report.start(
        profile,
        "cirq",
        cirq.__version__,
        layout=layout,
        unknown_gates=unknown_gates,
        readout=readout,
    )
    report.record_effects(profile.effects)
    qubits = _QubitMap(profile, layout)
    report.mark_exact("gate noise as Kraus channels after each gate")
    if readout:
        report.mark_exact(
            "readout assignment error (confusion_map mid-circuit, an equivalent channel before"
            " terminal measurements)"
        )
    else:
        report.omit("readout error (readout=False)")
    report.mark_exact("preparation error after each reset")
    report.mark_exact("thermal relaxation during WaitGate")
    report.omit("preparation error of the initial state (only resets get it)")
    report.omit("idle noise outside WaitGate (unscheduled idle time)")
    return NoiseVaultNoiseModel(
        profile, report, qubits, unknown_gates=unknown_gates, readout=readout
    )
