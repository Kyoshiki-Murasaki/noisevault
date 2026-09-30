"""The resolved layer: the noise one qubit or one gate application gets under the format rules.

Built lazily from a Profile and cached per lookup. All-to-all connectivity is never expanded
into pairs unless :meth:`NoiseTable.edges` is called.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Literal

from . import gates, metrics
from .profile import Connectivity, GateSpec, GateState, Idle, merge_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .profile import Profile

Origin = Literal["record", "reversed_record", "default"]
# Why a lookup found no noise: the gate is not in the profile, the pair is not connected, the
# targets do not fit the gate, a target qubit is disabled, or no native gate fits (typical).
UnavailableKind = Literal["undefined", "not_connected", "bad_target", "qubit_disabled", "no_native"]


@dataclass(frozen=True)
class QubitNoise:
    index: int
    t1_ns: float | None
    t2_ns: float | None
    dephasing_rate_per_s: float | None  # extra Z error with probability rate * time
    readout: tuple[float, float] | None  # (P(1|0), P(0|1)); None means unknown
    prep_error: float | None  # None means unknown
    disabled: bool


@dataclass(frozen=True)
class GateNoise:
    gate: str
    qubits: tuple[int, ...]
    state: GateState
    avg_infidelity: float | None  # None unless state is "calibrated"
    pauli: tuple[float, ...] | None  # only for a pauli spec; labels relative to ``qubits``
    duration_ns: float | None
    origin: Origin
    spec: GateSpec  # merged spec with its qualifiers


@dataclass(frozen=True)
class Unavailable:
    gate: str
    qubits: tuple[int, ...]
    kind: UnavailableKind
    reason: str


class NoiseTable:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.num_qubits = profile.device.num_qubits
        self.technology = profile.device.technology
        self.all_to_all = not isinstance(profile.connectivity, Connectivity)
        conn = profile.connectivity
        self._directed = isinstance(conn, Connectivity) and conn.directed
        self._edges = set() if not isinstance(conn, Connectivity) else set(conn.edges)
        self._records = {(r.gate, r.qubits): r for r in profile.calibrations}
        self._record_counts = Counter(r.gate for r in profile.calibrations)
        self._with_metric = {name for name, spec in profile.gates.items() if spec.metric}
        self._with_metric |= {r.gate for r in profile.calibrations if r.metric}
        self._qubit_records = {q.index: q for q in profile.qubits}
        self._qubits: dict[int, QubitNoise] = {}
        self._gates: dict[tuple[str, tuple[int, ...]], GateNoise | Unavailable] = {}

    # gate definitions ----------------------------------------------------------------------

    def arity(self, name: str) -> int | None:
        spec, info = self.profile.gates.get(name), gates.lookup(name)
        if spec is not None and spec.qubits is not None:
            return spec.qubits
        return info.arity if info else None

    def symmetric(self, name: str) -> bool:
        spec = self.profile.gates.get(name)
        if spec is not None and spec.symmetric is not None:
            return spec.symmetric
        return gates.is_symmetric(name)

    def natives(self, arity: int) -> tuple[str, ...]:
        """Defined, non-virtual unitary gates of ``arity``, in definition order."""
        out = []
        for name, spec in self.profile.gates.items():
            info = gates.lookup(name)
            if spec.virtual or self.arity(name) != arity or (info and info.unitary is None):
                continue
            out.append(name)
        return tuple(out)

    # qubits --------------------------------------------------------------------------------

    def qubit(self, index: int) -> QubitNoise:
        if not 0 <= index < self.num_qubits:
            raise IndexError(f"qubit {index} is outside 0..{self.num_qubits - 1}")
        if index not in self._qubits:
            self._qubits[index] = self._resolve_qubit(index)
        return self._qubits[index]

    def _resolve_qubit(self, index: int) -> QubitNoise:
        idle = self.profile.idle or Idle()
        record = self._qubit_records.get(index)
        t1 = record.t1_us if record and record.t1_us is not None else idle.t1_us
        t2 = record.t2_us if record and record.t2_us is not None else idle.t2_us
        dephasing = idle.dephasing_rate_per_s
        if record and record.dephasing_rate_per_s is not None:
            dephasing = record.dephasing_rate_per_s
        readout = record.readout if record and record.readout is not None else self.profile.readout
        prep = record.prep if record and record.prep is not None else self.profile.prep
        return QubitNoise(
            index=index,
            t1_ns=None if t1 is None else t1 * 1000,
            t2_ns=None if t2 is None else t2 * 1000,
            dephasing_rate_per_s=dephasing,
            readout=readout.pair if readout else None,
            prep_error=prep.error if prep else None,
            disabled=bool(record and record.disabled),
        )

    # gates ---------------------------------------------------------------------------------

    def gate(self, name: str, qubits: Sequence[int]) -> GateNoise | Unavailable:
        key = (name, tuple(qubits))
        if key not in self._gates:
            self._gates[key] = self._resolve_gate(*key)
        return self._gates[key]

    def allowed(self, name: str, qubits: Sequence[int]) -> bool:
        found = self.gate(name, qubits)
        return isinstance(found, GateNoise) and found.state != "disabled"

    def typical(self, arity: int, qubits: Sequence[int]) -> GateNoise | Unavailable:
        """The calibrated native of ``arity`` with the most records that is usable on ``qubits``.

        Ties go to the alphabetically first name, so the choice depends only on fingerprinted
        content. A directed 2-qubit gate may use its record on the reversed pair
        (origin ``reversed_record``).
        """
        qubits = tuple(qubits)
        candidates = [name for name in self.natives(arity) if name in self._with_metric]
        candidates.sort(key=lambda name: (-self._record_counts[name], name))
        for name in candidates:
            found = self.gate(name, qubits)
            if isinstance(found, GateNoise) and found.state in ("calibrated", "disabled"):
                if found.state == "calibrated":
                    return found
                continue  # disabled here: the next candidate, never its reversed record
            if arity == 2 and not self.symmetric(name):
                flipped = self._reversed_record(name, qubits)
                if flipped is not None and flipped.state == "calibrated":
                    return flipped
        return Unavailable(
            "typical",
            qubits,
            "no_native",
            f"no calibrated {arity}-qubit native gate is usable on {qubits}",
        )

    def edges(self) -> list[tuple[int, int]]:
        """Connectivity pairs; for all-to-all every pair (a < b), built only on request."""
        if self.all_to_all:
            return list(combinations(range(self.num_qubits), 2))
        return sorted(self._edges)

    def _resolve_gate(self, name: str, qubits: tuple[int, ...]) -> GateNoise | Unavailable:
        spec = self.profile.gates.get(name)
        if spec is None:
            info = gates.lookup(name)
            if info and info.family == "z" and len(qubits) == 1:
                rz = self.gate("rz", qubits)
                if isinstance(rz, GateNoise) and rz.state == "ideal":
                    return GateNoise(name, qubits, "ideal", None, None, None, "default", rz.spec)
            return Unavailable(name, qubits, "undefined", f"{name} is not defined in this profile")
        problem = self._target_problem(name, qubits)
        if problem:
            return Unavailable(name, qubits, *problem)
        record = self._records.get((name, qubits))
        if record is not None:
            return self._noise(name, qubits, merge_spec(spec, record), "record")
        if len(qubits) == 2 and self.symmetric(name):
            flipped = self._reversed_record(name, qubits)
            if flipped is not None:
                return flipped
        if self._connected(qubits):
            return self._noise(name, qubits, spec, "default")
        reason = f"{name} has no calibration on {qubits} and connectivity does not allow it"
        return Unavailable(name, qubits, "not_connected", reason)

    def _target_problem(
        self, name: str, qubits: tuple[int, ...]
    ) -> tuple[UnavailableKind, str] | None:
        arity = self.arity(name)
        if len(qubits) != arity:
            return "bad_target", f"{name} acts on {arity} qubits, got {len(qubits)}"
        if len(set(qubits)) != len(qubits):
            return "bad_target", f"targets of {name} must be distinct, got {qubits}"
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                return "bad_target", f"qubit {q} is outside 0..{self.num_qubits - 1}"
            if self.qubit(q).disabled:
                return "qubit_disabled", f"qubit {q} is disabled"
        return None

    def _reversed_record(self, name: str, qubits: tuple[int, ...]) -> GateNoise | None:
        record = self._records.get((name, qubits[::-1]))
        if record is None or self._target_problem(name, qubits):
            return None
        spec = merge_spec(self.profile.gates[name], record)
        return self._noise(name, qubits, spec, "reversed_record")

    def _connected(self, qubits: tuple[int, ...]) -> bool:
        if len(qubits) == 1 or self.all_to_all:
            return True
        if len(qubits) > 2:
            return False
        a, b = qubits
        return (a, b) in self._edges or (not self._directed and (b, a) in self._edges)

    def _noise(
        self, name: str, qubits: tuple[int, ...], spec: GateSpec, origin: Origin
    ) -> GateNoise:
        state = _state(spec)
        pauli, r = None, None
        metric = spec.metric
        if metric is not None and state == "calibrated":
            kind, value = metric
            if kind == "pauli":
                pauli = tuple(value)
                if origin == "reversed_record":
                    pauli = metrics.swap_pauli_2q(pauli)
                    spec = spec.model_copy(update={"pauli": pauli})
                r = metrics.avg_from_pauli(pauli)
            else:
                r = metrics.to_avg_infidelity(kind, value, len(qubits))
        return GateNoise(name, qubits, state, r, pauli, spec.duration_ns, origin, spec)


def _state(spec: GateSpec) -> GateState:
    if spec.disabled:
        return "disabled"
    if spec.virtual:
        return "ideal"
    return "calibrated" if spec.metric else "uncalibrated"
