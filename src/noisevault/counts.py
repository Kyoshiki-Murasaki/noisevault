"""The counts file: a run on a device, bound to one profile, its qubits and the exact ops it ran.

:func:`plan` gives the circuits to run, :func:`simulate` draws counts for them from a profile,
and :func:`load_counts` reads a counts file. ``nv compare`` scores a profile on the result. The
models are the file, as :class:`~noisevault.Profile` is: frozen, extra keys refused, canonical
after validation. Counts keys are stored with classical bit 0 on the left and zero entries
dropped, so the same run gives the same bytes and the same ``sha256`` whichever bit order the
file was written in.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    BeforeValidator,
    Field,
    PlainSerializer,
    Strict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from . import __version__, check, gates
from .errors import CountsError, NoiseVaultError, did_you_mean, plural
from .profile import (
    Count,
    CountsSource,
    Fingerprint,
    FrozenDict,
    JsonObject,
    Profile,
    QubitIndex,
    UtcDatetime,
    _Model,
    _readable_json,
    _sha256,
    _string_keys,
    read_json_file,
    write_atomically,
)
from .reference import MAX_QUBITS, Op, charged_as, outcome_bits, probabilities
from .table import GateNoise

COUNTS_FORMAT = "1.0"
_MAX_OUTCOMES = 4096
BitOrder = Literal["clbit0_left", "qiskit"]


_OP_EXAMPLE = '["rz", [0], [1.5708]]'
_KEY_HINT = "write each key in 0 and 1, one bit per circuit qubit"
_DELAY_HINT = 'give the idle time in nanoseconds, 0 or more, such as ["delay", [0], [68.0]]'
_NOT_OPS = {
    "measure": "every circuit qubit is measured once, after the last op",
    "reset": "every shot starts with each qubit in 0",
}
_OP_NAMES = sorted(["delay", *(n for n, g in gates.GATES.items() if g.unitary is not None)])


def _op_from_wire(value: Any) -> Op:
    if isinstance(value, Op):
        value = (value.name, value.qubits, value.params)
    shaped = isinstance(value, list | tuple) and len(value) == 3
    name, qubits, params = value if shaped else (None, None, None)
    if not (
        isinstance(name, str)
        and isinstance(qubits, list | tuple)
        and all(isinstance(q, int) and not isinstance(q, bool) and q >= 0 for q in qubits)
        and isinstance(params, list | tuple)
        and all(isinstance(p, int | float) and not isinstance(p, bool) for p in params)
    ):
        raise CountsError(
            f"{_shown(value)} is not an op",
            hint=f"write an op as [name, [circuit qubits], [parameters]], such as {_OP_EXAMPLE}",
        )
    op = Op(name, tuple(qubits), tuple(_real(p) for p in params))
    if op.name == "delay":
        _check_delay(op)
    else:
        _check_gate(op)
    return op


def _op_to_wire(op: Op) -> list[Any]:
    return [op.name, list(op.qubits), list(op.params)]


def _check_delay(op: Op) -> None:
    if len(op.qubits) != 1:
        raise CountsError(
            f"a delay idles one qubit, not {len(op.qubits)}", hint="write one delay for each qubit"
        )
    if len(op.params) != 1:
        raise CountsError(f"a delay takes one duration, not {len(op.params)}", hint=_DELAY_HINT)
    (duration,) = op.params
    if not math.isfinite(duration):
        raise CountsError(f"delay duration {duration} is not a finite number", hint=_DELAY_HINT)
    if duration < 0:
        raise CountsError(f"delay duration {duration:g} ns is negative", hint=_DELAY_HINT)


def _check_gate(op: Op) -> None:
    if op.name in _NOT_OPS:
        raise CountsError(
            f"ops cannot hold {op.name}, because {_NOT_OPS[op.name]}", hint=f"remove the {op.name}"
        )
    info = gates.lookup(op.name)
    if info is None or info.unitary is None:
        raise CountsError(
            f"{_shown(op.name)} is not a gate NoiseVault can simulate",
            hint=f"{did_you_mean(op.name, _OP_NAMES)}use a gate name from the registry, such as"
            " sx, cz or rz, or delay",
        )
    if len(op.qubits) != info.arity:
        raise CountsError(f"{op.name} acts on {plural(info.arity, 'qubit')}, not {len(op.qubits)}")
    if len(set(op.qubits)) != len(op.qubits):
        raise CountsError(
            f"{op.name} acts on qubits {list(op.qubits)}; its targets must be distinct"
        )
    if len(op.params) != len(info.params):
        names = f" ({', '.join(info.params)})" if info.params else ""
        raise CountsError(
            f"{op.name} takes {plural(len(info.params), 'parameter')}{names}, not {len(op.params)}"
        )
    bad = next((p for p in op.params if not math.isfinite(p)), None)
    if bad is not None:
        raise CountsError(f"{op.name} parameter {bad} is not a finite number")


WireOp = Annotated[Op, BeforeValidator(_op_from_wire), PlainSerializer(_op_to_wire)]


@dataclass(frozen=True)
class _RunRule:
    flag: str | None
    option: str | None
    required: bool | str
    consequence: str
    remedy: str


_RUN_RULES = (
    _RunRule(
        "transpiled",
        None,
        False,
        "the device may have run other ops than the listed ones",
        "as planned, without transpiling them",
    ),
    _RunRule(
        "gate_twirling",
        "twirling.enable_gates",
        False,
        "the device ran random Pauli gates that the ops do not list",
        "without gate twirling",
    ),
    _RunRule(
        "measure_twirling",
        "twirling.enable_measure",
        False,
        "the device flipped qubits at random before measuring them, which changes readout error",
        "without measurement twirling",
    ),
    _RunRule(
        "dynamical_decoupling",
        "dynamical_decoupling.enable",
        False,
        "the device added pulses to idle qubits that the ops do not list",
        "without dynamical decoupling",
    ),
    _RunRule(
        "init_qubits",
        "execution.init_qubits",
        True,
        "a shot may start where the previous one ended, not from 0",
        "with every qubit reset before each shot",
    ),
    _RunRule(
        None,
        "execution.meas_type",
        "classified",
        "the job returned IQ data, not bitstrings",
        "with classified measurements",
    ),
)

SAMPLER_V2_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {rule.option: rule.required for rule in _RUN_RULES if rule.option}
)


class ProfileBinding(_Model):
    """The profile a run was planned from: its id and its fingerprint without unmodeled_error."""

    id: str
    fingerprint: Fingerprint


class Execution(_Model):
    """How the device ran the circuits. Format 1.0 admits only runs of exactly the listed ops,
    each shot from a fresh ground state, with no twirling, decoupling or mitigation.

    ``client`` is free text, such as "qiskit-ibm-runtime 0.49.0 SamplerV2, qiskit 2.5.2", and
    ``options`` holds the options as submitted.
    """

    client: str
    transpiled: Literal[False]
    gate_twirling: Literal[False]
    measure_twirling: Literal[False]
    dynamical_decoupling: Literal[False]
    init_qubits: Literal[True]
    job_ids: tuple[str, ...] = ()
    options: JsonObject = Field(default_factory=FrozenDict)

    @model_validator(mode="before")
    @classmethod
    def _flags(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        for rule in _RUN_RULES:
            if rule.flag is None or rule.flag not in data:
                continue
            value = data[rule.flag]
            if not isinstance(value, bool):
                raise CountsError(f"{rule.flag} is {_shown(value)}; give true or false")
            if value != rule.required:
                raise CountsError(
                    f"{rule.flag} is {_shown(value)}, so {rule.consequence}",
                    hint=f"run the circuits again {rule.remedy}",
                )
        return data

    @model_validator(mode="after")
    def _no_contradiction(self) -> Execution:
        for rule in _RUN_RULES:
            found = _option(self.options, rule.option) if rule.option else None
            if found is not None and found[1] != rule.required:
                path, value = found
                required = (
                    _shown(rule.required) if isinstance(rule.required, str) else rule.required
                )
                raise CountsError(
                    f"options.{path} is {_shown(value)}, so {rule.consequence}",
                    hint=f"run the circuits again with the SamplerV2 option {rule.option}"
                    f" set to {required}",
                )
        return self


def _option(options: Mapping[str, Any], path: str) -> tuple[str, Any] | None:
    parts = path.split(".")
    value: Any = options
    for depth, part in enumerate(parts):
        if not isinstance(value, Mapping):
            stands_for_the_whole_path = ".".join(parts[:depth])
            return stands_for_the_whole_path, value
        if part not in value:
            return None
        value = value[part]
    return path, value


class PlannedCircuit(_Model):
    """A circuit on physical qubits. Circuit qubit i is ``qubits[i]`` and is measured into
    classical bit i, after every op, all qubits together."""

    name: str
    qubits: Annotated[tuple[QubitIndex, ...], Field(min_length=1, max_length=MAX_QUBITS)]
    ops: tuple[WireOp, ...]

    @model_validator(mode="after")
    def _ops_fit(self) -> PlannedCircuit:
        repeated = next((q for q, times in Counter(self.qubits).items() if times > 1), None)
        if repeated is not None:
            raise CountsError(f"qubits lists qubit {repeated} twice")
        n = len(self.qubits)
        for i, op in enumerate(self.ops):
            outside = next((q for q in op.qubits if q >= n), None)
            if outside is not None:
                measured = {1: "circuit qubit 0", 2: "circuit qubits 0 and 1"}.get(
                    n, f"circuit qubits 0 to {n - 1}"
                )
                raise CountsError(
                    f"ops[{i}] acts on circuit qubit {_shown(outside)}, but the circuit measures"
                    f" only {measured}",
                    hint="circuit qubit i is qubits[i], so list the qubit in qubits or fix the op",
                )
        return self


class MeasuredCircuit(PlannedCircuit):
    shots: Count
    counts: Annotated[
        dict[str, Annotated[int, Strict(), Field(ge=0)]], BeforeValidator(_string_keys)
    ]

    @field_validator("counts")
    @classmethod
    def _canonical_counts(cls, counts: dict[str, int], info: ValidationInfo) -> FrozenDict:
        width = len(info.data.get("qubits", ()))
        for key in counts:
            if not set(key) <= {"0", "1"}:
                raise CountsError(
                    f"key {_shown(key)} holds characters other than 0 and 1", hint=_KEY_HINT
                )
            if width and len(key) != width:
                raise CountsError(
                    f"key {_shown(key)} has {plural(len(key), 'bit')}, but the circuit"
                    f" measures {plural(width, 'qubit')}",
                    hint=_KEY_HINT,
                )
        return FrozenDict({key: counts[key] for key in sorted(counts) if counts[key]})

    @model_validator(mode="after")
    def _counts_fit(self) -> MeasuredCircuit:
        total = sum(self.counts.values())
        if total != self.shots:
            raise CountsError(
                f"counts sum to {total}, but shots is {self.shots}",
                hint="give the count of every outcome, so they add up to shots",
            )
        return self

    def vector(self) -> np.ndarray:
        """A fresh int64 array of 2**n counts, big-endian: circuit qubit 0 is the most
        significant bit, the reference's order. Fresh on each call, so the model stays
        immutable and its hash stays true."""
        width = len(self.qubits)
        keys = (outcome_bits(index, width) for index in range(2**width))
        return np.array([self.counts.get(key, 0) for key in keys], dtype=np.int64)


class MeasuredCounts(_Model):
    """A counts file. ``run_at`` is when the device started the first circuit. ``bit_order``
    "qiskit" puts classical bit 0 rightmost in a key; after validation it is "clbit0_left"."""

    nv_counts: Literal["1.0"]
    source: CountsSource
    profile: ProfileBinding
    backend: str
    run_at: UtcDatetime
    bit_order: BitOrder
    execution: Execution
    circuits: Annotated[tuple[MeasuredCircuit, ...], Field(min_length=1)]

    @model_validator(mode="before")
    @classmethod
    def _canonical_bits(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or data.get("bit_order") != "qiskit":
            return data
        circuits = data.get("circuits")
        if not isinstance(circuits, list | tuple):
            return data
        flipped = []
        for circuit in circuits:
            if isinstance(circuit, MeasuredCircuit):
                circuit = circuit.model_dump()
            counts = circuit.get("counts") if isinstance(circuit, Mapping) else None
            if isinstance(counts, Mapping):
                reverse = {(k[::-1] if isinstance(k, str) else k): v for k, v in counts.items()}
                circuit = {**circuit, "counts": reverse}
            flipped.append(circuit)
        return {**data, "bit_order": "clbit0_left", "circuits": flipped}

    @model_validator(mode="after")
    def _distinct(self) -> MeasuredCounts:
        names: set[str] = set()
        for i, circuit in enumerate(self.circuits):
            if circuit.name in names:
                raise CountsError(
                    f"circuits[{i}] repeats the name {_shown(circuit.name)}",
                    hint="give each circuit its own name",
                )
            names.add(circuit.name)
        outcomes = sum(2 ** len(c.qubits) for c in self.circuits)
        if outcomes > _MAX_OUTCOMES:
            raise CountsError(
                f"the circuits have {outcomes} outcomes in all, more than the {_MAX_OUTCOMES}"
                " nv compare holds in memory",
                hint="split the circuits over several counts files",
            )
        return self

    @property
    def sha256(self) -> str:
        """``sha256:`` and the hex SHA-256 of the canonical JSON of :meth:`to_dict`."""
        return "sha256:" + _sha256(self.to_dict())

    @property
    def qubits(self) -> tuple[int, ...]:
        """Every physical qubit measured, in first-seen order."""
        return tuple(dict.fromkeys(q for c in self.circuits for q in c.qubits))

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> MeasuredCounts:
        """A copy; with ``update`` the result is validated and hashed like a new file."""
        if update:
            return type(self).model_validate({**self.to_dict(), **update})
        return super().model_copy(deep=deep)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def save(self, path: str | Path) -> Path:
        """Write readable canonical JSON, replacing ``path``."""
        path = Path(path)
        write_atomically(path, (_readable_json(self.to_dict()) + "\n").encode("utf-8"))
        return path


_PLAIN_ERRORS = {
    "missing": "missing; format 1.0 requires it",
    "extra_forbidden": "not a counts format 1.0 key",
    "string_pattern_mismatch": "not a full fingerprint; give all 64 hex digits",
}


def load_counts(path: str | Path) -> MeasuredCounts:
    """Read a counts file (``.json`` or gzip-compressed JSON).

    Every refusal is a :class:`~noisevault.CountsError` with a one-line message that starts with
    the path and a ``hint`` that says what to do. A file that cannot be read raises OSError.
    """
    path = Path(path)
    data = read_json_file(path, "counts", CountsError)
    if not isinstance(data, dict) or "nv_counts" not in data:
        if isinstance(data, dict) and "noisevault" in data:
            raise CountsError(
                f"{path} is a profile, not a counts file",
                hint="nv compare takes the profile first and the counts file second",
            )
        raise CountsError(
            f"{path} is not a counts file (it has no nv_counts key)",
            hint=f'give a counts file, which holds "nv_counts": "{COUNTS_FORMAT}"',
        )
    if data["nv_counts"] != COUNTS_FORMAT:
        raise CountsError(
            f"{path} is counts format {_shown(data['nv_counts'])}, and this NoiseVault reads"
            f' only "{COUNTS_FORMAT}"',
            hint="upgrade NoiseVault to read a newer format",
        )
    try:
        return MeasuredCounts.model_validate(data)
    except ValidationError as exc:
        message, hint = _first_issue(exc)
        raise CountsError(f"{path}: {message}", hint=hint) from None


def _first_issue(exc: ValidationError) -> tuple[str, str | None]:
    error = exc.errors()[0]
    where = ""
    for part in error["loc"]:
        if isinstance(part, int):
            where += f"[{part}]"
        else:
            where += f".{part}" if where else str(part)
    cause = error.get("ctx", {}).get("error")
    if isinstance(cause, NoiseVaultError):
        text, hint = cause.message, cause.hint
    else:
        text = _PLAIN_ERRORS.get(error["type"]) or error["msg"].removeprefix("Value error, ")
        hint = None
    text = f"{where}: {text}" if where else text
    total = exc.error_count()
    return (f"{text} (1 of {total} problems)" if total > 1 else text), hint


def plan(
    profile: Profile, layout: Mapping[Hashable, int] | Sequence[int] | None = None
) -> tuple[PlannedCircuit, ...]:
    """The circuits a hardware run executes: the ``nv check`` circuits on the ``nv check``
    chain, scheduled as soon as possible, with every wait written as an explicit delay.

    Each qubit is padded with a delay to the circuit's end, so every gap is filled and any
    backend scheduling policy gives the same timeline. Durations come from the calibration as
    stated; a virtual gate or a missing duration takes 0 ns. A profile with unmodeled-error
    factors plans the run of its calibration.
    """
    base = profile.uncorrected()
    chain, circuits = check.plan_circuits(base, layout, purpose="run")
    return tuple(_scheduled(base, c.name, tuple(chain[: c.num_qubits]), c.ops) for c in circuits)


def _scheduled(
    profile: Profile, name: str, qubits: tuple[int, ...], ops: Sequence[Op]
) -> PlannedCircuit:
    free = [Fraction(0)] * len(qubits)
    timed = []
    for op in ops:
        start = max(free[q] for q in op.qubits)
        timed += [
            Op("delay", (q,), (float(start - free[q]),)) for q in op.qubits if free[q] < start
        ]
        timed.append(op)
        end = start + _duration(profile, op, qubits)
        for q in op.qubits:
            free[q] = end
    finish = max(free)
    timed += [Op("delay", (q,), (float(finish - t),)) for q, t in enumerate(free) if t < finish]
    return PlannedCircuit(name=name, qubits=qubits, ops=tuple(timed))


def _duration(profile: Profile, op: Op, qubits: tuple[int, ...]) -> Fraction:
    unitary = gates.unitary(op.name, op.params)
    found = profile.table.gate(
        charged_as(profile, op.name, unitary), [qubits[q] for q in op.qubits]
    )
    if not isinstance(found, GateNoise) or found.state == "ideal" or found.duration_ns is None:
        return Fraction(0)
    return Fraction(found.duration_ns)


def simulate(
    truth: Profile,
    circuits: Sequence[PlannedCircuit],
    *,
    shots: int,
    seed: int,
    run_at: datetime | None = None,
) -> MeasuredCounts:
    """Counts sampled from ``truth``, its unmodeled-error factors included, and bound to its
    calibration, as a run of ``circuits`` would record them.

    ``run_at`` defaults to the current UTC time. The same ``seed`` and ``run_at`` give the same
    counts and the same ``sha256``.
    """
    check.validate_shots(shots)
    rng = np.random.default_rng(seed)
    measured = []
    for circuit in circuits:
        n = len(circuit.qubits)
        probs = probabilities(
            truth, circuit.ops, n, layout=circuit.qubits, readout=True, unknown_gates="error"
        )
        draws = rng.multinomial(shots, probs)
        counts = {outcome_bits(i, n): int(k) for i, k in enumerate(draws)}
        measured.append(
            {
                "name": circuit.name,
                "qubits": circuit.qubits,
                "ops": circuit.ops,
                "shots": shots,
                "counts": counts,
            }
        )
    return MeasuredCounts.model_validate(
        {
            "nv_counts": COUNTS_FORMAT,
            "source": "simulated",
            "profile": ProfileBinding(id=truth.id, fingerprint=truth.calibration_fingerprint),
            "backend": truth.device.name,
            "run_at": datetime.now(UTC) if run_at is None else run_at,
            "bit_order": "clbit0_left",
            "execution": Execution(
                client=f"noisevault {__version__} simulate",
                transpiled=False,
                gate_twirling=False,
                measure_twirling=False,
                dynamical_decoupling=False,
                init_qubits=True,
                options={"seed": seed, "unmodeled_error": truth.to_dict().get("unmodeled_error")},
            ),
            "circuits": measured,
        }
    )


def _real(value: int | float) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _shown(value: Any, limit: int = 40) -> str:
    text = json.dumps(value, ensure_ascii=False, default=repr)
    return text if len(text) <= limit else text[: limit - 3] + "..."
