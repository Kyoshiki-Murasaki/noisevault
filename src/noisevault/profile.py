"""The NoiseVault profile format 1.0: models, validation, hashing, ids and refs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
import warnings
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cached_property
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    Strict,
    ValidationInfo,
    field_validator,
    model_validator,
)

from . import compat, gates, metrics
from .errors import MigrationWarning
from .units import DURATION, T1, T2, normalize_times

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    from .check import CheckResult
    from .diff import ProfileDiff
    from .table import GateNoise, NoiseTable

FORMAT_VERSION = "1.0"


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


UtcDatetime = Annotated[
    AwareDatetime, AfterValidator(_to_utc), PlainSerializer(_iso_z, when_used="json")
]
# Strict scalars: a hand-written true, "1" or 3.0 is a typo to report, not a value to coerce.
Real = Annotated[float, Strict()]  # still accepts an int
Flag = Annotated[bool, Strict()]
Count = Annotated[int, Strict(), Field(ge=1)]
QubitIndex = Annotated[int, Strict(), Field(ge=0)]
Probability = Annotated[float, Strict(), Field(ge=0, le=1)]
NonNegative = Annotated[float, Strict(), Field(ge=0)]
Positive = Annotated[float, Strict(), Field(gt=0)]
Technology = Literal["superconducting", "trapped_ion", "neutral_atom", "spin", "photonic", "other"]
GateState = Literal["ideal", "calibrated", "uncalibrated", "disabled"]
EffectType = Literal[
    "leakage",
    "atom_loss",
    "erasure",
    "crosstalk_measurement",
    "crosstalk_zz",
    "coherent_overrotation",
]
_FORBIDDEN_NAME = re.compile(r"[@/\\]")


class _Model(BaseModel):
    # model_copy(update=...) skips validation, so a nested instance must be checked again
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )


class FrozenDict(dict):
    """A dict that refuses changes, so a validated profile cannot drift from its fingerprint."""

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("a Profile is immutable; use profile.model_copy(update=...) to change it")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self) -> tuple[type, tuple[dict]]:
        return FrozenDict, (dict(self),)


def _freeze(value: Any, where: str = "") -> Any:
    """Read-only copy of JSON data: mappings become FrozenDict, lists and tuples become tuples.

    Anything that would not survive a save unchanged is refused: a non-string key (``0`` and
    ``"0"`` would collide), a set or other object, and a nonfinite number, which JSON would
    write as null (``allow_inf_nan=False`` does not reach values typed ``Any``).
    """
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: the key {key!r} is not a string")
            frozen[key] = _freeze(item, f"{where}.{key}" if where else key)
        return FrozenDict(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze(item, f"{where}[{i}]") for i, item in enumerate(value))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: {value} is not a finite number")
    if value is not None and not isinstance(value, str | int | float):
        raise ValueError(f"{where}: a {type(value).__name__} is not JSON data")
    return value


def _string_keys(value: Any) -> Any:
    """Pydantic decodes bytes keys to str, so ``{b"x": 1, "x": 2}`` would load as ``{"x": 2}``."""
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise ValueError(f"the key {key!r} is not a string")
    return value


JsonObject = Annotated[dict[str, Any], BeforeValidator(_string_keys), AfterValidator(_freeze)]


class Device(_Model):
    name: str
    vendor: str | None = None
    technology: Technology
    num_qubits: Count
    processor: str | None = None
    calibrated_at: UtcDatetime | None = None

    @field_validator("name", "vendor")
    @classmethod
    def _plain_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value != value.strip() or _FORBIDDEN_NAME.search(value):
            raise ValueError(f"{value!r} must be non-empty, unpadded, and free of @ / \\")
        return value


class Connectivity(_Model):
    directed: Flag = False  # declared first so the edge validator can read it
    edges: tuple[tuple[QubitIndex, QubitIndex], ...]

    @field_validator("edges")
    @classmethod
    def _canonical_edges(
        cls, edges: tuple[tuple[int, int], ...], info: ValidationInfo
    ) -> tuple[tuple[int, int], ...]:
        """Distinct edges, sorted; an undirected edge is written (low, high)."""
        directed = info.data.get("directed", False)
        seen: set[tuple[int, int]] = set()
        for a, b in edges:
            if a == b:
                raise ValueError(f"edge [{a}, {b}] joins a qubit to itself")
            key = (a, b) if directed else (min(a, b), max(a, b))
            if key in seen:
                raise ValueError(f"edge [{a}, {b}] is listed twice")
            seen.add(key)
        return tuple(sorted(seen))


class _GateFields(_Model):
    """Fields shared by a gate definition and a calibration record."""

    avg_infidelity: Real | None = None
    process_infidelity: Real | None = None
    depolarizing_param: Real | None = None
    pauli: tuple[Real, ...] | None = None
    duration_ns: NonNegative | None = None
    virtual: Flag | None = None
    disabled: Flag | None = None
    method: Literal["rb", "irb", "srb", "xeb", "gst", "layer", "model", "vendor"] | None = None
    measured: Literal["isolated", "simultaneous"] | None = None
    statistic: Literal["individual", "median", "mean"] | None = None
    scope: Literal["gate", "cycle"] | None = None
    includes: tuple[Literal["1q_dressing", "leakage", "spam"], ...] | None = None
    stderr: NonNegative | None = None
    assumption: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _units(cls, data: Any) -> Any:
        return normalize_times(data, (DURATION,))

    @field_validator("includes")
    @classmethod
    def _sorted_includes(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        return None if value is None else tuple(sorted(set(value)))

    @model_validator(mode="after")
    def _one_metric(self) -> _GateFields:
        given = [key for key in metrics.METRIC_KEYS if getattr(self, key) is not None]
        if len(given) > 1:
            raise ValueError(f"give one error metric, got {', '.join(given)}")
        return self

    @property
    def metric(self) -> tuple[metrics.MetricKind, Any] | None:
        for key in metrics.METRIC_KEYS:
            value = getattr(self, key)
            if value is not None:
                return key, value
        return None


class GateSpec(_GateFields):
    """A gate definition: arity, direction and device-wide defaults."""

    qubits: Count | None = None
    symmetric: Flag | None = None


class _RecordKey(_Model):
    gate: str
    qubits: tuple[QubitIndex, ...] = Field(min_length=1)


class CalibrationRecord(_GateFields, _RecordKey):
    """Calibration of one gate on specific qubits; overrides the definition field by field."""


class Readout(_Model):
    p1_given_0: Probability | None = None
    p0_given_1: Probability | None = None
    error: Probability | None = None
    duration_ns: NonNegative | None = None

    @model_validator(mode="before")
    @classmethod
    def _units(cls, data: Any) -> Any:
        return normalize_times(data, (DURATION,))

    @model_validator(mode="after")
    def _one_form(self) -> Readout:
        pair = (self.p1_given_0 is not None, self.p0_given_1 is not None)
        if self.error is not None and any(pair):
            raise ValueError("give either error or p1_given_0 + p0_given_1, not both")
        if self.error is None and not all(pair):
            raise ValueError("readout needs error, or both p1_given_0 and p0_given_1")
        return self

    @property
    def pair(self) -> tuple[float, float]:
        """(P(1|0), P(0|1))."""
        if self.error is not None:
            return self.error, self.error
        return self.p1_given_0, self.p0_given_1  # type: ignore[return-value]


class Prep(_Model):
    error: Probability


class Idle(_Model):
    t1_us: Positive | None = None
    t2_us: Positive | None = None
    t2_kind: Literal["echo", "ramsey", "cpmg"] | None = None
    dephasing_rate_per_s: NonNegative | None = None

    @model_validator(mode="before")
    @classmethod
    def _units(cls, data: Any) -> Any:
        return normalize_times(data, (T1, T2))


class _QubitKey(_Model):
    index: QubitIndex


class QubitRecord(Idle, _QubitKey):
    """Per-qubit values; idle fields override ``idle`` one by one, readout and prep as a whole."""

    readout: Readout | None = None
    prep: Prep | None = None
    label: str | None = None
    coords: tuple[Real, ...] | None = None
    disabled: Flag | None = None


class Effect(_Model):
    """Physics the format records but no adapter implements yet (always reported omitted)."""

    type: EffectType
    gate: str | None = None
    on: Literal["readout", "idle"] | None = None
    qubits: tuple[QubitIndex, ...] | None = None
    prob: Probability | None = None
    rate_per_s: NonNegative | None = None
    strength_hz: Real | None = None
    angle_rad: Real | None = None
    heralded: Flag | None = None
    allow: Literal["omit", "approximate", "exact"] = "omit"

    @model_validator(mode="after")
    def _target(self) -> Effect:
        if (self.gate is None) == (self.on is None):
            raise ValueError("an effect names exactly one of gate or on")
        return self


class Provenance(_Model):
    data_kind: Literal["measured", "vendor_model", "spec_sheet", "hypothetical", "unknown"] = (
        "unknown"
    )
    source_kind: (
        Literal[
            "package_snapshot",
            "public_api",
            "account_api",
            "user_file",
            "published_data",
            "vendor_sample",
            "hand_written",
            "derived",
            "other",
        ]
        | None
    ) = None
    source: str | None = None
    source_url: str | None = None
    license: str | None = None
    attribution: str | None = None
    redistributable: Literal["yes", "no", "unknown"] = "unknown"
    retrieved_at: UtcDatetime | None = None
    source_hash: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None = None
    tool: str | None = None
    derived_from: str | None = None
    notes: tuple[str, ...] = ()
    extra: JsonObject = Field(default_factory=FrozenDict)


# "IBM Quantum, via qiskit-ibm-runtime", "X (via Y)", "X via Y"
_VIA = re.compile(r"(?P<who>.+?)(?:,\s*|\s*\(\s*|\s+)via\s+(?P<via>.+?)\)?")
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex(text: str) -> str:
    return "".join(_LATEX_SPECIALS.get(char, char) for char in text)


def merge_spec(definition: GateSpec, record: CalibrationRecord) -> GateSpec:
    """Apply a record to its definition: field by field, except the metric is replaced whole."""
    base = definition.model_dump(exclude_none=True)
    override = record.model_dump(exclude_none=True, exclude={"gate", "qubits"})
    if any(key in override for key in metrics.METRIC_KEYS):
        for key in metrics.METRIC_KEYS:
            base.pop(key, None)
    return GateSpec.model_validate(base | override)


def _without_defaults(name: str, spec: GateSpec) -> GateSpec:
    """Drop an arity or direction that only restates the registry default."""
    info = gates.lookup(name)
    drop: dict[str, None] = {}
    if spec.symmetric is not None and spec.symmetric == gates.is_symmetric(name):
        drop["symmetric"] = None
    if info is not None and spec.qubits == info.arity:
        drop["qubits"] = None
    return spec.model_copy(update=drop) if drop else spec


def _spec_issues(spec: GateSpec, arity: int) -> list[str]:
    issues = []
    metric = spec.metric
    if metric is not None:
        try:
            metrics.check_metric(metric[0], metric[1], arity)
        except ValueError as exc:
            issues.append(str(exc))
    if spec.virtual and (metric is not None or (spec.duration_ns or 0) > 0):
        issues.append("a virtual gate carries no error metric and no duration")
    return issues


class Profile(_Model):
    noisevault: Literal["1.0"]
    device: Device
    connectivity: Literal["all_to_all"] | Connectivity
    gates: Annotated[dict[str, GateSpec], BeforeValidator(_string_keys)]
    readout: Readout | None = None
    prep: Prep | None = None
    idle: Idle | None = None
    qubits: tuple[QubitRecord, ...] = ()
    calibrations: tuple[CalibrationRecord, ...] = ()
    effects: tuple[Effect, ...] = ()
    benchmarks: JsonObject = Field(default_factory=FrozenDict)
    provenance: Provenance = Field(default_factory=Provenance)
    extensions: JsonObject = Field(default_factory=FrozenDict)

    # canonical form: entry order and restated registry defaults do not change the physics,
    # so they change neither the saved file nor the fingerprint

    @field_validator("gates")
    @classmethod
    def _canonical_gates(cls, value: dict[str, GateSpec]) -> FrozenDict:
        return FrozenDict({name: _without_defaults(name, spec) for name, spec in value.items()})

    @field_validator("qubits")
    @classmethod
    def _qubits_by_index(cls, value: tuple[QubitRecord, ...]) -> tuple[QubitRecord, ...]:
        return tuple(sorted(value, key=lambda q: q.index))

    @field_validator("calibrations")
    @classmethod
    def _records_by_locus(
        cls, value: tuple[CalibrationRecord, ...]
    ) -> tuple[CalibrationRecord, ...]:
        return tuple(sorted(value, key=lambda r: (r.gate, r.qubits)))

    @field_validator("effects")
    @classmethod
    def _effects_by_content(cls, value: tuple[Effect, ...]) -> tuple[Effect, ...]:
        return tuple(
            sorted(
                value, key=lambda e: canonical_json(e.model_dump(mode="json", exclude_none=True))
            )
        )

    @model_validator(mode="after")
    def _consistent(self) -> Profile:
        issues = _profile_issues(self)
        if issues:
            raise ValueError("\n".join(issues))
        return self

    # identity and hashes -------------------------------------------------------------------

    @property
    def id(self) -> str:
        return profile_id(self.device.vendor, self.device.name)

    @cached_property
    def fingerprint(self) -> str:
        """sha256 of the canonical physics: everything except provenance and extensions."""
        physics = {k: v for k, v in self.to_dict().items() if k not in ("provenance", "extensions")}
        return _sha256(physics)

    @cached_property
    def artifact_hash(self) -> str:
        """sha256 of the canonical JSON of the whole profile."""
        return _sha256(self.to_dict())

    @property
    def short_fingerprint(self) -> str:
        return "nv:" + self.fingerprint[:12]

    @cached_property
    def table(self) -> NoiseTable:
        from .table import NoiseTable

        return NoiseTable(self)

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Profile:
        """A copy; with ``update`` the result is validated like any new profile."""
        if update:
            return type(self).model_validate({**self.to_dict(), **update})
        copy = super().model_copy(deep=deep)
        copy.__dict__.pop("table", None)  # the cached table points at the original
        return copy

    # serialization -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
        for key in ("qubits", "calibrations", "effects", "benchmarks", "extensions"):
            if not data.get(key):
                data.pop(key, None)
        return data

    def to_json(self) -> str:
        """Readable canonical JSON: one line per section entry, gate and record."""
        return _readable_json(self.to_dict())

    def save(self, path: str | Path) -> Path:
        """Write canonical JSON, replacing ``path``; a ``.gz`` suffix writes reproducible gzip."""
        path = Path(path)
        if path.suffix == ".gz":
            text = json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)
            data = gzip_reproducibly(text.encode("utf-8"))
        else:
            data = (self.to_json() + "\n").encode("utf-8")
        write_atomically(path, data)
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        """Validate a profile dict; a format 0.1 dict is upgraded in memory with a warning."""
        if compat.is_v01(data):
            data = compat.upgrade_v01(data)
            warnings.warn(
                "upgraded a NoiseVault 0.1 file to format 1.0 in memory; save it to keep 1.0",
                MigrationWarning,
                stacklevel=2,
            )
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        return load_file(path)

    @classmethod
    def uniform(
        cls,
        name: str,
        *,
        technology: Technology,
        num_qubits: int,
        one_qubit_error: float,
        two_qubit_error: float | None = None,
        readout_error: float | None = None,
        t1_us: float | None = None,
        t2_us: float | None = None,
        one_qubit_ns: float | None = None,
        two_qubit_ns: float | None = None,
        connectivity: Literal["all_to_all"] | list[tuple[int, int]] = "all_to_all",
    ) -> Profile:
        """A hypothetical device where every gate of an arity has the same error.

        Every registry 1-qubit unitary gate, and on two or more qubits every 2-qubit one, is
        defined (except z-family gates, free through a virtual ``rz``, and multi-entangler gates,
        which must be decomposed), so any circuit of such gates resolves to calibrated noise
        without approximation. Only a device of two or more qubits needs ``two_qubit_error`` and
        uses ``two_qubit_ns``.
        """
        if num_qubits >= 2 and two_qubit_error is None:
            raise ValueError(
                f"a {num_qubits}-qubit device needs two_qubit_error, the average infidelity"
                " of its 2-qubit gates"
            )
        defs: dict[str, dict[str, Any]] = {"rz": {"virtual": True}}
        for info in gates.GATES.values():
            if info.unitary is None or info.family == "z" or info.multi_entangler:
                continue
            if info.arity == 1:
                defs[info.name] = {"avg_infidelity": one_qubit_error, "duration_ns": one_qubit_ns}
            elif info.arity == 2 and num_qubits >= 2:
                defs[info.name] = {"avg_infidelity": two_qubit_error, "duration_ns": two_qubit_ns}
        return cls.model_validate(
            {
                "noisevault": FORMAT_VERSION,
                "device": {"name": name, "technology": technology, "num_qubits": num_qubits},
                "connectivity": connectivity
                if connectivity == "all_to_all"
                else {"edges": connectivity, "directed": False},
                "gates": defs,
                "readout": None if readout_error is None else {"error": readout_error},
                "idle": {"t1_us": t1_us, "t2_us": t2_us}
                if t1_us is not None or t2_us is not None
                else None,
                "provenance": {"data_kind": "hypothetical", "source_kind": "hand_written"},
            }
        )

    # presentation --------------------------------------------------------------------------

    def __repr__(self) -> str:
        dev = self.device
        when = dev.calibrated_at.date().isoformat() if dev.calibrated_at else "undated"
        return (
            f"<Profile {self.id}@{when} {dev.technology} {dev.num_qubits}q"
            f" {self.short_fingerprint}>"
        )

    __str__ = __repr__

    def summary(self) -> str:
        dev, table, prov = self.device, self.table, self.provenance
        when = dev.calibrated_at.date().isoformat() if dev.calibrated_at else "undated"
        lines = [
            f"{self.id}@{when}  {dev.technology}, {dev.num_qubits} qubits,"
            f" {self.short_fingerprint}",
            f"source: {prov.source or 'unknown'} (license {prov.license or 'unknown'},"
            f" redistributable {prov.redistributable})",
        ]
        for name in self.gates:
            lines.append(
                f"  {name:<10} {table.arity(name)}q  {_describe_loci(gate_loci(self, name))}"
            )
        qubits = [table.qubit(i) for i in range(dev.num_qubits)]
        t1 = [q.t1_ns / 1000 for q in qubits if q.t1_ns is not None]
        readout = [sum(q.readout) / 2 for q in qubits if q.readout is not None]
        if t1:
            lines.append(f"  median T1 {_median(t1):.4g} us")
        lines.append(
            f"  median readout error {_median(readout):.3g}" if readout else "  readout unknown"
        )
        return "\n".join(lines)

    def citation(self, style: Literal["text", "bibtex"] = "text") -> str:
        dev, prov = self.device, self.provenance
        when = _iso_z(dev.calibrated_at) if dev.calibrated_at else "undated"
        who = prov.attribution or dev.vendor or "unknown source"
        if style == "text":
            return (
                f"{who}. Calibration of {dev.name}, {when}. {prov.source or 'source unknown'}. "
                f"NoiseVault profile {self.id}, fingerprint sha256:{self.fingerprint}."
            )
        dated = dev.calibrated_at is not None
        key = re.sub(r"[^a-z0-9]+", "_", f"{self.id}_{when[:10]}" if dated else self.id)
        title = f"Calibrated noise of {dev.name}" + (f" at {when}" if dated else "")
        year = f"  year = {{{when[:4]}}},\n" if dated else ""
        via = _VIA.fullmatch(who)
        retrieved = f"Retrieved via {via['via']}. " if via else ""
        published = f"NoiseVault profile {self.id}, sha256:{self.fingerprint}"
        note = f"{retrieved}Source: {prov.source or 'unknown'}; license {prov.license or 'unknown'}"
        # Double braces: BibTeX would lowercase the title and split an organization into
        # first and last names.
        return (
            f"@misc{{nv_{key},\n"
            f"  title = {{{{{_latex(title)}}}}},\n"
            f"  author = {{{{{_latex(via['who'] if via else who)}}}}},\n"
            f"{year}"
            f"  howpublished = {{{_latex(published)}}},\n"
            f"  note = {{{_latex(note)}}}\n"
            f"}}"
        )

    # operations implemented in other modules -----------------------------------------------

    def suggest_layout(self, n: int) -> dict[int, int]:
        from .layout import suggest_layout

        return suggest_layout(self, n)

    def diff(self, other: Profile, *, top: int = 5) -> ProfileDiff:
        from .diff import diff

        return diff(self, other, top=top)

    def check(
        self,
        *,
        frameworks: Sequence[str] | None = None,
        layout: Mapping[Hashable, int] | Sequence[int] | None = None,
        shots: int = 20_000,
        seed: int | None = 0,
    ) -> CheckResult:
        from .check import check

        return check(self, frameworks=frameworks, layout=layout, shots=shots, seed=seed)

    def to_qiskit(self, *, unknown_gates: str = "typical", **options: Any) -> Any:
        from .frameworks.qiskit import to_qiskit

        return to_qiskit(self, unknown_gates=unknown_gates, **options)

    def to_cirq(self, *, layout: Any = None, unknown_gates: str = "typical", **options: Any) -> Any:
        from .frameworks.cirq import to_cirq

        return to_cirq(self, layout=layout, unknown_gates=unknown_gates, **options)

    def to_pennylane(
        self, *, layout: Any = None, unknown_gates: str = "typical", **options: Any
    ) -> Any:
        from .frameworks.pennylane import to_pennylane

        return to_pennylane(self, layout=layout, unknown_gates=unknown_gates, **options)

    def to_stim(
        self,
        circuit: Any,
        *,
        layout: Any = None,
        readout: str = "symmetrize",
        tick_ns: float | None = None,
        existing_noise: str = "error",
        unknown_gates: str = "typical",
        **options: Any,
    ) -> Any:
        from .frameworks.stim import to_stim

        return to_stim(
            self,
            circuit,
            layout=layout,
            readout=readout,
            tick_ns=tick_ns,
            existing_noise=existing_noise,
            unknown_gates=unknown_gates,
            **options,
        )


def gate_loci(profile: Profile, name: str) -> list[GateNoise]:
    """Gate ``name`` resolved on every locus it can run on, as the exports resolve it.

    The candidates are each enabled qubit or connected pair (in both orders) and each recorded
    locus. A symmetric gate's pair counts once when both orders resolve to the same state, error
    and duration, and twice when a record makes them differ. A locus the table refuses is left
    out.
    """
    from .table import GateNoise

    table = profile.table
    arity = table.arity(name)
    enabled = [q for q in range(table.num_qubits) if not table.qubit(q).disabled]
    candidates: Iterable[tuple[int, ...]]
    if arity == 1:
        candidates = [(q,) for q in enabled]
    elif table.all_to_all:
        candidates = permutations(enabled, arity)
    else:
        pairs = table.listed_pairs() if arity == 2 else []
        recorded = [r.qubits for r in profile.calibrations if r.gate == name]
        candidates = sorted({*recorded, *(p for a, b in pairs for p in ((a, b), (b, a)))})
    symmetric = arity == 2 and table.symmetric(name)
    found: list[GateNoise] = []
    for qubits in candidates:
        noise = table.gate(name, qubits)
        if not isinstance(noise, GateNoise):
            continue
        if symmetric and qubits[0] > qubits[1]:
            lower = table.gate(name, qubits[::-1])
            if isinstance(lower, GateNoise) and _resolved_alike(lower, noise):
                continue
        found.append(noise)
    return found


def _resolved_alike(a: GateNoise, b: GateNoise) -> bool:
    return (a.state, a.avg_infidelity, a.duration_ns) == (b.state, b.avg_infidelity, b.duration_ns)


_STATE_WORDS = {"ideal": "virtual", "uncalibrated": "no error metric", "disabled": "disabled"}


def _describe_loci(found: list[GateNoise]) -> str:
    """One gate's resolved loci in words: its calibrated error, then each other state's count."""
    if not found:
        return "usable on no locus"
    states = Counter(noise.state for noise in found)
    errors = [noise.avg_infidelity for noise in found if noise.avg_infidelity is not None]
    parts = []
    if errors and states["calibrated"] == len(found) and len(set(errors)) == 1:
        parts.append(f"avg infidelity {errors[0]:.3g} everywhere")
    elif errors:
        parts.append(f"median avg infidelity {_median(errors):.3g} over {_loci(len(errors))}")
    for state, word in _STATE_WORDS.items():
        if states[state]:
            parts.append(
                word if states[state] == len(found) else f"{word} on {_loci(states[state])}"
            )
    return ", ".join(parts)


def _loci(count: int) -> str:
    return f"{count} {'locus' if count == 1 else 'loci'}"


def _profile_issues(profile: Profile) -> list[str]:
    """Cross-field rules of the format; each issue names where it is."""
    issues: list[str] = []
    n = profile.device.num_qubits
    if not _ID.match(profile.id):
        issues.append(
            f"device: the profile id {profile.id!r} (from vendor and name) may only use letters,"
            " digits and _ . - so that refs can name it"
        )
    arity: dict[str, int] = {}
    for name, spec in profile.gates.items():
        info = gates.lookup(name)
        if info is None and spec.qubits is None:
            issues.append(f"gates.{name}: not a registry gate, so state its arity with qubits")
            continue
        if info is not None and spec.qubits is not None and spec.qubits != info.arity:
            issues.append(f"gates.{name}: qubits={spec.qubits} but {name} acts on {info.arity}")
            continue
        arity[name] = spec.qubits or info.arity  # type: ignore[union-attr]
        issues += [f"gates.{name}: {msg}" for msg in _spec_issues(spec, arity[name])]

    if isinstance(profile.connectivity, Connectivity):
        for a, b in profile.connectivity.edges:
            if max(a, b) >= n:
                issues.append(f"connectivity edge [{a}, {b}] is outside 0..{n - 1}")

    seen_qubits: set[int] = set()
    for record in profile.qubits:
        if record.index >= n:
            issues.append(f"qubits: index {record.index} is outside 0..{n - 1}")
        if record.index in seen_qubits:
            issues.append(f"qubits: index {record.index} is listed twice")
        seen_qubits.add(record.index)

    seen_loci: set[tuple[str, tuple[int, ...]]] = set()
    for record in profile.calibrations:
        where = f"calibrations ({record.gate} on {list(record.qubits)})"
        if record.gate not in profile.gates:
            issues.append(f"{where}: gate {record.gate!r} is not defined in gates")
            continue
        if record.gate not in arity:
            continue  # its definition is already reported
        if len(record.qubits) != arity[record.gate]:
            issues.append(f"{where}: {record.gate} acts on {arity[record.gate]} qubits")
        if len(set(record.qubits)) != len(record.qubits):
            issues.append(f"{where}: targets must be distinct")
        if any(q >= n for q in record.qubits):
            issues.append(f"{where}: qubit outside 0..{n - 1}")
        key = (record.gate, record.qubits)
        if key in seen_loci:
            issues.append(f"{where}: a second record for the same gate and qubits")
        seen_loci.add(key)
        merged = merge_spec(profile.gates[record.gate], record)
        issues += [f"{where}: {msg}" for msg in _spec_issues(merged, arity[record.gate])]

    for effect in profile.effects:
        where = f"effects ({effect.type} on {effect.gate or effect.on})"
        if effect.gate is not None and effect.gate not in profile.gates:
            issues.append(f"{where}: gate {effect.gate!r} is not defined in gates")
        if effect.qubits and max(effect.qubits) >= n:
            issues.append(f"{where}: qubit outside 0..{n - 1}")
    return issues


# helpers ----------------------------------------------------------------------------------


def canonical_json(data: Any) -> str:
    """Sorted keys, no whitespace, floats as repr(float): the text every hash is taken over."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _readable_json(data: dict[str, Any]) -> str:
    def flat(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def block(value: Any, indent: str) -> str:
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            items = [f"{indent}  {flat(k)}: {flat(v)}" for k, v in value.items()]
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            items = [f"{indent}  {flat(v)}" for v in value]
        else:
            return flat(value)
        open_, close = ("{", "}") if isinstance(value, dict) else ("[", "]")
        return open_ + "\n" + ",\n".join(items) + "\n" + indent + close

    body = ",\n".join(f"  {flat(k)}: {block(v, '  ')}" for k, v in data.items())
    return "{\n" + body + "\n}"


def _sha256(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def profile_id(vendor: str | None, name: str) -> str:
    """``name`` if it already starts with ``vendor_``, else ``vendor_name``; lowercase."""
    name = re.sub(r"\s+", "-", name.lower())
    if vendor is None:
        return name
    vendor = re.sub(r"\s+", "-", vendor.lower())
    return name if name.startswith(vendor + "_") else f"{vendor}_{name}"


# No file name, mtime 0, best compression, OS "unknown". gzip.compress(mtime=0) before Python
# 3.13 lets zlib write its platform's OS byte instead, so the same profile got other bytes.
_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"


def gzip_reproducibly(data: bytes) -> bytes:
    """``data`` as gzip whose bytes depend only on ``data``, not on the Python that wrote it."""
    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    body = deflate.compress(data) + deflate.flush()
    return _GZIP_HEADER + body + struct.pack("<II", zlib.crc32(data), len(data) & 0xFFFFFFFF)


def write_atomically(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data``: a failed write leaves the old file whole.

    The temporary file's dot name keeps vault listings from seeing it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        tmp = Path(handle.name)
    try:
        if path.exists():
            shutil.copymode(path, tmp)
        else:  # NamedTemporaryFile creates 0600; a new file should get the usual mode
            mask = os.umask(0)
            os.umask(mask)
            tmp.chmod(0o666 & ~mask)
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_file(path: str | Path) -> Profile:
    """Read a profile from ``.json`` or gzip-compressed JSON (0.1 files are upgraded)."""
    return load_bytes(Path(path).read_bytes())


def load_bytes(raw: bytes) -> Profile:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return Profile.from_dict(json.loads(raw))


def json_schema() -> dict[str, Any]:
    """JSON Schema of format 1.0 (structure only; cross-field rules live in the validator)."""
    return Profile.model_json_schema()


@dataclass(frozen=True)
class Ref:
    """A catalog reference: ``id``, ``id@YYYY-MM-DD`` or ``id@<ISO timestamp>``."""

    id: str
    date: date | None = None
    timestamp: datetime | None = None


_ID = re.compile(r"^[a-z0-9][a-z0-9_.\-]*$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_ref(ref: str | Path) -> Path | Ref:
    """A path (has a separator, a .json/.gz suffix, or exists) or a catalog :class:`Ref`."""
    if isinstance(ref, Path):
        return ref
    text = ref.strip()
    if "/" in text or "\\" in text or text.endswith((".json", ".gz")) or Path(text).exists():
        return Path(text)
    ident, sep, at = text.lower().partition("@")
    if not _ID.match(ident):
        raise ValueError(f"{ref!r} is not a profile id such as ibm_fez or ibm_fez@2025-02-26")
    if not sep:
        return Ref(ident)
    if _DATE.match(at):
        try:
            return Ref(ident, date=date.fromisoformat(at))
        except ValueError:
            raise ValueError(
                f"{ref!r}: {at} is not a calendar date; give one such as 2025-02-26"
            ) from None
    try:
        stamp = datetime.fromisoformat(at.upper())
    except ValueError:
        stamp = None
    if stamp is None or stamp.tzinfo is None:
        raise ValueError(f"{ref!r}: after @ give a date (2025-02-26) or a timestamp with timezone")
    return Ref(ident, timestamp=stamp.astimezone(UTC))
