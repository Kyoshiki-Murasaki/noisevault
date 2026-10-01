"""The NoiseVault profile format 1.0: models, validation, hashing, ids and refs."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
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
    from collections.abc import Hashable, Mapping, Sequence

    from .check import CheckResult
    from .diff import ProfileDiff
    from .table import NoiseTable

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
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FrozenDict(dict):
    """A dict that refuses changes, so a validated profile cannot drift from its fingerprint."""

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("a Profile is immutable; use profile.model_copy(update=...) to change it")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self) -> tuple[type, tuple[dict]]:
        return FrozenDict, (dict(self),)


def _freeze(value: Any) -> Any:
    """Read-only copy of JSON-like data: dicts become FrozenDict, lists become tuples."""
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


JsonObject = Annotated[dict[str, Any], AfterValidator(_freeze)]


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
    gates: dict[str, GateSpec]
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
        """Write canonical JSON; a ``.gz`` suffix writes reproducible gzip."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            text = json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)
            path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        else:
            path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        """Validate a profile dict; a format 0.1 dict is upgraded in memory with a warning."""
        if compat.is_v01(data):
            warnings.warn(
                "upgraded a NoiseVault 0.1 file to format 1.0 in memory; save it to keep 1.0",
                MigrationWarning,
                stacklevel=2,
            )
            data = compat.upgrade_v01(data)
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
        two_qubit_error: float,
        readout_error: float | None = None,
        t1_us: float | None = None,
        t2_us: float | None = None,
        one_qubit_ns: float | None = None,
        two_qubit_ns: float | None = None,
        connectivity: Literal["all_to_all"] | list[tuple[int, int]] = "all_to_all",
    ) -> Profile:
        """A hypothetical device where every gate of an arity has the same error.

        Every registry 1-qubit and 2-qubit unitary gate is defined (except z-family gates, free
        through a virtual ``rz``, and multi-entangler gates, which must be decomposed), so any
        circuit of such gates resolves to calibrated noise without approximation.
        """
        defs: dict[str, dict[str, Any]] = {"rz": {"virtual": True}}
        for info in gates.GATES.values():
            if info.unitary is None or info.family == "z" or info.multi_entangler:
                continue
            if info.arity == 1:
                defs[info.name] = {"avg_infidelity": one_qubit_error, "duration_ns": one_qubit_ns}
            elif info.arity == 2:
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
        from .table import GateNoise

        dev, table, prov = self.device, self.table, self.provenance
        when = dev.calibrated_at.date().isoformat() if dev.calibrated_at else "undated"
        lines = [
            f"{self.id}@{when}  {dev.technology}, {dev.num_qubits} qubits,"
            f" {self.short_fingerprint}",
            f"source: {prov.source or 'unknown'} (license {prov.license or 'unknown'},"
            f" redistributable {prov.redistributable})",
        ]
        errors: dict[str, list[float]] = {}
        for record in self.calibrations:
            found = table.gate(record.gate, record.qubits)
            if isinstance(found, GateNoise) and found.avg_infidelity is not None:
                errors.setdefault(record.gate, []).append(found.avg_infidelity)
        for name, spec in self.gates.items():
            arity = table.arity(name)
            if spec.virtual:
                lines.append(f"  {name:<10} virtual")
            elif name in errors:
                values = errors[name]
                lines.append(
                    f"  {name:<10} {arity}q  median avg infidelity {_median(values):.3g}"
                    f" over {len(values)} records"
                )
            elif spec.metric is not None:
                r = metrics.to_avg_infidelity(*spec.metric, arity)  # type: ignore[arg-type]
                lines.append(f"  {name:<10} {arity}q  avg infidelity {r:.3g} everywhere")
            else:
                lines.append(f"  {name:<10} {arity}q  no error metric")
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
        return (
            f"@misc{{nv_{key},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{who}}},\n"
            f"{year}"
            f"  howpublished = {{NoiseVault profile {self.id}, sha256:{self.fingerprint}}},\n"
            f"  note = {{Source: {prov.source or 'unknown'};"
            f" license {prov.license or 'unknown'}}}\n"
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
    ident, _, at = text.lower().partition("@")
    if not _ID.match(ident):
        raise ValueError(f"{ref!r} is not a profile id such as ibm_fez or ibm_fez@2025-02-26")
    if not at:
        return Ref(ident)
    if _DATE.match(at):
        return Ref(ident, date=date.fromisoformat(at))
    try:
        stamp = datetime.fromisoformat(at.upper())
    except ValueError:
        stamp = None
    if stamp is None or stamp.tzinfo is None:
        raise ValueError(f"{ref!r}: after @ give a date (2025-02-26) or a timestamp with timezone")
    return Ref(ident, timestamp=stamp.astimezone(UTC))
