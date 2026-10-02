"""The NoiseVault profile format 1.0: models, validation, hashing, ids and refs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import struct
import warnings
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
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
    SerializerFunctionWrapHandler,
    Strict,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)

from . import __version__, compat, gates, metrics
from .errors import MigrationWarning
from .units import DURATION, T1, T2, normalize_times

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    from .check import CheckResult
    from .compare import Comparison
    from .counts import MeasuredCounts
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
Sha256 = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CountsSource = Literal["hardware", "simulated"]
Bound = Literal["lower", "upper"]
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
    source_hash: Sha256 | None = None
    tool: str | None = None
    derived_from: str | None = None
    notes: tuple[str, ...] = ()
    extra: JsonObject = Field(default_factory=FrozenDict)


class ErrorFactor(_Model):
    """One factor on the profile's own error rates, with its 95% interval when fitted.

    ``bound`` says the interval reached an end of the fit's domain, so only one side is a
    confidence limit: "lower" keeps ``high`` only (printed "at most"), "upper" keeps ``low``
    only (printed "at least"). A hand-written factor has no interval and no bound.
    """

    factor: NonNegative
    low: NonNegative | None = None
    high: NonNegative | None = None
    bound: Bound | None = None

    @model_validator(mode="after")
    def _interval(self) -> ErrorFactor:
        if self.bound == "lower" and (self.low is not None or self.high is None):
            raise ValueError('bound "lower" needs high and no low')
        if self.bound == "upper" and (self.high is not None or self.low is None):
            raise ValueError('bound "upper" needs low and no high')
        if self.bound is None and (self.low is None) != (self.high is None):
            raise ValueError(
                'give low and high together, or set bound "lower" (high only) or "upper" (low only)'
            )
        if self.low is not None and self.factor < self.low:
            raise ValueError(f"factor {self.factor} is below low {self.low}")
        if self.high is not None and self.factor > self.high:
            raise ValueError(f"factor {self.factor} is above high {self.high}")
        return self


class CountsFit(_Model):
    """The counts the factors were fitted to. Every field is required, ``p_value`` included."""

    counts: Sha256
    source: CountsSource
    qubits: Annotated[tuple[QubitIndex, ...], Field(min_length=1)]
    run_at: UtcDatetime
    calibration: Fingerprint
    p_value: Probability | None
    impossible_shots: Annotated[int, Strict(), Field(ge=0)]

    @model_validator(mode="after")
    def _consistent(self) -> CountsFit:
        repeated = [q for q, times in Counter(self.qubits).items() if times > 1]
        if repeated:
            raise ValueError(f"qubit {repeated[0]} is listed twice")
        if self.impossible_shots and self.p_value != 0:
            raise ValueError(f"impossible_shots is {self.impossible_shots}, so p_value must be 0")
        return self

    @model_serializer(mode="wrap")
    def _every_field(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep a null p_value, which a profile's exclude_none dump would otherwise drop."""
        data = handler(self)
        return {name: data.get(name) for name in type(self).model_fields}


class UnmodeledError(_Model):
    """Error the calibration leaves out, as factors on the profile's own error rates.

    ``gates`` scales every calibrated gate error, ``readout`` scales P(1|0) and P(0|1)
    together. T1, T2, dephasing, preparation error, durations and effects are never scaled.
    """

    gates: ErrorFactor | None = None
    readout: ErrorFactor | None = None
    fit: CountsFit | None = None

    @model_validator(mode="after")
    def _shape(self) -> UnmodeledError:
        if self.gates is None and self.readout is None:
            raise ValueError("give a gates factor, a readout factor or both")
        for name, factor in (("gates", self.gates), ("readout", self.readout)):
            if factor is None:
                continue
            interval = factor.low is not None or factor.high is not None
            if self.fit is not None and not interval:
                raise ValueError(
                    f"{name}: a fitted factor states its interval; give low and high,"
                    " or one of them with bound"
                )
            if self.fit is None and interval:
                raise ValueError(
                    f"{name}: an interval needs the fit it came from;"
                    " add fit, or drop low, high and bound"
                )
        return self

    def lines(self) -> tuple[str, ...]:
        """The phrases ``nv show`` prints one per line and reports join with "; "."""
        phrases = [f"{axis} errors {_describe_factor(f)}" for axis, f in self._factors()]
        fit = self.fit
        if fit is not None:
            short = fit.counts.removeprefix("sha256:")[:12]
            phrases.append(f"fitted to {fit.source} counts sha256:{short}")
            qubits = "-".join(map(str, fit.qubits))
            on = f"on qubit {qubits}" if len(fit.qubits) == 1 else f"on qubits {qubits}"
            run = fit.run_at.date().isoformat()
            phrases.append(f"{on}, run {run} ({_describe_p(fit.p_value)})")
            if fit.impossible_shots:
                shots = "shot" if fit.impossible_shots == 1 else "shots"
                phrases.append(f"the profile ruled out {fit.impossible_shots} {shots}")
        phrases.append("T1, T2 and preparation error are not scaled")
        return tuple(phrases)

    def _factors(self) -> tuple[tuple[str, ErrorFactor], ...]:
        named = (("gate", self.gates), ("readout", self.readout))
        return tuple((axis, factor) for axis, factor in named if factor is not None)


def _describe_factor(factor: ErrorFactor) -> str:
    """The factor and its interval in words, such as "x0.096 (95% interval, at most 0.311)"."""
    text = f"x{factor.factor:.3g}"
    if factor.bound == "lower":
        return f"{text} (95% interval, at most {factor.high:.3g})"
    if factor.bound == "upper":
        return f"{text} (95% interval, at least {factor.low:.3g})"
    if factor.low is not None:
        return f"{text} (95% interval {factor.low:.3g} to {factor.high:.3g})"
    return text


POOR_FIT_P_VALUE = 0.01


def _describe_p(p_value: float | None) -> str:
    if p_value is None:
        return "fit not testable"
    return f"p = {p_value:.2g}" + (", a poor fit" if p_value < POOR_FIT_P_VALUE else "")


def _unmodeled_clause(unmodeled: UnmodeledError | None) -> str:
    """What a citation adds after the fingerprint for a profile with unmodeled-error factors."""
    if unmodeled is None:
        return ""
    rates = ", ".join(f"{axis} error rates x{f.factor:.3g}" for axis, f in unmodeled._factors())
    fit = unmodeled.fit
    fitted = "" if fit is None else f" fitted to {fit.source} counts {fit.counts}"
    return f"; unmodeled-error factors ({rates}){fitted}"


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
    unmodeled_error: UnmodeledError | None = None
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

    # pydantic revalidates, copies and pickles an instance through __dict__, so caches use slots
    __slots__ = ("_fingerprint", "_artifact_hash", "_table")

    @property
    def fingerprint(self) -> str:
        """sha256 of the canonical physics: everything except provenance and extensions."""
        try:
            return self._fingerprint
        except AttributeError:
            physics = {k: v for k, v in self.to_dict().items() if k not in _NOT_PHYSICS}
            object.__setattr__(self, "_fingerprint", _sha256(physics))
            return self._fingerprint

    @property
    def artifact_hash(self) -> str:
        """sha256 of the canonical JSON of the whole profile."""
        try:
            return self._artifact_hash
        except AttributeError:
            object.__setattr__(self, "_artifact_hash", _sha256(self.to_dict()))
            return self._artifact_hash

    @property
    def short_fingerprint(self) -> str:
        return "nv:" + self.fingerprint[:12]

    @property
    def table(self) -> NoiseTable:
        try:
            return self._table
        except AttributeError:
            from .table import NoiseTable

            object.__setattr__(self, "_table", NoiseTable(self))
            return self._table

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Profile:
        """A copy; with ``update`` the result is validated like any new profile."""
        if update:
            return type(self).model_validate({**self.to_dict(), **update})
        return super().model_copy(deep=deep)

    def uncorrected(self) -> Profile:
        """This profile without ``unmodeled_error``: the calibration as stated."""
        if self.unmodeled_error is None:
            return self
        return self.model_copy(update={"unmodeled_error": None})

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
        dev, prov, stated = self.device, self.provenance, self.uncorrected()
        when = dev.calibrated_at.date().isoformat() if dev.calibrated_at else "undated"
        lines = [
            f"{self.id}@{when}  {dev.technology}, {dev.num_qubits} qubits,"
            f" {self.short_fingerprint}",
            f"source: {prov.source or 'unknown'} (license {prov.license or 'unknown'},"
            f" redistributable {prov.redistributable})",
        ]
        for name in self.gates:
            lines.append(
                f"  {name:<10} {stated.table.arity(name)}q"
                f"  {_describe_loci(gate_stats(stated, name))}"
            )
        medians = qubit_medians(stated)
        if medians.t1_us is not None:
            lines.append(f"  median T1 {medians.t1_us:.4g} us")
        lines.append(
            "  readout unknown"
            if medians.readout_error is None
            else f"  median readout error {medians.readout_error:.3g}"
        )
        note = unmodeled_note(self)
        if note:
            lines.append(f"  unmodeled error: {'; '.join(note)}")
        return "\n".join(lines)

    def citation(self, style: Literal["text", "bibtex"] = "text") -> str:
        dev, prov = self.device, self.provenance
        when = _iso_z(dev.calibrated_at) if dev.calibrated_at else "undated"
        who = prov.attribution or dev.vendor or "unknown source"
        ref = f"{self.id}@{when}" if dev.calibrated_at else self.id
        profile = f"NoiseVault {__version__} profile {ref}"
        pinned = f"sha256:{self.fingerprint}{_unmodeled_clause(self.unmodeled_error)}"
        if style == "text":
            return (
                f"{who}. Calibration of {dev.name}, {when}. {prov.source or 'source unknown'}. "
                f"{profile}, fingerprint {pinned}."
            )
        dated = dev.calibrated_at is not None
        key = re.sub(r"[^a-z0-9]+", "_", f"{self.id}_{when[:10]}" if dated else self.id)
        title = f"Calibrated noise of {dev.name}" + (f" at {when}" if dated else "")
        year = f"  year = {{{when[:4]}}},\n" if dated else ""
        via = _VIA.fullmatch(who)
        retrieved = f"Retrieved via {via['via']}. " if via else ""
        published = f"{profile}, {pinned}"
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

    def compare(self, counts: MeasuredCounts) -> Comparison:
        from .compare import compare

        return compare(self, counts)

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


@dataclass(frozen=True)
class QubitMedians:
    t1_us: float | None
    t2_us: float | None
    readout_error: float | None
    p1_given_0: float | None
    p0_given_1: float | None


def qubit_medians(profile: Profile) -> QubitMedians:
    table = profile.table
    working = [q for q in map(table.qubit, range(table.num_qubits)) if not q.disabled]
    readout = [q.readout for q in working if q.readout is not None]
    return QubitMedians(
        t1_us=_median([q.t1_ns / 1000 for q in working if q.t1_ns is not None]),
        t2_us=_median([q.t2_ns / 1000 for q in working if q.t2_ns is not None]),
        readout_error=_median([(a + b) / 2 for a, b in readout]),
        p1_given_0=_median([a for a, _ in readout]),
        p0_given_1=_median([b for _, b in readout]),
    )


@dataclass(frozen=True)
class GateStats:
    states: Counter[GateState]
    errors: tuple[float, ...]
    durations_ns: tuple[float, ...]

    @property
    def median_error(self) -> float | None:
        return _median(self.errors)

    @property
    def median_duration_ns(self) -> float | None:
        return _median(self.durations_ns)


def gate_stats(profile: Profile, name: str) -> GateStats:
    found = gate_loci(profile, name)
    usable = [g for g in found if g.state != "disabled"]
    return GateStats(
        states=Counter(g.state for g in found),
        errors=tuple(g.avg_infidelity for g in usable if g.avg_infidelity is not None),
        durations_ns=tuple(g.duration_ns for g in usable if g.duration_ns is not None),
    )


_STATE_WORDS = {"ideal": "virtual", "uncalibrated": "no error metric", "disabled": "disabled"}


def _describe_loci(stats: GateStats) -> str:
    """One gate's resolved loci in words: its calibrated error, then each other state's count."""
    states, errors, total = stats.states, stats.errors, stats.states.total()
    if not total:
        return "usable on no locus"
    parts = []
    if errors and states["calibrated"] == total and len(set(errors)) == 1:
        parts.append(f"avg infidelity {errors[0]:.3g} everywhere")
    elif errors:
        parts.append(f"median avg infidelity {stats.median_error:.3g} over {_loci(len(errors))}")
    for state, word in _STATE_WORDS.items():
        if states[state]:
            parts.append(word if states[state] == total else f"{word} on {_loci(states[state])}")
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

    unmodeled = profile.unmodeled_error
    if unmodeled is None:
        return issues
    stated_readout = profile.readout is not None or any(
        q.readout is not None for q in profile.qubits
    )
    if unmodeled.readout is not None and not stated_readout:
        issues.append(
            "unmodeled_error.readout: the profile states no readout error,"
            " so the factor scales nothing"
        )
    fit = unmodeled.fit
    if fit is None:
        return issues
    issues += [
        f"unmodeled_error.fit.qubits: qubit {q} is outside 0..{n - 1}" for q in fit.qubits if q >= n
    ]
    calibration = _calibration_fingerprint(profile)
    if fit.calibration != calibration:
        issues.append(
            f"unmodeled_error.fit.calibration: fitted to calibration nv:{fit.calibration[:12]},"
            f" but this profile's calibration is nv:{calibration[:12]};"
            " drop unmodeled_error or refit with nv compare"
        )
    return issues


_NOT_PHYSICS = ("provenance", "extensions")


def _calibration_fingerprint(profile: Profile) -> str:
    """``profile.uncorrected().fingerprint``, computed without building a second profile."""
    skip = (*_NOT_PHYSICS, "unmodeled_error")
    return _sha256({k: v for k, v in profile.to_dict().items() if k not in skip})


def unmodeled_note(profile: Profile) -> tuple[str, ...]:
    """The unmodeled-error phrases ``nv show``, reports and ``summary()`` print; () without it."""
    if profile.unmodeled_error is None:
        return ()
    return (*profile.unmodeled_error.lines(), *profile.table.unscaled())


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


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


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


_NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)


def write_atomically(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data``: a failed write leaves the old file whole.

    The temporary file gets the mode of the file it replaces, so nobody who cannot read the
    old file can read the new contents. Python can read the umask only by setting it, and all
    threads share one umask, so this function never touches the umask.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = 0o666
    while True:
        hidden_tmp = path.with_name(f".{path.name}.{os.urandom(8).hex()}.tmp")
        try:
            fd = os.open(hidden_tmp, _NEW_FILE, mode)
        except FileExistsError:
            continue
        break
    try:
        with open(fd, "wb") as handle:
            handle.write(data)
        if path.exists():
            shutil.copymode(path, hidden_tmp)
        os.replace(hidden_tmp, path)
    except BaseException:
        hidden_tmp.unlink(missing_ok=True)
        raise


def load_file(path: str | Path) -> Profile:
    """Read a profile from ``.json`` or gzip-compressed JSON (0.1 files are upgraded)."""
    return load_bytes(Path(path).read_bytes())


def load_bytes(raw: bytes) -> Profile:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    try:
        data = json.loads(raw)
    except RecursionError:
        raise _too_deep(raw.decode(json.detect_encoding(raw), "surrogatepass")) from None
    return Profile.from_dict(data)


_STRING_OR_BRACKET = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]')


def _too_deep(text: str) -> json.JSONDecodeError:
    """The parse error for JSON nested deeper than ``json.loads`` takes, at its deepest point."""
    depth = deepest = at = 0
    for token in _STRING_OR_BRACKET.finditer(text):
        if token[0] in ("[", "{"):
            depth += 1
            if depth > deepest:
                deepest, at = depth, token.start()
        elif token[0] in ("]", "}"):
            depth -= 1
    return json.JSONDecodeError(f"nested {deepest} levels deep", text, at)


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
