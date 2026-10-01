"""The conversion report every framework export carries as ``.report``.

Adapters fill one Report per exported object; events keep accumulating as circuits are
processed. The same helpers decide how clamps, effects and repeated warnings are recorded, so
every framework reports the same way.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import NoiseApproximationWarning, UnsupportedEffect

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .channels import GateChannels
    from .profile import Effect, GateSpec, Profile

HONESTY = "Calibration-derived models approximate the hardware; they are not a digital twin."
_INCLUDES = {  # includes item -> what the model does with it
    "1q_dressing": "explicit single-qubit gates in the circuit add their own error on top",
    "leakage": "applied as depolarizing noise; no population leaves the qubit",
    "spam": "readout and preparation noise, where applied, count state prep and measurement again",
}


@dataclass(frozen=True)
class Approximation:
    what: str
    how: str
    detail: str = ""


@dataclass(frozen=True)
class Clamp:
    gate: str
    qubits: tuple[int, ...]
    requested: float
    achieved: float


@dataclass
class Report:
    profile_id: str
    fingerprint: str
    framework: str
    framework_version: str | None
    noisevault_version: str
    options: dict[str, Any] = field(default_factory=dict)
    exact: list[str] = field(default_factory=list)
    approximated: list[Approximation] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    clamped: list[Clamp] = field(default_factory=list)
    events: dict[str, Counter[str]] = field(default_factory=dict)
    _warned: set[str] = field(default_factory=set, repr=False, compare=False)

    @classmethod
    def start(
        cls, profile: Profile, framework: str, framework_version: str | None, **options: Any
    ) -> Report:
        from . import __version__

        return cls(
            profile_id=profile.id,
            fingerprint=profile.fingerprint,
            framework=framework,
            framework_version=framework_version,
            noisevault_version=__version__,
            options=options,
        )

    # recording ------------------------------------------------------------------------------

    def mark_exact(self, what: str) -> None:
        _append_new(self.exact, what)

    def approximate(self, what: str, how: str, detail: str = "") -> None:
        _append_new(self.approximated, Approximation(what, how, detail))

    def omit(self, what: str) -> None:
        _append_new(self.omitted, what)

    def mark_unknown(self, what: str) -> None:
        _append_new(self.unknown, what)

    def count(self, event: str, key: str, n: int = 1) -> None:
        self.events.setdefault(event, Counter())[key] += n

    def record_channels(self, built: GateChannels) -> None:
        """Record the bookkeeping of one gate's channels: clamps, qualifiers, reversed records."""
        gate = built.gate
        if built.inexact:
            if not any(c.gate == gate.gate and c.qubits == gate.qubits for c in self.clamped):
                self.clamped.append(
                    Clamp(gate.gate, gate.qubits, built.requested, built.achieved)  # type: ignore[arg-type]
                )
        for q in built.t2_clamped:
            self.record_t2_clamp(q)
        self._record_qualifiers(gate.gate, gate.spec)
        if gate.origin == "reversed_record":
            self.count("reversed_record_used", gate.gate)

    def record_t2_clamp(self, qubit: int) -> None:
        self.approximate(f"T2 of qubit {qubit}", "clamped to 2*T1", "the stated T2 exceeds 2*T1")

    def _record_qualifiers(self, name: str, spec: GateSpec) -> None:
        """Qualifiers that make the stated number differ from the error of the gate alone."""
        what = f"{name} error"
        if spec.scope == "cycle":
            self.approximate(
                what,
                "a per-cycle error applied to each gate",
                "it also counts the surrounding layer",
            )
        for item in spec.includes or ():
            self.approximate(what, f"the stated error includes {item}", _INCLUDES[item])
        if spec.statistic in ("median", "mean"):
            self.approximate(what, f"a device {spec.statistic} applied to every locus")
        if spec.assumption:
            self.approximate(what, "read under an importer assumption", spec.assumption)

    def record_effects(self, effects: Iterable[Effect]) -> None:
        """Effects are not implemented by any adapter in this release: omit or refuse."""
        for effect in effects:
            target = effect.gate or effect.on
            if effect.allow != "omit":
                raise UnsupportedEffect(
                    f"effect {effect.type} on {target} asks for allow={effect.allow!r}, but"
                    f" {self.framework} export does not model effects yet; set allow to 'omit'"
                    " to convert without it"
                )
            self.omit(f"effect {effect.type} on {target}")

    def warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(message, NoiseApproximationWarning, stacklevel=3)

    # output ---------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "fingerprint": self.fingerprint,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "noisevault_version": self.noisevault_version,
            "options": _jsonable(self.options),
            "exact": list(self.exact),
            "approximated": [a.__dict__.copy() for a in self.approximated],
            "omitted": list(self.omitted),
            "unknown": list(self.unknown),
            "clamped": [
                {
                    "gate": c.gate,
                    "qubits": list(c.qubits),
                    "requested": c.requested,
                    "achieved": c.achieved,
                }
                for c in self.clamped
            ],
            "events": {name: dict(counts) for name, counts in self.events.items()},
        }

    def summary(self) -> str:
        version = f" {self.framework_version}" if self.framework_version else ""
        lines = [
            f"NoiseVault {self.noisevault_version} -> {self.framework}{version}:"
            f" {self.profile_id} (nv:{self.fingerprint[:12]})"
        ]
        if self.options:
            lines.append("options: " + ", ".join(f"{k}={v!r}" for k, v in self.options.items()))
        if self.exact:
            lines.append("exact: " + ", ".join(self.exact))
        for a in self.approximated:
            detail = f" ({a.detail})" if a.detail else ""
            lines.append(f"approximated: {a.what}: {a.how}{detail}")
        if self.omitted:
            lines.append("omitted: " + ", ".join(self.omitted))
        if self.unknown:
            lines.append("unknown (no noise applied): " + ", ".join(self.unknown))
        noisier = [c for c in self.clamped if c.achieved > c.requested]
        quieter = [c for c in self.clamped if c.achieved < c.requested]
        if noisier:
            lines.append(
                f"clamped: {len(noisier)} gate(s) noisier than stated because relaxation"
                f" alone exceeds the stated error; largest {_worst(noisier)}"
            )
        if quieter:
            lines.append(
                f"clamped: {len(quieter)} gate(s) less noisy than stated because the strongest"
                f" depolarizing noise on top of relaxation falls short; largest {_worst(quieter)}"
            )
        for name, counts in self.events.items():
            lines.append(f"{name}: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
        lines.append(HONESTY)
        return "\n".join(lines)


def _worst(clamps: list[Clamp]) -> str:
    c = max(clamps, key=lambda c: abs(c.achieved - c.requested))
    return f"{c.gate}{list(c.qubits)} {c.requested:.3g} -> {c.achieved:.3g}"


def _append_new(items: list, item: Any) -> None:
    if item not in items:
        items.append(item)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)
