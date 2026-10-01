"""Drift between two profiles: device-wide medians, the largest per-qubit and per-pair changes,
gates and qubits that were disabled or re-enabled, and qubits added or removed.

Per-qubit values are the resolved ones (a qubit record, else the device default). The typical
1-qubit and 2-qubit errors are those of the native gate the conversion rules would use there
(:meth:`NoiseTable.typical`). On all-to-all devices the pairs compared are those with a
2-qubit calibration record, or one pair carrying the device default when there is none.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from .table import GateNoise

if TYPE_CHECKING:
    from .profile import Profile
    from .table import NoiseTable

Pair = tuple[int, int]
# metric name, unit for display, and whether a larger value is better
METRICS: dict[str, tuple[str, bool]] = {
    "t1_us": ("T1 (us)", True),
    "t2_us": ("T2 (us)", True),
    "error_1q": ("1q error", False),
    "error_2q": ("2q error", False),
    "readout_error": ("readout error", False),
}


@dataclass(frozen=True)
class Change:
    """One value before and after; ``where`` is "" for a device median, else the qubit or pair."""

    metric: str
    before: float | None
    after: float | None
    where: str = ""

    @property
    def relative(self) -> float | None:
        if self.before is None or self.after is None or self.before == 0:
            return None
        return (self.after - self.before) / self.before

    @property
    def worse(self) -> bool:
        """True when the change makes the device noisier."""
        rel = self.relative
        return rel is not None and (rel < 0 if METRICS[self.metric][1] else rel > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "where": self.where or None,
            "before": self.before,
            "after": self.after,
            "relative": self.relative,
        }


@dataclass(frozen=True)
class ProfileDiff:
    before: str  # ref of the first profile
    after: str
    before_fingerprint: str
    after_fingerprint: str
    time_delta: timedelta | None
    medians: tuple[Change, ...]
    qubits: tuple[Change, ...]  # largest per-qubit relative changes, largest first
    pairs: tuple[Change, ...]
    newly_disabled: tuple[str, ...]
    reenabled: tuple[str, ...]
    qubits_added: tuple[int, ...]
    qubits_removed: tuple[int, ...]
    warnings: tuple[str, ...]

    @property
    def identical(self) -> bool:
        return self.before_fingerprint == self.after_fingerprint

    def summary(self) -> str:
        lines = [f"{self.before} -> {self.after} ({describe_delta(self.time_delta)})"]
        lines += [f"warning: {w}" for w in self.warnings]
        if self.identical:
            lines.append("same physics: the fingerprints are equal")
            return "\n".join(lines)
        lines.append(f"  {'device median':<22}{'before':>12}{'after':>12}{'change':>10}")
        for c in self.medians:
            lines.append(
                f"  {METRICS[c.metric][0]:<22}{fmt_metric(c.metric, c.before):>12}"
                f"{fmt_metric(c.metric, c.after):>12}{fmt_relative(c.relative):>10}"
            )
        for title, changes in (("qubits", self.qubits), ("pairs", self.pairs)):
            if changes:
                lines.append(f"largest changes by {title[:-1]}:")
                lines += [
                    f"  {_where(title, c.where):<14} {METRICS[c.metric][0]:<14}"
                    f" {fmt_metric(c.metric, c.before)} -> {fmt_metric(c.metric, c.after)}"
                    f" ({fmt_relative(c.relative)})"
                    for c in changes
                ]
        for label, items in (
            ("newly disabled", self.newly_disabled),
            ("re-enabled", self.reenabled),
            ("qubits added", tuple(map(str, self.qubits_added))),
            ("qubits removed", tuple(map(str, self.qubits_removed))),
        ):
            if items:
                lines.append(f"{label}: {', '.join(items)}")
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "identical": self.identical,
            "time_delta_seconds": None
            if self.time_delta is None
            else self.time_delta.total_seconds(),
            "medians": [c.to_dict() for c in self.medians],
            "qubits": [c.to_dict() for c in self.qubits],
            "pairs": [c.to_dict() for c in self.pairs],
            "newly_disabled": list(self.newly_disabled),
            "reenabled": list(self.reenabled),
            "qubits_added": list(self.qubits_added),
            "qubits_removed": list(self.qubits_removed),
            "warnings": list(self.warnings),
        }


def diff(a: Profile, b: Profile, *, top: int = 5) -> ProfileDiff:
    """How ``b`` differs from ``a``; ``top`` limits the per-qubit and per-pair lists."""
    if top < 0:
        raise ValueError(f"top={top}: give 0 or more")
    warnings = []
    if a.id != b.id:
        warnings.append(
            f"these are different devices ({a.id} and {b.id}); qubits and pairs are matched"
            " by index"
        )
    when_a, when_b = a.device.calibrated_at, b.device.calibrated_at
    delta = when_b - when_a if when_a and when_b else None
    if delta is not None and delta < timedelta(0):
        warnings.append(
            f"{_ref(b)} is older than {_ref(a)}; before and after follow argument order, not time"
        )

    qa, qb = _qubit_values(a.table), _qubit_values(b.table)
    pa, pb = _pair_values(a.table), _pair_values(b.table)
    medians = tuple(
        Change(
            m, _median(pa if m == "error_2q" else qa, m), _median(pb if m == "error_2q" else qb, m)
        )
        for m in METRICS
    )
    newly_disabled, reenabled = _availability(a, b)
    common = range(min(a.device.num_qubits, b.device.num_qubits))
    return ProfileDiff(
        before=_ref(a),
        after=_ref(b),
        before_fingerprint=a.fingerprint,
        after_fingerprint=b.fingerprint,
        time_delta=delta,
        medians=medians,
        qubits=_largest(qa, qb, lambda q: str(q), top),
        pairs=_largest(pa, pb, lambda p: f"{p[0]}-{p[1]}", top),
        newly_disabled=newly_disabled,
        reenabled=reenabled,
        qubits_added=tuple(q for q in range(b.device.num_qubits) if q not in common),
        qubits_removed=tuple(q for q in range(a.device.num_qubits) if q not in common),
        warnings=tuple(warnings),
    )


# values -------------------------------------------------------------------------------------


def _qubit_values(table: NoiseTable) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for i in range(table.num_qubits):
        q = table.qubit(i)
        if q.disabled:
            continue
        values = {
            "t1_us": None if q.t1_ns is None else q.t1_ns / 1000,
            "t2_us": None if q.t2_ns is None else q.t2_ns / 1000,
            "error_1q": _error(table.typical(1, (i,))),
            "readout_error": None if q.readout is None else sum(q.readout) / 2,
        }
        out[i] = {k: v for k, v in values.items() if v is not None}
    return out


def _pair_values(table: NoiseTable) -> dict[Pair, dict[str, float]]:
    out: dict[Pair, dict[str, float]] = {}
    for a, b in _pairs(table):
        errors = [_error(table.typical(2, pair)) for pair in ((a, b), (b, a))]
        known = [e for e in errors if e is not None]
        if known:
            out[(a, b)] = {"error_2q": min(known)}
    return out


def _pairs(table: NoiseTable) -> list[Pair]:
    if not table.all_to_all:
        return table.edges()
    arity = table.arity
    loci = {
        (min(r.qubits), max(r.qubits))
        for r in table.profile.calibrations
        if len(r.qubits) == 2 and arity(r.gate) == 2
    }
    return sorted(loci) or ([(0, 1)] if table.num_qubits > 1 else [])


def _error(found: Any) -> float | None:
    return found.avg_infidelity if isinstance(found, GateNoise) else None


def _median(values: dict[Any, dict[str, float]], metric: str) -> float | None:
    found = [v[metric] for v in values.values() if metric in v]
    return statistics.median(found) if found else None


def _largest(
    before: dict[Any, dict[str, float]],
    after: dict[Any, dict[str, float]],
    label: Callable[[Any], str],
    top: int,
) -> tuple[Change, ...]:
    moved = [
        (key, Change(metric, before[key][metric], value, label(key)))
        for key in before.keys() & after.keys()
        for metric, value in after[key].items()
        if metric in before[key] and value != before[key][metric]
    ]
    moved = [(key, c) for key, c in moved if c.relative is not None]
    moved = _collapse_uniform(moved, before.keys() & after.keys())
    moved.sort(key=lambda kc: (-abs(kc[1].relative), *_order(kc[0]), kc[1].metric))  # type: ignore[arg-type]
    return tuple(c for _, c in moved[:top])


def _order(key: Any) -> tuple[bool, Any]:
    return (False, ()) if key == "all" else (True, key)


def _collapse_uniform(moved: list[tuple[Any, Change]], keys: set[Any]) -> list[tuple[Any, Change]]:
    """One "all" row per metric that changed the same way everywhere (device-wide values)."""
    out = []
    for metric in dict.fromkeys(c.metric for _, c in moved):
        rows = [(key, c) for key, c in moved if c.metric == metric]
        if len(rows) == len(keys) and len({(c.before, c.after) for _, c in rows}) == 1:
            c = rows[0][1]
            out.append(("all", Change(metric, c.before, c.after, "all")))
        else:
            out += rows
    return out


def _availability(a: Profile, b: Profile) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Gate loci and qubits usable in ``a`` and disabled in ``b``, and the reverse.

    A locus that one profile does not have at all (a directed gate calibrated in the other
    direction, a qubit past the end) is neither: it was not usable before, or is not now.
    """
    loci = sorted({*_disabled(a), *_disabled(b)})
    state_a, state_b = [_state(a, k) for k in loci], [_state(b, k) for k in loci]
    labels = [f"{gate or 'qubit'} {'-'.join(map(str, qubits))}" for gate, qubits in loci]
    pairs = list(zip(labels, state_a, state_b, strict=True))
    return (
        tuple(label for label, x, y in pairs if x == "usable" and y == "disabled"),
        tuple(label for label, x, y in pairs if x == "disabled" and y == "usable"),
    )


def _disabled(profile: Profile) -> set[tuple[str, tuple[int, ...]]]:
    """Disabled gate loci as (gate, qubits) and disabled qubits as ("", (index,))."""
    gates = {(r.gate, r.qubits) for r in profile.calibrations if r.disabled}
    return gates | {("", (q.index,)) for q in profile.qubits if q.disabled}


def _state(profile: Profile, locus: tuple[str, tuple[int, ...]]) -> str | None:
    """ "usable", "disabled", or None when the profile has no such gate locus or qubit."""
    gate, qubits = locus
    table = profile.table
    if not gate:
        if qubits[0] >= table.num_qubits:
            return None
        return "disabled" if table.qubit(qubits[0]).disabled else "usable"
    found = table.gate(gate, qubits)
    if not isinstance(found, GateNoise):
        return None
    return "disabled" if found.state == "disabled" else "usable"


def _where(title: str, where: str) -> str:
    return f"all {title}" if where == "all" else f"{title[:-1]} {where}"


def _ref(profile: Profile) -> str:
    when = profile.device.calibrated_at
    return f"{profile.id}@{when.date().isoformat()}" if when else profile.id


# formatting ---------------------------------------------------------------------------------


def fmt_error(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


def fmt_time(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def fmt_metric(metric: str, value: float | None) -> str:
    return fmt_time(value) if metric in ("t1_us", "t2_us") else fmt_error(value)


def fmt_relative(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1%}"


def describe_delta(delta: timedelta | None) -> str:
    if delta is None:
        return "calibration time unknown"
    if delta == timedelta(0):
        return "same calibration time"
    span = abs(delta)
    days, hours = span.days, span.seconds // 3600
    minutes = span.seconds % 3600 // 60
    parts = [_plural(days, "day"), _plural(hours, "hour")] if days or hours else []
    text = " ".join(p for p in parts if p) or _plural(minutes, "minute") or "under a minute"
    return f"{text} {'later' if delta > timedelta(0) else 'earlier'}"


def _plural(n: int, unit: str) -> str:
    return "" if n == 0 else f"{n} {unit}{'' if n == 1 else 's'}"
