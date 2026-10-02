"""Score a profile against measured counts, and fit how far its error rates must scale.

``compare`` binds the counts to the profile's calibration, then fits one factor on every gate
error and one on every readout error by multinomial maximum likelihood over all circuits. Only
the reference simulator scores the model, so the fit evaluates the same profile, with the same
factors, that every export applies. Each interval inverts a likelihood-ratio test. Where the
chi-square cutoff would cover too little, parametric draws recalibrate it.
"""

from __future__ import annotations

import math
import textwrap
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import gates, metrics
from .channels import gate_channels
from .conversion import resolve_op
from .errors import (
    CountsError,
    DisabledGateError,
    LayoutError,
    MissingCalibrationError,
    NoiseVaultError,
)
from .profile import (
    POOR_FIT_P_VALUE,
    Bound,
    ErrorFactor,
    UnmodeledError,
    _calibration_fingerprint,
    _describe_factor,
    _iso_z,
)
from .reference import Op, charged_as, probabilities
from .report import Report
from .table import GateNoise, _no_power, _qubits

if TYPE_CHECKING:
    from .counts import MeasuredCounts, PlannedCircuit
    from .profile import Profile

FACTOR_RANGE = (0.05, 20.0)
LEVEL = 0.95
CHI2_95 = 3.841458820694124
GATE_NODES = 25
FINE_POINTS = 193
RESAMPLES = 400
WINDOW_POINTS = 33
END_TOL = 0.02
IMPOSSIBLE = 1e-12
MIN_INFORMATIVE_SHOTS = 100
SEPARABLE = 1e-6
FD_STEP = 1e-3
NOTE = (
    "Factors multiply the profile's error rates, so x2 means about twice the errors.",
    "Fitted on these qubits, they absorb crosstalk, leakage, coherent error and",
    "idle error beyond T1 and T2. Saved with -o, they apply to every qubit.",
)
NEXT_READOUT = (
    "add a circuit with no gates, which only readout error moves.",
    "The circuits from noisevault.counts.plan() include one.",
)

Axis = Literal["gate", "readout"]
_AXES: tuple[Axis, Axis] = ("gate", "readout")
_LO, _HI = math.log(FACTOR_RANGE[0]), math.log(FACTOR_RANGE[1])
_STEP = (_HI - _LO) / (FINE_POINTS - 1)
_MIN_WINDOW = 0.05
_SAME_WAY = "gate error and readout error move these counts the same way"
_LABEL = 16
_WIDTH = 80
_CACHE_SIZE = 4
_CHUNK = 1 << 22


@dataclass(frozen=True)
class Estimate:
    factor: float
    low: float | None
    high: float | None
    bound: Bound | None

    def describe(self) -> str:
        """The factor and its 95% interval, such as "x1.84 (95% interval 1.54 to 2.12)"."""
        return _describe_factor(
            ErrorFactor(factor=self.factor, low=self.low, high=self.high, bound=self.bound)
        )


@dataclass(frozen=True)
class NoEstimate:
    reason: str


FactorResult = Estimate | NoEstimate


@dataclass(frozen=True)
class CircuitScore:
    name: str
    qubits: tuple[int, ...]
    shots: int
    tvd_profile: float
    tvd_fitted: float
    shot_noise_95: float
    impossible: int


@dataclass(frozen=True)
class Comparison:
    profile: Profile
    counts: MeasuredCounts = field(repr=False)
    circuits: tuple[CircuitScore, ...]
    gates: FactorResult
    readout: FactorResult
    deviance: float
    dof: int
    p_value: float | None
    ruled_out: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def impossible_shots(self) -> int:
        return sum(score.impossible for score in self.circuits)

    @property
    def dispersion(self) -> float:
        """How far the deviance exceeds its degrees of freedom; 1 when it cannot be tested."""
        return max(1.0, self.deviance / self.dof) if self.dof > 0 else 1.0

    @property
    def calibration_age(self) -> timedelta | None:
        when = self.profile.device.calibrated_at
        return None if when is None else self.counts.run_at - when

    def fitted_profile(self) -> Profile:
        """The calibration with an ``unmodeled_error`` block that records this fit."""
        if isinstance(self.gates, NoEstimate):
            if isinstance(self.readout, NoEstimate):
                raise NoiseVaultError(
                    "gate and readout factors are not identified; nothing to save"
                )
            raise NoiseVaultError("the gate factor is not identified; nothing to save")
        base = self.profile.uncorrected()
        block: dict[str, Any] = {"gates": _saved(self.gates)}
        if isinstance(self.readout, Estimate):
            block["readout"] = _saved(self.readout)
        block["fit"] = {
            "counts": self.counts.sha256,
            "source": self.counts.source,
            "qubits": list(self.counts.qubits),
            "run_at": self.counts.run_at,
            "calibration": base.fingerprint,
            "p_value": self.p_value,
            "impossible_shots": self.impossible_shots,
        }
        return base.model_copy(update={"unmodeled_error": block})

    def summary(self, *, counts_file: str | None = None) -> str:
        """The text ``nv compare`` prints, every line at most 80 columns for the check plan."""
        lines = [*self._header(counts_file), "", *self._table(), "", *self._findings()]
        if self._fitted:
            lines += ["", *NOTE]
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        age = self.calibration_age
        return {
            "profile_id": self.profile.id,
            "fingerprint": self.profile.fingerprint,
            "calibration": _calibration_fingerprint(self.profile),
            "counts": self.counts.sha256,
            "source": self.counts.source,
            "backend": self.counts.backend,
            "run_at": _iso_z(self.counts.run_at),
            "calibration_age_hours": None if age is None else age.total_seconds() / 3600,
            "qubits": list(self.counts.qubits),
            "circuits": [
                {
                    "name": s.name,
                    "qubits": list(s.qubits),
                    "shots": s.shots,
                    "tvd_profile": s.tvd_profile,
                    "tvd_fitted": s.tvd_fitted,
                    "shot_noise_95": s.shot_noise_95,
                    "impossible": s.impossible,
                }
                for s in self.circuits
            ],
            "gates": _factor_dict(self.gates),
            "readout": _factor_dict(self.readout),
            "deviance": self.deviance,
            "dof": self.dof,
            "dispersion": self.dispersion,
            "p_value": self.p_value,
            "resamples": RESAMPLES,
            "impossible_shots": self.impossible_shots,
            "ruled_out": list(self.ruled_out),
            "notes": list(self.notes),
            "note": " ".join(NOTE) if self._fitted else None,
        }

    @property
    def _fitted(self) -> bool:
        return isinstance(self.gates, Estimate) or isinstance(self.readout, Estimate)

    def _header(self, counts_file: str | None) -> list[str]:
        profile, counts = self.profile, self.counts
        when = profile.device.calibrated_at
        ref = profile.id if when is None else f"{profile.id}@{when.date().isoformat()}"
        digest = counts.sha256.removeprefix("sha256:")[:12]
        source = (
            f"{counts.source} counts sha256:{digest}"
            if counts_file is None
            else f"counts {counts_file}, {counts.source}, sha256:{digest}"
        )
        age = self.calibration_age
        since = (
            "calibration age unknown (the profile has no date)"
            if age is None
            else f"{_duration(age)} after calibration"
        )
        run = counts.run_at.strftime("%Y-%m-%d %H:%MZ")
        return [
            f"{ref} {profile.short_fingerprint} on {_on(counts.qubits)}",
            source,
            f"run {run}, {since}",
        ]

    def _table(self) -> list[str]:
        width = max(len("circuit"), *(len(s.name) for s in self.circuits))
        shots = max(len("shots"), *(len(str(s.shots)) for s in self.circuits))
        impossible = self.impossible_shots > 0
        head = f"{'circuit':<{width}}  {'shots':>{shots}}  profile TVD  fitted TVD  noise TVD 95%"
        rows = [head + ("  impossible shots" if impossible else "")]
        for s in self.circuits:
            row = (
                f"{s.name:<{width}}  {s.shots:>{shots}}  {s.tvd_profile:>11.4f}"
                f"  {s.tvd_fitted:>10.4f}  {s.shot_noise_95:>13.4f}"
            )
            if impossible:
                row += f"  {s.impossible:>16}"
            elif _beyond_noise(s):
                row += "  beyond noise"
            rows.append(row)
        return rows

    def _findings(self) -> list[str]:
        gate, readout = self.gates, self.readout
        shared = (
            isinstance(gate, NoEstimate)
            and isinstance(readout, NoEstimate)
            and gate.reason == readout.reason
        )
        entries = [
            ("gate errors", [_say(gate), *([gate.reason] if _reason(gate) and not shared else [])]),
            ("readout errors", [_say(readout), *([readout.reason] if _reason(readout) else [])]),
            ("fit", self._verdict()),
        ]
        if _SAME_WAY in (_reason(gate), _reason(readout)):
            entries.append(("next", list(NEXT_READOUT)))
        if self.notes:
            entries.append(("note", list(self.notes)))
        lines = []
        for label, values in entries:
            for i, value in enumerate(values):
                wrapped = textwrap.wrap(value, _WIDTH - _LABEL, break_on_hyphens=False)
                lines.append(f"{label if i == 0 else '':<{_LABEL}}{wrapped[0]}")
                lines += [" " * _LABEL + part for part in wrapped[1:]]
        return lines

    def _verdict(self) -> list[str]:
        flagged = [s.name for s in self.circuits if _beyond_noise(s)]
        impossible = self.impossible_shots
        if impossible:
            total = sum(s.shots for s in self.circuits)
            lines = [
                "ruled out (p = 0)",
                f"the profile gives {_count(impossible, 'shot')} probability 0",
                *self.ruled_out,
            ]
            if impossible < total:
                lines.append(f"the fit uses the other {total - impossible} shots")
            if flagged:
                lines.append(f"beyond shot noise on {_words(flagged)}")
            return lines
        if self.p_value is None:
            return ["not testable (no degrees of freedom left after fitting)"]
        p = f"p = {self.p_value:.2g}"
        if self.p_value < POOR_FIT_P_VALUE:
            where = [f"on {_words(flagged)}"] if flagged else []
            return [f"beyond shot noise ({p})", *where, "no one pair of factors fits every circuit"]
        where = "overall" if flagged else "on every circuit"
        return [f"within shot noise {where} ({p})"]


def compare(profile: Profile, counts: MeasuredCounts) -> Comparison:
    """Score ``profile`` on ``counts`` and fit its gate and readout factors.

    Raises CountsError when the counts were not planned from this profile's calibration, ran
    before it, or contain an op the profile does not calibrate on its qubits.
    """
    base = _bind(profile, counts)
    circuits = counts.circuits
    surface = _Surface.cached(base, circuits)
    measured = np.concatenate([c.vector() for c in circuits]).astype(float)
    observed = np.where(surface.supported, measured, 0.0)
    shots = np.array([observed[part].sum() for part in surface.slices])

    log_gate, log_readout, best = surface.maximum(observed)
    gate = None if surface.gate is None else math.exp(log_gate)
    readout = None if surface.readout is None else math.exp(log_readout)
    fitted = _exact(base, circuits, gate, readout)
    info = _fisher(base, counts, fitted, gate, readout)
    unfitted = _identify(surface, observed, info)
    eigen = np.linalg.eigvalsh(info)
    rank = int(np.sum(eigen > SEPARABLE * eigen[-1])) if eigen[-1] > 0 else 0
    dof = int(surface.supported.sum()) - len(circuits) - rank
    deviance = _deviance(observed, fitted, surface.slices)
    dispersion = max(1.0, deviance / dof) if dof > 0 else 1.0

    seed = int(counts.sha256[7:23], 16)
    fit = _Fit(surface, observed, (log_gate, log_readout, best), shots, dispersion, seed)
    results = {axis: unfitted[axis] or fit.interval(axis) for axis in _AXES}

    draws = fit.draw(fitted, np.random.default_rng(seed))
    every = np.column_stack([observed, draws])
    top, draw_gate, draw_readout = fit.draw_max(every)
    surface_deviance = 2 * (_saturated(every, surface.slices) - top)
    refits = surface.probs_at(draw_gate[1:], draw_readout[1:])

    impossible = measured - observed
    if impossible.sum():
        p_value: float | None = 0.0
    elif dof <= 0:
        p_value = None
    else:
        exceed = np.sum(surface_deviance[1:] >= surface_deviance[0])
        p_value = float((1 + exceed) / (1 + RESAMPLES))

    as_given = _exact(profile, circuits, None, None)
    scores = []
    for c, part, n in zip(circuits, surface.slices, shots, strict=True):
        frequencies = measured[part] / c.shots
        if n:
            noise = 0.5 * np.abs(draws[part].T / n - refits[:, part]).sum(axis=1)
            noise_95 = float(np.percentile(noise, 100 * LEVEL))
        else:
            noise_95 = 0.0
        scores.append(
            CircuitScore(
                name=c.name,
                qubits=tuple(c.qubits),
                shots=c.shots,
                tvd_profile=_tvd(frequencies, as_given[part]),
                tvd_fitted=_tvd(frequencies, fitted[part]),
                shot_noise_95=noise_95,
                impossible=int(impossible[part].sum()),
            )
        )
    return Comparison(
        profile=profile,
        counts=counts,
        circuits=tuple(scores),
        gates=results["gate"],
        readout=results["readout"],
        deviance=deviance,
        dof=dof,
        p_value=p_value,
        ruled_out=_ruled_out(surface, circuits, impossible),
        notes=_notes(base, circuits),
    )


class _Surface:
    """Outcome probabilities at any (log gate factor, log readout factor) point.

    Exact reference runs at the gate nodes give the probabilities before readout. Between
    nodes they mix linearly in log factor, so every vector stays a distribution.
    ``metrics.scale_readout`` then applies readout error exactly. An axis that no circuit
    moves along is None and stays at factor 1.
    """

    _built: OrderedDict[Any, _Surface] = OrderedDict()

    def __init__(
        self,
        nodes: np.ndarray,
        unread: np.ndarray,
        pairs: tuple[tuple[tuple[float, float] | None, ...], ...],
        slices: tuple[slice, ...],
        gate: np.ndarray | None,
        readout: np.ndarray | None,
    ) -> None:
        self.nodes = nodes
        self.unread = unread
        self.pairs = pairs
        self.slices = slices
        self.gate = gate
        self.readout = readout
        reads = _grid(readout)
        peaks = [self.probs_at(np.full(len(reads), x), reads).max(axis=0) for x in nodes]
        self.supported = np.max(peaks, axis=0) > IMPOSSIBLE

    @classmethod
    def cached(cls, base: Profile, circuits: Sequence[PlannedCircuit]) -> _Surface:
        """The surface for this calibration and these circuit definitions, built once.

        The key holds the circuits' qubits and ops, never their counts, so every run of one
        plan shares a surface.
        """
        key = (base.fingerprint, tuple((c.qubits, c.ops) for c in circuits))
        if key in cls._built:
            cls._built.move_to_end(key)
            return cls._built[key]
        surface = cls._built[key] = cls.build(base, circuits)
        if len(cls._built) > _CACHE_SIZE:
            cls._built.popitem(last=False)
        return surface

    @classmethod
    def build(cls, base: Profile, circuits: Sequence[PlannedCircuit]) -> _Surface:
        fine = np.linspace(_LO, _HI, FINE_POINTS)
        nodes = np.unique(
            np.concatenate(
                [
                    fine[:: (FINE_POINTS - 1) // (GATE_NODES - 1)],
                    np.log(_floor_factors(base, circuits)),
                ]
            )
        )
        sizes = [2 ** len(c.qubits) for c in circuits]
        starts = np.cumsum([0, *sizes])
        slices = tuple(slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:], strict=True))

        def run(log_gate: float) -> np.ndarray:
            return _exact(base, circuits, math.exp(log_gate), None, read=False)

        low, high = run(nodes[0]), run(nodes[-1])
        if all(_tvd(low[part], high[part]) < 1e-9 for part in slices):
            gate, nodes, unread = None, np.zeros(1), run(0.0)[None, :]
        else:
            gate = np.unique(np.concatenate([fine, nodes]))
            unread = np.array([low, *(run(x) for x in nodes[1:-1]), high])
        pairs = tuple(tuple(base.table.qubit(q).readout for q in c.qubits) for c in circuits)
        scales = any(
            pair is not None
            and metrics.scale_readout(pair, FACTOR_RANGE[0])
            != metrics.scale_readout(pair, FACTOR_RANGE[1])
            for per_circuit in pairs
            for pair in per_circuit
        )
        return cls(nodes, unread, pairs, slices, gate, fine if scales else None)

    def axis(self, axis: Axis) -> np.ndarray | None:
        return self.gate if axis == "gate" else self.readout

    def probs_at(self, log_gate: np.ndarray, log_readout: np.ndarray) -> np.ndarray:
        """(points, cells) probabilities at each (log gate, log readout) point."""
        unread = self._unread_at(np.asarray(log_gate, dtype=float))
        factors, which = np.unique(np.asarray(log_readout, dtype=float), return_inverse=True)
        which = which.ravel()
        matrices = {
            pair: np.array([_confusion(pair, float(x)) for x in factors])[which]
            for pair in {pair for pairs in self.pairs for pair in pairs if pair is not None}
        }
        out = np.empty_like(unread)
        for part, pairs in zip(self.slices, self.pairs, strict=True):
            block = unread[:, part].reshape(-1, *(2,) * len(pairs))
            for i, pair in enumerate(pairs):
                if pair is not None:
                    block = _read(block, i + 1, matrices[pair])
            out[:, part] = block.reshape(len(unread), -1)
        return out

    def loglik_at(
        self, log_gate: np.ndarray, log_readout: np.ndarray, counts: np.ndarray
    ) -> np.ndarray:
        return _loglik(self.probs_at(log_gate, log_readout), counts)

    def loglik(self, counts: np.ndarray) -> np.ndarray:
        """The log-likelihood on the whole fine grid: (gate, readout) or (gate, readout, draws)."""
        gate, readout = _grid(self.gate), _grid(self.readout)
        points_gate, points_readout = (a.ravel() for a in np.meshgrid(gate, readout, indexing="ij"))
        counts = np.asarray(counts, dtype=float)
        out = np.empty((len(points_gate), *counts.shape[1:]))
        chunk = max(1, _CHUNK // len(counts))
        for start in range(0, len(points_gate), chunk):
            part = slice(start, start + chunk)
            out[part] = self.loglik_at(points_gate[part], points_readout[part], counts)
        return out.reshape(len(gate), len(readout), *counts.shape[1:])

    def maximum(self, counts: np.ndarray) -> tuple[float, float, float]:
        """(log gate, log readout, log-likelihood) at the maximum, refined between grid points.

        Among equally likely points the one nearest factor 1 on both axes wins, so a plateau
        gives its edge. Two nested windows then place the maximum within 1/128 of a grid step.
        """
        gate, readout = _grid(self.gate), _grid(self.readout)
        values = self.loglik(counts)
        i, j = _nearest_one(values, gate[:, None], readout[None, :])
        best_gate, best_readout, best = float(gate[i]), float(readout[j]), float(values[i, j])
        for span in (_STEP, _STEP / 8):
            near_gate = _around(self.gate, best_gate, span)
            near_readout = _around(self.readout, best_readout, span)
            g, r = np.meshgrid(near_gate, near_readout, indexing="ij")
            window = self.loglik_at(g.ravel(), r.ravel(), counts).reshape(g.shape)
            a, b = _nearest_one(window, g, r)
            if window[a, b] >= best:
                best_gate, best_readout, best = float(g[a, b]), float(r[a, b]), float(window[a, b])
        return best_gate, best_readout, best

    def moves(self, axis: Axis) -> list[bool]:
        """Per circuit, whether its outcomes change between the ends of ``axis``."""
        ends, one = np.array([_LO, _HI]), np.zeros(2)
        probs = self.probs_at(ends, one) if axis == "gate" else self.probs_at(one, ends)
        return [_tvd(probs[0, part], probs[1, part]) > 1e-9 for part in self.slices]

    def _unread_at(self, log_gate: np.ndarray) -> np.ndarray:
        if len(self.nodes) == 1:
            return np.repeat(self.unread, len(log_gate), axis=0)
        x = np.clip(log_gate, self.nodes[0], self.nodes[-1])
        k = np.clip(np.searchsorted(self.nodes, x, side="right") - 1, 0, len(self.nodes) - 2)
        w = ((x - self.nodes[k]) / (self.nodes[k + 1] - self.nodes[k]))[:, None]
        return (1 - w) * self.unread[k] + w * self.unread[k + 1]


class _Fit:
    """The profile likelihood around one maximum, and the parametric draws that calibrate it."""

    def __init__(
        self,
        surface: _Surface,
        observed: np.ndarray,
        estimate: tuple[float, float, float],
        shots: np.ndarray,
        dispersion: float,
        seed: int,
    ) -> None:
        self.surface = surface
        self.observed = observed
        self.at: dict[Axis, float] = {"gate": estimate[0], "readout": estimate[1]}
        self.best = estimate[2]
        self.shots = shots
        self.dispersion = dispersion
        self.cutoff = CHI2_95 * dispersion
        self.seed = seed
        self._wilks: dict[Axis, dict[int, float | None]] = {}
        self._window: _Window | None = None

    def interval(self, axis: Axis) -> FactorResult:
        """The 95% interval on ``axis`` by test inversion, with the chi-square cutoff as floor."""
        at = self.at[axis]
        ends: dict[int, float | None] = {}
        for side, edge in ((-1, _LO), (1, _HI)):
            wilks = self.wilks(axis)[side]
            ends[side] = None if wilks is None else self._end(axis, wilks, side, edge)
        low, high = ends[-1], ends[1]
        if low is None and high is None:
            return NoEstimate(f"the counts do not constrain the {axis} factor")
        bound: Bound | None = "lower" if low is None else "upper" if high is None else None
        return Estimate(
            factor=math.exp(at),
            low=None if low is None else math.exp(low),
            high=None if high is None else math.exp(high),
            bound=bound,
        )

    def _end(self, axis: Axis, wilks: float, side: int, edge: float) -> float | None:
        """One end of the interval. It starts at the Wilks end and moves outward while the
        draws keep accepting.

        Steps of 0.25, 0.5, 1, ... Wilks half-widths find a rejected point, then bisection
        closes in to END_TOL of the half-width. The end is None when the test keeps the
        domain end.
        """
        half = max(abs(wilks - self.at[axis]), 1e-6)
        tol = max(END_TOL * half, 1e-6)

        def beyond(x: float) -> bool:
            return (x - edge) * side >= 0

        inside = edge if beyond(wilks + side * tol) else wilks + side * tol
        if not self.keep(axis, inside):
            return wilks
        step = 0.25 * half
        while True:
            if inside == edge:
                return None
            candidate = inside + side * step
            if beyond(candidate):
                if self.keep(axis, edge):
                    return None
                outside = edge
                break
            if not self.keep(axis, candidate):
                outside = candidate
                break
            inside, step = candidate, 2 * step
        while abs(outside - inside) > tol:
            middle = (inside + outside) / 2
            if self.keep(axis, middle):
                inside = middle
            else:
                outside = middle
        return inside

    def wilks(self, axis: Axis) -> dict[int, float | None]:
        """Where the likelihood ratio crosses the chi-square cutoff on each side; None at a
        domain end it never reaches."""
        if axis not in self._wilks:
            at, ends = self.at[axis], {}
            for side, edge in ((-1, _LO), (1, _HI)):
                if self.lr(axis, edge) <= self.cutoff:
                    ends[side] = None
                    continue
                inside, outside = at, edge
                while abs(outside - inside) > 1e-6:
                    middle = (inside + outside) / 2
                    if self.lr(axis, middle) <= self.cutoff:
                        inside = middle
                    else:
                        outside = middle
                ends[side] = inside
            self._wilks[axis] = ends
        return self._wilks[axis]

    def lr(self, axis: Axis, held: float) -> float:
        return 2 * (self.best - self.restricted(axis, held)[1])

    def restricted(self, axis: Axis, held: float) -> tuple[float, float]:
        """(the other axis at its best, the log-likelihood) with ``axis`` held at ``held``."""
        other = self.surface.axis(_other(axis))
        if other is None:
            return 0.0, float(self._line(axis, held, np.zeros(1), self.observed)[0])
        center = float(other[int(np.argmax(self._line(axis, held, other, self.observed)))])
        line = np.linspace(max(center - _STEP, _LO), min(center + _STEP, _HI), WINDOW_POINTS)
        values = self._line(axis, held, line, self.observed)[:, None]
        k = np.argmax(values, axis=0)
        shift, gain = _refine(values, k, line[1] - line[0])
        return float(line[k[0]] + shift[0]), float(values[k[0], 0] + gain[0])

    def keep(self, axis: Axis, held: float) -> bool:
        """True when the test does not reject ``held``. Either the chi-square cutoff accepts
        it, or RESAMPLES draws at ``held`` show that its likelihood ratio is not extreme."""
        observed = self.lr(axis, held)
        if observed <= self.cutoff:
            return True
        nuisance, _ = self.restricted(axis, held)
        point = _point(axis, held, np.array([nuisance]))
        draws = self.draw(self.surface.probs_at(*point)[0], np.random.default_rng(self.seed))
        drawn = 2 * (self.draw_max(draws)[0] - self.draw_restricted(draws, axis, held))
        p = (1 + np.sum(drawn >= observed / self.dispersion - 1e-9)) / (1 + RESAMPLES)
        return bool(p > 1 - LEVEL)

    def draw(self, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """(cells, RESAMPLES) counts drawn from ``probs`` with the observed supported shots."""
        probs = np.where(self.surface.supported, probs, 0.0)
        parts = []
        for part, shots in zip(self.surface.slices, self.shots, strict=True):
            cell = probs[part]
            if shots:
                parts.append(rng.multinomial(int(shots), cell / cell.sum(), size=RESAMPLES).T)
            else:
                parts.append(np.zeros((len(cell), RESAMPLES)))
        return np.concatenate(parts).astype(float)

    def draw_max(self, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each column's (log-likelihood, log gate, log readout) at its maximum on the window
        around the estimate, refined by a parabola along each axis."""
        window = self.window()
        gate, readout = window.gate, window.readout
        values = _loglik(window.probs, draws).reshape(len(gate), len(readout), -1)
        columns = np.arange(values.shape[-1])
        i, j = np.unravel_index(
            np.argmax(values.reshape(-1, len(columns)), axis=0), values.shape[:2]
        )
        shift_gate, gain_gate = _refine(values[:, j, columns], i, _spacing(gate))
        shift_readout, gain_readout = _refine(values[i, :, columns].T, j, _spacing(readout))
        top = values[i, j, columns] + gain_gate + gain_readout
        return top, gate[i] + shift_gate, readout[j] + shift_readout

    def draw_restricted(self, draws: np.ndarray, axis: Axis, held: float) -> np.ndarray:
        """Each column's maximum on the window's line through ``held``, refined by a parabola."""
        line = self.window().gate if axis == "readout" else self.window().readout
        values = self._line(axis, held, line, draws)
        k = np.argmax(values, axis=0)
        return values[k, np.arange(values.shape[1])] + _refine(values, k, _spacing(line))[1]

    def window(self) -> _Window:
        """WINDOW_POINTS per axis spanning 4 Wilks half-widths around the estimate, with the
        probabilities at every window point."""
        if self._window is None:
            spans: dict[Axis, np.ndarray] = {}
            for axis in _AXES:
                if self.surface.axis(axis) is None:
                    spans[axis] = np.zeros(1)
                    continue
                at = self.at[axis]
                ends = [e for e in self.wilks(axis).values() if e is not None]
                half = max(max((abs(e - at) for e in ends), default=(_HI - _LO) / 4), _MIN_WINDOW)
                spans[axis] = np.linspace(
                    max(at - 4 * half, _LO), min(at + 4 * half, _HI), WINDOW_POINTS
                )
            g, r = (a.ravel() for a in np.meshgrid(spans["gate"], spans["readout"], indexing="ij"))
            self._window = _Window(spans["gate"], spans["readout"], self.surface.probs_at(g, r))
        return self._window

    def _line(self, axis: Axis, held: float, others: np.ndarray, counts: np.ndarray) -> np.ndarray:
        return self.surface.loglik_at(*_point(axis, held, others), counts)


@dataclass(frozen=True)
class _Window:
    gate: np.ndarray
    readout: np.ndarray
    probs: np.ndarray


def _bind(profile: Profile, counts: MeasuredCounts) -> Profile:
    """The calibration the counts bind to, or CountsError naming why they do not."""
    if counts.backend != profile.device.name:
        raise CountsError(
            f"these counts ran on {counts.backend}, but the profile describes"
            f" {profile.device.name}",
            hint="give the profile the counts were planned from",
        )
    planned, calibration = counts.profile.fingerprint, _calibration_fingerprint(profile)
    if planned != calibration:
        raise CountsError(
            f"these counts were planned from nv:{planned[:12]}; this profile's calibration is"
            f" nv:{calibration[:12]}",
            hint=f"run `nv list` to find nv:{planned[:12]}",
        )
    when = profile.device.calibrated_at
    if when is not None and counts.run_at < when:
        raise CountsError(
            f"these counts ran at {counts.run_at:%Y-%m-%d %H:%MZ}, before the calibration they"
            f" were planned from, taken at {when:%Y-%m-%d %H:%MZ}",
            hint="check run_at in the counts file",
        )
    base = profile.uncorrected()
    table = base.table
    report = Report.start(base, "reference", None)
    for c in counts.circuits:
        for q in c.qubits:
            if not 0 <= q < table.num_qubits:
                raise CountsError(
                    f"circuit {c.name} measures qubit {q}, but {profile.id} has qubits"
                    f" 0..{table.num_qubits - 1}"
                )
            if table.qubit(q).disabled:
                raise CountsError(
                    f"circuit {c.name} measures qubit {q}, which {profile.id} marks disabled"
                )
        for op in c.ops:
            if op.name == "delay":
                continue
            targets = [c.qubits[q] for q in op.qubits]
            try:
                resolve_op(
                    table,
                    charged_as(base, op.name, _unitary(op)),
                    targets,
                    unknown_gates="error",
                    report=report,
                )
            except (MissingCalibrationError, DisabledGateError, LayoutError) as exc:
                raise CountsError(
                    f"circuit {c.name}: {exc.message}",
                    hint="nv compare scores only ops the profile calibrates on their qubits",
                ) from None
    return base


def _floor_factors(base: Profile, circuits: Sequence[PlannedCircuit]) -> tuple[float, ...]:
    """Each measured gate's relaxation-floor factor, where its scaled error meets the error of
    its relaxation alone. A node there keeps the kink in the probabilities on a node."""
    table, floors = base.table, set()
    for c in circuits:
        for op in c.ops:
            if op.name == "delay":
                continue
            qubits = tuple(c.qubits[q] for q in op.qubits)
            found = table.gate(charged_as(base, op.name, _unitary(op)), qubits)
            if not isinstance(found, GateNoise) or found.state != "calibrated" or found.pauli:
                continue
            n = len(qubits)
            built = gate_channels(found, [table.qubit(q) for q in qubits])
            stated = metrics.depolarizing_from_avg(found.avg_infidelity, n)  # type: ignore[arg-type]
            relaxed = metrics.depolarizing_from_avg(built.relaxation, n)
            if 0 < relaxed < 1 and 0 < stated < 1:
                factor = math.log1p(-relaxed) / math.log1p(-stated)
                if FACTOR_RANGE[0] < factor < FACTOR_RANGE[1]:
                    floors.add(factor)
    return tuple(sorted(floors))


def _fisher(
    base: Profile,
    counts: MeasuredCounts,
    center: np.ndarray,
    gate: float | None,
    readout: float | None,
) -> np.ndarray:
    """The 2 x 2 expected information in (log gate, log readout) at the fitted factors, from
    central differences of exact probabilities. An absent axis has no information."""
    circuits = counts.circuits
    step = math.exp(FD_STEP)
    slopes = []
    for i, value in enumerate((gate, readout)):
        if value is None:
            slopes.append(np.zeros_like(center))
            continue
        up, down = [gate, readout], [gate, readout]
        up[i], down[i] = value * step, value / step
        slopes.append((_exact(base, circuits, *up) - _exact(base, circuits, *down)) / (2 * FD_STEP))
    jacobian = np.stack(slopes)
    weights = np.concatenate([np.full(2 ** len(c.qubits), c.shots) for c in circuits])
    keep = center > IMPOSSIBLE
    return (jacobian[:, keep] * (weights[keep] / center[keep])) @ jacobian[:, keep].T


def _identify(
    surface: _Surface, counts: np.ndarray, info: np.ndarray
) -> dict[Axis, NoEstimate | None]:
    """Which axes get no estimate, and why; None means fit the axis. The rule is per axis,
    so a locally flat axis keeps a one-sided interval and the other axis keeps its estimate."""
    shots = [counts[part].sum() for part in surface.slices]
    if not sum(shots):
        every = NoEstimate("the profile rules out every shot")
        return {"gate": every, "readout": every}
    out: dict[Axis, NoEstimate | None] = {}
    for axis in _AXES:
        if surface.axis(axis) is None:
            out[axis] = NoEstimate(_absent(surface, axis))
            continue
        responding = int(
            sum(n for n, moves in zip(shots, surface.moves(axis), strict=True) if moves)
        )
        if responding < MIN_INFORMATIVE_SHOTS:
            out[axis] = NoEstimate(f"only {_count(responding, 'shot')} respond to {axis} error")
        else:
            out[axis] = None
    if out["gate"] is None and out["readout"] is None:
        diagonal = np.diag(info)
        flat = diagonal <= SEPARABLE * diagonal.max()
        eigen = np.linalg.eigvalsh(info)
        if not flat.any() and eigen[0] < SEPARABLE * eigen[-1]:
            same = NoEstimate(_SAME_WAY)
            out = {"gate": same, "readout": same}
    return out


def _absent(surface: _Surface, axis: Axis) -> str:
    if axis == "gate":
        return "no circuit's outcomes move with gate error"
    if any(pair and sum(pair) for pairs in surface.pairs for pair in pairs):
        return "the readout error of the measured qubits does not scale"
    return "the measured qubits state no readout error"


def _ruled_out(
    surface: _Surface, circuits: Sequence[PlannedCircuit], impossible: np.ndarray
) -> tuple[str, ...]:
    """One phrase per stated zero readout error that explains impossible shots."""
    found: dict[tuple[int, int], dict[str, int]] = {}
    for c, part, pairs in zip(circuits, surface.slices, surface.pairs, strict=True):
        shots = impossible[part]
        if not shots.any():
            continue
        n = len(c.qubits)
        unread = surface.unread[:, part].reshape(-1, *(2,) * n)
        for i, pair in enumerate(pairs):
            if pair is None:
                continue
            marginal = unread.sum(axis=tuple(a for a in range(1, n + 1) if a != i + 1))
            for bit, stated in ((1, pair[0]), (0, pair[1])):
                if stated or marginal[:, bit].max() > IMPOSSIBLE:
                    continue
                cells = [k for k in np.flatnonzero(shots) if (k >> (n - 1 - i)) & 1 == bit]
                if cells:
                    per = found.setdefault((c.qubits[i], bit), {})
                    per[c.name] = per.get(c.name, 0) + int(shots[cells].sum())
    phrases = []
    for (qubit, bit), per in sorted(found.items()):
        stated = "P(1|0)" if bit else "P(0|1)"
        runs = _words([f"{_count(n, 'shot', c)}" for c, n in per.items()])
        phrases.append(f"qubit {qubit} has {stated} = 0, and {runs} read it as {bit}")
    return tuple(phrases)


def _notes(base: Profile, circuits: Sequence[PlannedCircuit]) -> tuple[str, ...]:
    """What the fit cannot scale or charge on the measured qubits."""
    table = base.table
    unscaled: dict[tuple[str, str], list[tuple[int, ...]]] = {}
    idle: set[int] = set()
    for c in circuits:
        for op in c.ops:
            qubits = tuple(c.qubits[q] for q in op.qubits)
            if op.name == "delay":
                if table.qubit(qubits[0]).relaxation_unknown:
                    idle.add(qubits[0])
                continue
            found = table.gate(charged_as(base, op.name, _unitary(op)), qubits)
            if not isinstance(found, GateNoise) or found.state != "calibrated":
                continue
            reason = _no_power(found.spec.metric, len(qubits))
            if reason and qubits not in unscaled.setdefault((found.gate, reason), []):
                unscaled[(found.gate, reason)].append(qubits)
    notes = [
        f"{name} on {_qubits(on)} is not scaled ({reason})"
        for (name, reason), on in unscaled.items()
    ]
    measured = sorted({q for c in circuits for q in c.qubits})
    chance = [(q,) for q in measured if sum(table.qubit(q).readout or (0, 0)) >= 1]
    if chance:
        notes.append(f"readout of {_qubits(chance)} is not scaled (no better than chance)")
    if idle:
        notes.append(
            f"delays on {_qubits([(q,) for q in sorted(idle)])} add no idle error"
            " (no T1 or T2 stated)"
        )
    return tuple(notes)


def _exact(
    base: Profile,
    circuits: Sequence[PlannedCircuit],
    gate: float | None,
    readout: float | None,
    *,
    read: bool = True,
) -> np.ndarray:
    """Reference probabilities of every circuit, with the factors applied to ``base``.

    One report started from ``base`` serves every circuit. A report started from the scaled
    profile would list what its factors leave unscaled, which reads every calibration record
    and costs more than the runs.
    """
    scaled = _with_factors(base, gate, readout)
    report = Report.start(base, "reference", None)
    return np.concatenate(
        [
            probabilities(
                scaled,
                c.ops,
                len(c.qubits),
                layout=c.qubits,
                readout=read,
                unknown_gates="error",
                report=report,
            )
            for c in circuits
        ]
    )


def _with_factors(base: Profile, gate: float | None, readout: float | None) -> Profile:
    """``base`` carrying these factors, without validating the whole profile again.

    ``base`` is valid and the factors are validated on their own. A readout factor comes only
    when the measured qubits state readout error, so the profile rule that refuses one
    without readout data cannot apply. Skipping the full validation halves each exact run.
    """
    factors = {
        name: {"factor": value}
        for name, value in (("gates", gate), ("readout", readout))
        if value is not None
    }
    if not factors:
        return base
    fields = {name: getattr(base, name) for name in type(base).model_fields}
    return type(base).model_construct(**{**fields, "unmodeled_error": UnmodeledError(**factors)})


def _loglik(probs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """sum n log p over cells, with 0 log 0 = 0 and n log 0 = -inf for n > 0.

    probs (points, cells) and counts (cells,) or (cells, draws) give (points,) or
    (points, draws). Observed and drawn counts go through this one kernel.
    """
    counts = np.asarray(counts, dtype=float)
    zero = probs <= 0
    with np.errstate(divide="ignore"):
        out = np.where(zero, 0.0, np.log(probs)) @ counts
    if zero.any():
        out[(zero.astype(float) @ (counts > 0)) > 0] = -np.inf
    return out


def _deviance(counts: np.ndarray, probs: np.ndarray, slices: Sequence[slice]) -> float:
    total = 0.0
    for part in slices:
        n, p = counts[part], probs[part]
        seen = n > 0
        with np.errstate(divide="ignore"):
            total += 2 * float(np.sum(n[seen] * np.log(n[seen] / (n.sum() * p[seen]))))
    return total


def _saturated(counts: np.ndarray, slices: Sequence[slice]) -> np.ndarray:
    """Each column's log-likelihood at its own frequencies."""
    total = np.zeros(counts.shape[1])
    for part in slices:
        n = counts[part]
        with np.errstate(divide="ignore", invalid="ignore"):
            total += np.where(n > 0, n * np.log(n / n.sum(axis=0)), 0.0).sum(axis=0)
    return total


def _nearest_one(values: np.ndarray, gate: np.ndarray, readout: np.ndarray) -> tuple[int, int]:
    """The best point; among ties, the one nearest factor 1 on both axes."""
    top = float(values.max())
    ties = values >= top - max(1e-9, 1e-12 * abs(top))
    distance = np.where(ties, np.abs(gate) + np.abs(readout), np.inf)
    i, j = np.unravel_index(int(np.argmin(distance)), values.shape)
    return int(i), int(j)


def _refine(line: np.ndarray, k: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Per column of ``line`` (points, columns), the (shift, gain) of the parabola through
    its best point ``k`` and that point's neighbours; 0 at an end of the line."""
    zero = np.zeros(line.shape[1])
    if len(line) < 3:
        return zero, zero
    columns = np.arange(line.shape[1])
    inner = np.clip(k, 1, len(line) - 2)
    left, mid, right = (line[inner + d, columns] for d in (-1, 0, 1))
    curve = 2 * mid - left - right
    usable = (k == inner) & (curve > 0) & np.isfinite(left) & np.isfinite(right)
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(usable, 0.5 * (right - left) / curve * step, 0.0)
        gain = np.where(usable, (right - left) ** 2 / (8 * curve), 0.0)
    return shift, gain


def _point(axis: Axis, held: float, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.full(len(others), held)
    return (fixed, others) if axis == "gate" else (others, fixed)


def _other(axis: Axis) -> Axis:
    return "readout" if axis == "gate" else "gate"


def _grid(axis: np.ndarray | None) -> np.ndarray:
    return np.zeros(1) if axis is None else axis


def _around(axis: np.ndarray | None, center: float, span: float) -> np.ndarray:
    if axis is None:
        return np.zeros(1)
    line = np.linspace(max(center - span, _LO), min(center + span, _HI), WINDOW_POINTS)
    return np.unique(np.append(line, center))


def _spacing(line: np.ndarray) -> float:
    return float(line[1] - line[0]) if len(line) > 1 else 0.0


def _read(block: np.ndarray, axis: int, matrices: np.ndarray) -> np.ndarray:
    """Apply one qubit's confusion matrix per point: M[measured, prepared] along ``axis``."""
    m = matrices.reshape(len(matrices), 2, 2, *(1,) * (block.ndim - 2))
    zero, one = np.take(block, 0, axis=axis), np.take(block, 1, axis=axis)
    return np.stack(
        [m[:, 0, 0] * zero + m[:, 0, 1] * one, m[:, 1, 0] * zero + m[:, 1, 1] * one], axis=axis
    )


@lru_cache(maxsize=1 << 16)
def _confusion(pair: tuple[float, float], log_factor: float) -> np.ndarray:
    a, b = metrics.scale_readout(pair, math.exp(log_factor))
    return np.array([[1 - a, b], [a, 1 - b]])


def _unitary(op: Op) -> np.ndarray:
    return gates.GATES[op.name].unitary(*op.params)  # type: ignore[misc]


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def _saved(estimate: Estimate) -> dict[str, Any]:
    def rounded(value: float | None) -> float | None:
        return None if value is None else float(f"{value:.6g}")

    return {
        "factor": rounded(estimate.factor),
        "low": rounded(estimate.low),
        "high": rounded(estimate.high),
        "bound": estimate.bound,
    }


def _factor_dict(result: FactorResult) -> dict[str, Any]:
    if isinstance(result, NoEstimate):
        return {"not_identified": result.reason}
    return {"factor": result.factor, "low": result.low, "high": result.high, "bound": result.bound}


def _beyond_noise(score: CircuitScore) -> bool:
    """Decided on the values as printed, so a flagged row always shows the larger TVD."""
    return round(score.tvd_fitted, 4) > round(score.shot_noise_95, 4)


def _say(result: FactorResult) -> str:
    return result.describe() if isinstance(result, Estimate) else "not identified"


def _reason(result: FactorResult) -> str | None:
    return result.reason if isinstance(result, NoEstimate) else None


def _on(qubits: Sequence[int]) -> str:
    joined = "-".join(map(str, qubits))
    return f"qubit {joined}" if len(qubits) == 1 else f"qubits {joined}"


def _count(n: int, noun: str, kind: str = "") -> str:
    kind = f"{kind} " if kind else ""
    return f"{n} {kind}{noun}" + ("" if n == 1 else "s")


def _words(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _duration(age: timedelta) -> str:
    minutes = age.total_seconds() / 60
    if minutes < 60:
        return f"{round(minutes)} min"
    if minutes < 48 * 60:
        return f"{round(minutes / 60)} h"
    return f"{round(minutes / 1440)} days"
