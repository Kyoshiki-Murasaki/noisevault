"""Stim export: a noisy copy of a Stim circuit that carries its conversion report.

Every Clifford gate is followed by the Pauli twirl of the channel the shared conversion rules
give it (PAULI_CHANNEL_1 or PAULI_CHANNEL_2). Twirling keeps each gate's average fidelity but
drops relaxation's pull toward |0>, so the export matches the twirled model, not the full
channel. Measurements get readout error, resets get preparation error, and with ``tick_ns``
every qubit left idle in a TICK layer gets twirled relaxation for that time. Annotations,
REPEAT blocks, detectors and observables pass through unchanged.
"""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np

try:
    import stim
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError("the Stim export needs stim: pip install 'noisevault[stim]'") from exc

from .. import gates
from ..channels import ChannelSpec, pauli_twirl, thermal_relaxation_kraus
from ..conversion import UnknownGates, resolve_op
from ..errors import DisabledGateError, LayoutError, MissingCalibrationError, NoiseVaultError
from ..layout import normalize_layout
from ..profile import Profile
from ..report import Report
from ..table import GateNoise, Unavailable

Readout = Literal["symmetrize", "exact", "none"]
ExistingNoise = Literal["error", "keep", "strip"]
Kind = Literal["tick", "annotation", "pad", "noise", "measure", "reset", "gate"]
Layout = Mapping[int, int] | Sequence[int] | None
EventCounts = tuple[tuple[str, str, int], ...]  # (event, key, count) that one gate adds

_ANNOTATIONS = frozenset({"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"})
_HERALDED = frozenset({"HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"})
_COMBINED = frozenset({"MPP", "SPP", "SPP_DAG"})  # a target group is joined by combiners
_REGISTRY = {name: row.name for row in gates.GATES.values() for name in row.stim}
# Stim gates that are an inverse or a fixed angle of a registry gate take that gate's noise.
_SAME_NOISE_AS = {
    "ISWAP_DAG": "iswap",
    "SQRT_ZZ_DAG": "zz",
    "SQRT_XX": "rxx",
    "SQRT_XX_DAG": "rxx",
    "SQRT_YY": "ryy",
    "SQRT_YY_DAG": "ryy",
}
_NOISELESS = frozenset({"II"})  # a 2-qubit identity is no entangling gate
_MULTI_ENTANGLER = frozenset({"CXSWAP", "SWAPCX", "CZSWAP"})  # two native entanglers each
# The Pauli that undoes each reset: a failed Z reset leaves |1>, a failed X reset |->.
_PREP_FLIP = {
    "R": "X_ERROR",
    "RX": "Z_ERROR",
    "RY": "X_ERROR",
    "MR": "X_ERROR",
    "MRX": "Z_ERROR",
    "MRY": "X_ERROR",
}
_PAULI_CHANNEL = {3: "PAULI_CHANNEL_1", 15: "PAULI_CHANNEL_2"}


class ExistingNoiseError(NoiseVaultError, ValueError):
    """The input circuit already has noise and the caller did not say what to do with it."""


class NoiseVaultStimCircuit(stim.Circuit):
    """A noisy ``stim.Circuit`` carrying ``.report``, ``.profile`` and ``.layout``.

    ``.readout_flips`` is set for ``readout="exact"``: one row per measurement record,
    (P(flip | recorded 0), P(flip | recorded 1)), which :func:`sample_with_readout` applies.
    Stim methods that build a new circuit (``copy``, ``flattened``, ``+``) return a plain
    ``stim.Circuit`` without these attributes. Pickling and ``copy.deepcopy`` keep them, with
    probabilities at the 6 significant digits of Stim's own circuit pickling.
    """

    report: Report
    profile: Profile
    layout: dict[int, int]
    readout_flips: np.ndarray | None

    def __getstate__(self) -> tuple[Any, dict[str, Any]]:
        return super().__getstate__(), dict(self.__dict__)

    def __setstate__(self, state: tuple[Any, dict[str, Any]]) -> None:
        program, attributes = state
        super().__setstate__(program)
        self.__dict__.update(attributes)

    def detector_error_model(
        self, *, approximate_disjoint_errors: bool | float = True, **options: Any
    ) -> stim.DetectorErrorModel:
        """Stim's detector error model, reading PAULI_CHANNEL components as independent.

        Stim refuses PAULI_CHANNEL_1/2 without ``approximate_disjoint_errors``, an O(p^2)
        approximation, so it is on by default here; pass False to get Stim's refusal.
        """
        return super().detector_error_model(
            approximate_disjoint_errors=approximate_disjoint_errors, **options
        )


def to_stim(
    profile: Profile,
    circuit: stim.Circuit | str,
    *,
    layout: Layout = None,
    readout: Readout = "symmetrize",
    tick_ns: float | None = None,
    existing_noise: ExistingNoise = "error",
    unknown_gates: UnknownGates = "typical",
) -> NoiseVaultStimCircuit:
    """A copy of ``circuit`` with the profile's noise, and a report of what it approximated.

    ``layout`` maps Stim qubit indices to physical qubits (identity by default;
    :func:`layout_from_coords` derives one from QUBIT_COORDS). ``readout="symmetrize"`` flips
    each result with the mean of P(1|0) and P(0|1); ``"exact"`` leaves measurements perfect and
    stores the asymmetric error for :func:`sample_with_readout`; ``"none"`` adds no readout
    error. ``tick_ns`` is the duration of one TICK layer: qubits idle in a layer relax for that
    long. ``existing_noise`` says what to do with noise already in the circuit: ``"error"``
    raises, ``"keep"`` keeps it, ``"strip"`` removes it.
    """
    _check_choice("readout", readout, Readout)
    _check_choice("existing_noise", existing_noise, ExistingNoise)
    _check_choice("unknown_gates", unknown_gates, UnknownGates)
    if tick_ns is not None and not (np.isfinite(tick_ns) and tick_ns > 0):
        raise ValueError(f"tick_ns={tick_ns!r}: give the TICK layer duration in ns (> 0) or None")
    circuit = _as_circuit(circuit)
    found = _scan(circuit)
    _check_existing_noise(found, existing_noise)
    if readout == "exact":
        _check_exact_readout(found)

    physical = normalize_layout(sorted(found.qubits), layout, profile)
    report = Report.start(
        profile,
        "stim",
        stim.__version__,
        layout=layout,
        readout=readout,
        tick_ns=tick_ns,
        existing_noise=existing_noise,
        unknown_gates=unknown_gates,
    )
    report.record_effects(profile.effects)
    _describe(report, found, readout, tick_ns, existing_noise)
    exporter = _Exporter(profile, physical, report, readout, tick_ns, existing_noise, unknown_gates)
    lines, _ = exporter.block(circuit, set())
    exporter.finish()

    out = NoiseVaultStimCircuit("\n".join(lines))
    out.report, out.profile, out.layout = report, profile, physical
    out.readout_flips = exporter.record_flips(circuit) if readout == "exact" else None
    return out


def sample_with_readout(
    circuit: NoiseVaultStimCircuit, shots: int, *, seed: int | None = None
) -> np.ndarray:
    """Measurement samples, shape (shots, num_measurements), with exact asymmetric readout.

    The circuit must come from ``to_stim(..., readout="exact")``: Stim samples perfect
    measurements and each recorded bit then flips with the probability for its value.
    """
    flips = getattr(circuit, "readout_flips", None)
    if flips is None:
        raise ValueError(
            "sample_with_readout needs a circuit exported with readout='exact':"
            " profile.to_stim(circuit, readout='exact')"
        )
    rng = np.random.default_rng(seed)
    bits = circuit.compile_sampler(seed=int(rng.integers(2**63))).sample(shots)
    flip = rng.random(bits.shape) < np.where(bits, flips[:, 1], flips[:, 0])
    return bits ^ flip


def layout_from_coords(circuit: stim.Circuit | str, profile: Profile) -> dict[int, int]:
    """Place the circuit on the device by matching its QUBIT_COORDS to the profile's coords.

    Tries the eight rotations and reflections of the square lattice, also after a 45 degree
    turn (Stim's rotated surface codes put neighbors on diagonals), and every translation. A
    placement must land every qubit on an enabled device qubit and every 2-qubit gate on a pair
    with a calibrated native gate. Among valid placements it returns the one with the lowest
    summed 2-qubit gate error and mean readout error.
    """
    circuit = _as_circuit(circuit)
    found = _scan(circuit)
    coords = circuit.get_final_qubit_coordinates()
    missing = sorted(q for q in found.qubits if len(coords.get(q, ())) < 2)
    if missing:
        raise LayoutError(
            f"qubits {missing} have no 2D QUBIT_COORDS in the circuit; add them or pass layout="
        )
    device = _Device(profile)
    labels = sorted(found.qubits)
    if not labels:
        return {}
    points = np.array([coords[q][:2] for q in labels], dtype=float)
    column = {q: i for i, q in enumerate(labels)}
    pairs = np.array([(column[a], column[b]) for a, b in sorted(found.pairs)], dtype=int)
    cost = _PlacementCost(profile, pairs.reshape(-1, 2))
    best: tuple[float, np.ndarray] | None = None
    for transform in _TRANSFORMS:
        placements = device.placements(points @ transform.T)
        if not len(placements):
            continue
        totals = cost(placements)
        i = int(np.argmin(totals))
        if np.isfinite(totals[i]) and (best is None or totals[i] < best[0]):
            best = (float(totals[i]), placements[i])
    if best is None:
        raise LayoutError(
            f"no rotation or shift of the circuit's QUBIT_COORDS fits {profile.id}'s qubit coords"
            " with every 2-qubit gate on a connected pair; pass layout= explicitly"
        )
    return dict(zip(labels, map(int, best[1]), strict=True))


# circuit scan -----------------------------------------------------------------------------


@dataclass
class _Scan:
    qubits: set[int] = field(default_factory=set)  # every qubit an operation touches
    pairs: set[tuple[int, int]] = field(default_factory=set)  # 2-qubit gate targets
    noise: str | None = None  # first noise instruction, as text
    herald: str | None = None  # first heralded noise instruction (it adds records)
    feedback: str | None = None  # first gate controlled by a measurement record
    product: str | None = None  # first multi-qubit measurement


def _scan(circuit: stim.Circuit, found: _Scan | None = None) -> _Scan:
    found = found or _Scan()
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            _scan(item.body_copy(), found)
            continue
        kind = _kind(item.name)
        if kind == "pad" and item.gate_args_copy():
            found.noise = found.noise or _short(item)
        if kind in ("tick", "annotation", "pad"):
            continue
        if kind == "noise" or (kind == "measure" and item.gate_args_copy()):
            found.noise = found.noise or _short(item)
        if item.name in _HERALDED:
            found.herald = found.herald or _short(item)
        if kind == "noise":
            continue
        for group in item.target_groups():
            qubits = [t.qubit_value for t in group if t.qubit_value is not None]
            found.qubits.update(qubits)
            if any(t.is_measurement_record_target for t in group):
                found.feedback = found.feedback or _short(item)
            elif kind == "gate" and len(qubits) == 2:
                found.pairs.add((qubits[0], qubits[1]))
            if kind == "measure" and len(qubits) > 1:
                found.product = found.product or _short(item)
    return found


def _check_existing_noise(found: _Scan, policy: ExistingNoise) -> None:
    if found.noise is not None and policy == "error":
        raise ExistingNoiseError(
            f"the circuit already has noise ({found.noise}); pass existing_noise='strip' to"
            " replace it with the profile's noise, or existing_noise='keep' to add to it"
        )
    if found.herald is not None and policy == "strip":
        raise ExistingNoiseError(
            f"{found.herald} adds measurement records that later rec[] targets count, so it"
            " cannot be stripped; remove it from the circuit or pass existing_noise='keep'"
        )


def _check_exact_readout(found: _Scan) -> None:
    problem = None
    if found.feedback:
        problem = f"{found.feedback} feeds a measurement back into the circuit"
    if found.product:
        problem = f"{found.product} measures a multi-qubit product"
    if problem:
        raise ValueError(
            f"readout='exact' flips recorded bits after sampling, which is exact only for"
            f" single-qubit measurements that nothing reads during the circuit, but {problem};"
            " use readout='symmetrize'"
        )


# emission ---------------------------------------------------------------------------------


class _Exporter:
    """Emits Stim program text; text parsing is far faster than stim.Circuit.append."""

    def __init__(
        self,
        profile: Profile,
        physical: dict[int, int],
        report: Report,
        readout: Readout,
        tick_ns: float | None,
        existing_noise: ExistingNoise,
        unknown_gates: UnknownGates,
    ) -> None:
        self.table = profile.table
        self.physical = physical
        self.qubits = frozenset(physical)
        self.report = report
        self.readout = readout
        self.tick_ns = tick_ns
        self.keep_noise = existing_noise == "keep"
        self.unknown_gates = unknown_gates
        self.unknown: dict[str, set[int]] = {"readout": set(), "prep": set(), "idle": set()}
        self._gate_noise: dict[tuple[str, tuple[int, ...]], tuple[str, EventCounts]] = {}
        # Gate applications per (Stim gate, qubits); a REPEAT body counts once per pass.
        self._applied: Counter[tuple[str, tuple[int, ...]]] = Counter()
        self._passes = 1
        self._idle: dict[int, str] = {}
        self._twirls: dict[tuple, str] = {}
        self._handlers = {
            "tick": self._tick,
            "annotation": self._copy,
            "pad": self._pad,
            "noise": self._noise,
            "measure": self._measure,
            "reset": self._reset,
            "gate": self._gate,
        }

    def block(self, circuit: stim.Circuit, busy: set[int]) -> tuple[list[str], set[int]]:
        """Lines for ``circuit`` given the qubits already busy in the current TICK layer."""
        lines: list[str] = []
        for item in circuit:
            if isinstance(item, stim.CircuitRepeatBlock):
                busy = self._repeat(item, busy, lines)
            else:
                busy = self._handlers[_kind(item.name)](item, busy, lines)
        return lines, busy

    def _repeat(self, block: stim.CircuitRepeatBlock, busy: set[int], lines: list[str]) -> set[int]:
        body, count, outer = block.body_copy(), block.repeat_count, self._passes
        peel = self.tick_ns is not None and count > 1
        self._passes = outer * (1 if peel else count)
        first, after = self.block(body, set(busy))
        again = first
        if peel:
            # Idle noise at the body's first TICK depends on what ran before it, which differs
            # between the first pass and later ones; then the first pass is written out once.
            # The two walks count 1 and count - 1 passes, so events total count either way.
            self._passes = outer * (count - 1)
            again, after = self.block(body, set(after))
        self._passes = outer
        tag = f"[{block.tag}]" if block.tag else ""
        if again != first:
            lines.extend(first)
            count -= 1
        lines.extend([f"REPEAT{tag} {count} {{", *again, "}"])
        return after

    def _copy(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        lines.append(str(inst))
        return busy

    def _pad(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        keep = self.keep_noise or not inst.gate_args_copy()
        lines.append(str(inst) if keep else _text(inst, inst.target_groups(), []))
        return busy

    def _noise(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        if self.keep_noise:
            lines.append(str(inst))
        return busy

    def _tick(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        if self.tick_ns is not None:
            noise: dict[str, list[int]] = {}
            for q in sorted(self.qubits - busy):
                channel = self._idle_noise(q)
                if channel:
                    noise.setdefault(channel, []).append(q)
            lines.extend(_noise_lines(noise))
        lines.append(str(inst))
        return set()

    def _gate(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        chunks = _disjoint_chunks(inst.target_groups())
        for chunk in chunks:
            lines.append(str(inst) if len(chunks) == 1 else _text(inst, chunk))
            noise: dict[str, list[int]] = {}
            for group in chunk:
                if any(t.is_measurement_record_target or t.is_sweep_bit_target for t in group):
                    self.report.approximate(
                        "classically controlled Paulis", "no gate noise", "Pauli-frame updates"
                    )
                    continue
                qubits = tuple(t.value for t in group)
                busy.update(qubits)
                channel = self._gate_channel(inst.name, qubits)
                if channel:
                    noise.setdefault(channel, []).extend(qubits)
            lines.extend(_noise_lines(noise))
        return busy

    def _measure(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        stated = inst.gate_args_copy()
        stated_flip = stated[0] if stated and self.keep_noise else 0.0
        runs: list[tuple[float, list[list[stim.GateTarget]]]] = []
        measured: list[int] = []
        for group in inst.target_groups():
            qubits = [t.qubit_value for t in group]
            measured += qubits
            flip = _either(stated_flip, self._readout_flip(qubits))
            if runs and runs[-1][0] == flip:
                runs[-1][1].append(group)
            else:
                runs.append((flip, [group]))
        busy.update(measured)
        # Runs of equal flip probability keep the targets, and so the record order, unchanged.
        lines.extend(_text(inst, groups, [flip] if flip else []) for flip, groups in runs)
        if inst.name in _PREP_FLIP:
            self._prep(inst.name, measured, lines)
        return busy

    def _reset(self, inst: stim.CircuitInstruction, busy: set[int], lines: list[str]) -> set[int]:
        qubits = [t.value for t in inst.targets_copy()]
        busy.update(qubits)
        lines.append(str(inst))
        self._prep(inst.name, qubits, lines)
        return busy

    def _prep(self, name: str, qubits: list[int], lines: list[str]) -> None:
        noise: dict[str, list[int]] = {}
        for q in qubits:
            error = self.table.qubit(self.physical[q]).prep_error
            if error is None:
                self.unknown["prep"].add(self.physical[q])
            elif error > 0:
                noise.setdefault(f"{_PREP_FLIP[name]}({error!r})", []).append(q)
        lines.extend(_noise_lines(noise))

    # noise values -----------------------------------------------------------------------

    def _gate_channel(self, stim_name: str, qubits: tuple[int, ...]) -> str:
        key = (stim_name, qubits)
        if key not in self._gate_noise:
            self._gate_noise[key] = self._resolve(stim_name, qubits)
        channel, events = self._gate_noise[key]
        if events:
            self._applied[key] += self._passes
        return channel

    def _resolve(self, stim_name: str, qubits: tuple[int, ...]) -> tuple[str, EventCounts]:
        """Twirled channel of one gate and the report events that resolving it counted."""
        if stim_name in _NOISELESS:
            return "", ()
        wires = tuple(self.physical[q] for q in qubits)
        before = {event: Counter(counts) for event, counts in self.report.events.items()}
        try:
            if stim_name in _MULTI_ENTANGLER:
                raise MissingCalibrationError(
                    f"{stim_name} needs two native entangling gates, so no single calibration"
                    " describes it; decompose it into the profile's native gates first"
                )
            name = _canonical(stim_name)
            built = resolve_op(
                self.table, name, wires, unknown_gates=self.unknown_gates, report=self.report
            )
        except (MissingCalibrationError, DisabledGateError, LayoutError) as exc:
            raise type(exc)(self._explain(stim_name, qubits, wires, exc)) from exc
        return self._twirl(built.channels, wires), _added_events(before, self.report.events)

    def _explain(
        self, stim_name: str, qubits: tuple[int, ...], wires: tuple[int, ...], exc: Exception
    ) -> str:
        where = f"`{stim_name} {' '.join(map(str, qubits))}` (physical qubits {list(wires)}): {exc}"
        if len(wires) == 2 and isinstance(self.table.typical(2, wires), Unavailable):
            where += (
                "; pass layout= so 2-qubit gates land on connected pairs"
                " (profile.suggest_layout(n) proposes one, noisevault.stim.layout_from_coords"
                " matches the circuit's QUBIT_COORDS)"
            )
        return where

    def _twirl(self, channels: Sequence[ChannelSpec], wires: tuple[int, ...]) -> str:
        # Devices with shared defaults repeat the same channel on every pair: twirl it once.
        key = tuple(
            (c.kind, tuple(wires.index(w) for w in c.wires), b"".join(k.tobytes() for k in c.kraus))
            for c in channels
        )
        if key not in self._twirls:
            self._twirls[key] = _pauli_channel(pauli_twirl(channels, wires) if channels else ())
        return self._twirls[key]

    def _idle_noise(self, q: int) -> str:
        if q not in self._idle:
            noise = self.table.qubit(self.physical[q])
            if noise.t1_ns is None and noise.t2_ns is None and not noise.dephasing_rate_per_s:
                self.unknown["idle"].add(noise.index)
            kraus = thermal_relaxation_kraus(
                noise.t1_ns, noise.t2_ns, float(self.tick_ns or 0.0), noise.dephasing_rate_per_s
            )
            channel = ChannelSpec("thermal_relaxation", (noise.index,), tuple(kraus))
            self._idle[q] = self._twirl([channel], (noise.index,))
        return self._idle[q]

    def _readout_flip(self, qubits: Iterable[int]) -> float:
        """Symmetric flip of the recorded parity of ``qubits``, each read out independently."""
        if self.readout != "symmetrize":
            return 0.0
        parity = 1.0
        for q in qubits:
            a, b = self._readout(q)
            parity *= 1.0 - (a + b)
        return (1.0 - parity) / 2.0

    def _readout(self, q: int) -> tuple[float, float]:
        pair = self.table.qubit(self.physical[q]).readout
        if pair is None:
            self.unknown["readout"].add(self.physical[q])
            return 0.0, 0.0
        return pair

    def record_flips(self, circuit: stim.Circuit) -> np.ndarray:
        """(P(flip | recorded 0), P(flip | recorded 1)) for every measurement record, in order."""
        rows: list[tuple[float, float]] = []
        for inst in circuit.flattened():
            kind = _kind(inst.name)
            if inst.name in _HERALDED or inst.name == "MPAD":
                rows += [(0.0, 0.0)] * len(inst.targets_copy())
            elif kind == "measure":
                for t in inst.targets_copy():
                    a, b = self._readout(t.value)  # a = P(1|0), b = P(0|1)
                    rows.append((b, a) if t.is_inverted_result_target else (a, b))
        return np.array(rows, dtype=float).reshape(-1, 2)

    def finish(self) -> None:
        """Count every gate application's events and report the unknown values."""
        for key, applied in self._applied.items():
            for event, what, n in self._gate_noise[key][1]:
                self.report.count(event, what, n * (applied - 1))  # resolving counted one
        texts = {
            "readout": "readout error",
            "prep": "reset error",
            "idle": "T1/T2 for idle noise",
        }
        for key, qubits in self.unknown.items():
            if qubits:
                self.report.mark_unknown(f"{texts[key]} of physical qubits {_qubit_list(qubits)}")


def _describe(
    report: Report,
    found: _Scan,
    readout: Readout,
    tick_ns: float | None,
    existing_noise: ExistingNoise,
) -> None:
    report.approximate(
        "gate noise",
        "Pauli twirl of each gate's channel",
        "keeps average gate fidelity, drops relaxation's bias toward |0>",
    )
    report.approximate(
        "detector_error_model()",
        "Pauli channel components treated as independent",
        "Stim's approximate_disjoint_errors, an O(p^2) change",
    )
    if readout == "symmetrize":
        report.approximate(
            "readout error",
            "symmetric flip with the mean of P(1|0) and P(0|1)",
            "Stim flips results symmetrically; readout='exact' keeps the asymmetry",
        )
    elif readout == "exact":
        report.mark_exact("readout error, applied by sample_with_readout")
        report.approximate(
            "readout error inside the circuit",
            "none",
            "Stim's own samplers and detector_error_model() see perfect readout",
        )
    else:
        report.omit("readout error (readout='none')")
    report.omit("initial state preparation error (qubits start in |0> unless reset)")
    if tick_ns is None:
        report.omit("idle noise (pass tick_ns= to add relaxation at each TICK)")
    else:
        report.approximate(
            "idle noise",
            f"twirled relaxation for {tick_ns} ns on each qubit idle in a TICK layer",
            "qubits busy in a layer get only their gate or readout noise; the layer after the"
            " last TICK gets none",
        )
    if found.noise is not None:
        if existing_noise == "keep":
            report.approximate("noise already in the circuit", "kept, on top of the profile's")
        else:
            report.omit("noise already in the circuit (stripped)")


# helpers ----------------------------------------------------------------------------------


@functools.cache
def _kind(name: str) -> Kind:
    if name == "TICK":
        return "tick"
    if name in _ANNOTATIONS:
        return "annotation"
    if name == "MPAD":
        return "pad"
    data = stim.gate_data(name)
    if data.is_noisy_gate and (name in _HERALDED or not data.produces_measurements):
        return "noise"
    if data.produces_measurements:
        return "measure"
    if data.is_reset:
        return "reset"
    if data.is_unitary:
        return "gate"
    raise ValueError(f"the Stim export does not handle {name} instructions")


def _canonical(stim_name: str) -> str:
    """Registry name; unknown gates keep their Stim name and get the typical native's noise."""
    return _REGISTRY.get(stim_name) or _SAME_NOISE_AS.get(stim_name) or stim_name.lower()


def _pauli_channel(probs: Sequence[float]) -> str:
    """PAULI_CHANNEL_1/2 prefix; metrics' label order is Stim's, first letter on target 0."""
    if not any(probs):
        return ""
    return f"{_PAULI_CHANNEL[len(probs)]}({','.join(map(repr, probs))})"


def _noise_lines(noise: dict[str, list[int]]) -> list[str]:
    return [f"{prefix} {' '.join(map(str, qubits))}" for prefix, qubits in noise.items()]


def _disjoint_chunks(groups: list[list[stim.GateTarget]]) -> list[list[list[stim.GateTarget]]]:
    """Split target groups so no qubit repeats in a chunk, keeping each gate's noise in order."""
    chunks: list[list[list[stim.GateTarget]]] = [[]]
    used: set[int] = set()
    for group in groups:
        qubits = {t.qubit_value for t in group if t.qubit_value is not None}
        if used & qubits:
            chunks.append([])
            used = set()
        chunks[-1].append(group)
        used |= qubits
    return chunks


def _text(
    inst: stim.CircuitInstruction,
    groups: list[list[stim.GateTarget]],
    args: list[float] | None = None,
) -> str:
    targets: list[stim.GateTarget] = []
    for group in groups:
        for i, target in enumerate(group):
            if i and inst.name in _COMBINED:
                targets.append(stim.target_combiner())
            targets.append(target)
    args = inst.gate_args_copy() if args is None else args
    return str(stim.CircuitInstruction(inst.name, targets, args, tag=inst.tag))


def _added_events(before: dict[str, Counter[str]], after: dict[str, Counter[str]]) -> EventCounts:
    return tuple(
        (event, what, n)
        for event, counts in after.items()
        for what, n in (counts - before.get(event, Counter())).items()
    )


def _either(p: float, q: float) -> float:
    """Probability that exactly one of two independent flips happens."""
    return p + q - 2.0 * p * q


def _short(inst: stim.CircuitInstruction) -> str:
    text = str(inst)
    return text if len(text) <= 60 else text[:57] + "..."


def _qubit_list(qubits: set[int]) -> str:
    ordered = sorted(qubits)
    shown = ", ".join(map(str, ordered[:10]))
    return shown + (f", ... ({len(ordered)} qubits)" if len(ordered) > 10 else "")


def _as_circuit(circuit: Any) -> stim.Circuit:
    if isinstance(circuit, str):
        return stim.Circuit(circuit)
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"expected a stim.Circuit or Stim program text, got {type(circuit)!r}")
    return circuit


def _check_choice(name: str, value: Any, choices: Any) -> None:
    options = get_args(choices)
    if value not in options:
        raise ValueError(f"{name}={value!r}: choose one of {', '.join(map(repr, options))}")


_D8 = [np.array(m, dtype=float) for m in (
    [[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
    [[1, 0], [0, -1]], [[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]],
)]  # fmt: skip
_TURN_45 = 0.5 * np.array([[1.0, 1.0], [1.0, -1.0]])  # diagonal neighbors become grid neighbors
_TRANSFORMS = _D8 + [d @ _TURN_45 for d in _D8]


class _Device:
    """Enabled device qubits with coords, looked up by position (to 1e-6) in bulk."""

    def __init__(self, profile: Profile) -> None:
        usable = [
            (q.coords[:2], q.index)
            for q in profile.qubits
            if q.coords is not None and len(q.coords) >= 2 and not q.disabled
        ]
        if not usable:
            raise LayoutError(
                f"{profile.id} records no qubit coords; pass layout={{stim qubit: physical qubit}}"
            )
        self.index = np.array([i for _, i in usable], dtype=int)
        self.grid = _grid_units(np.array([c for c, _ in usable], dtype=float))
        self.low, self.high = self.grid.min(axis=0), self.grid.max(axis=0)
        keys = self._keys(self.grid)
        self.order = np.argsort(keys, kind="stable")
        self.sorted_keys = keys[self.order]

    def placements(self, points: np.ndarray) -> np.ndarray:
        """Physical qubits of every shift that lands all points on device qubits.

        Shape (shifts, points); a shift puts points[0] on a device qubit, in device order.
        """
        offsets = _grid_units(points - points[0])
        # Shifts that would push the circuit's bounding box off the device cannot fit.
        fits = (self.grid + offsets.min(axis=0) >= self.low).all(axis=1)
        fits &= (self.grid + offsets.max(axis=0) <= self.high).all(axis=1)
        keys = self._keys(self.grid[fits][:, None, :] + offsets[None, :, :])
        at = np.searchsorted(self.sorted_keys, keys).clip(max=len(self.sorted_keys) - 1)
        whole = (self.sorted_keys[at] == keys).all(axis=1)
        return self.index[self.order[at[whole]]]

    def _keys(self, grid: np.ndarray) -> np.ndarray:
        """One integer per point inside the device's bounding box."""
        inside = grid - self.low
        return inside[..., 0] * (self.high[1] - self.low[1] + 1) + inside[..., 1]


def _grid_units(points: np.ndarray) -> np.ndarray:
    return np.rint(points * 1e6).astype(np.int64)


class _PlacementCost:
    """Summed 2-qubit error of the circuit's pairs plus mean readout error; inf if unusable."""

    def __init__(self, profile: Profile, pairs: np.ndarray) -> None:
        self.table = profile.table
        self.pairs = pairs  # (pairs, 2) columns of a placement
        self.readout = np.array(
            [
                sum(r) / 2 if (r := self.table.qubit(i).readout) else 0.0
                for i in range(self.table.num_qubits)
            ],
            dtype=float,
        )
        self._pair: dict[int, float] = {}
        # A pair can host a 2-qubit gate only through connectivity or a calibration record, in
        # either direction; other pairs skip the (slow) typical-gate lookup.
        self.maybe: np.ndarray | None = None
        if not self.table.all_to_all:
            n = self.table.num_qubits
            linked = self.table.edges()
            linked += [tuple(r.qubits) for r in profile.calibrations if len(r.qubits) == 2]
            codes = [a * n + b for a, b in linked] + [b * n + a for a, b in linked]
            self.maybe = np.array(codes, dtype=np.int64)

    def __call__(self, placements: np.ndarray) -> np.ndarray:
        total = self.readout[placements].sum(axis=1)
        if len(self.pairs):
            n = self.table.num_qubits
            codes = placements[:, self.pairs[:, 0]] * n + placements[:, self.pairs[:, 1]]
            unique, inverse = np.unique(codes, return_inverse=True)
            errors = np.full(len(unique), np.nan)
            maybe = (
                np.ones(len(unique), bool) if self.maybe is None else np.isin(unique, self.maybe)
            )
            errors[maybe] = [self._pair_error(int(c), n) for c in unique[maybe]]
            total = total + errors[inverse.reshape(codes.shape)].sum(axis=1)
        return np.where(np.isnan(total), np.inf, total)

    def _pair_error(self, code: int, n: int) -> float:
        if code not in self._pair:
            found = self.table.typical(2, divmod(code, n))
            ok = isinstance(found, GateNoise)
            self._pair[code] = (found.avg_infidelity or 0.0) if ok else np.nan  # type: ignore[union-attr]
        return self._pair[code]
