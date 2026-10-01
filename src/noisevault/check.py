"""Conformance check: every framework export against the NoiseVault reference on small circuits.

``check(profile)`` builds a few circuits from the profile's own native gates on a well
calibrated chain of qubits, computes their outcome probabilities (readout included) with the
framework-free reference simulator, and runs the same circuits through each installed export
with that framework's own simulator and readout mechanism. Qiskit, Cirq and PennyLane are
compared exactly; Stim is sampled and compared against the Pauli-twirled reference, which is
the model it implements.

A pass certifies that the exports implement the same noise model as the reference on these
circuits. It says nothing about how closely that model matches the hardware.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, Any

import numpy as np

from . import gates
from .channels import pauli_kraus, pauli_twirl, readout_matrix
from .conversion import resolve_op
from .errors import LayoutError, NoiseApproximationWarning, NoiseVaultError, install_hint
from .layout import normalize_layout
from .reference import Op, _apply
from .reference import probabilities as reference_probabilities
from .report import Report
from .table import GateNoise

if TYPE_CHECKING:
    from .profile import Profile

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "stim")
EXACT_TOLERANCE = 1e-9
SIGMAS = 5.0
MAX_QUBITS = 4
NOTE = (
    "A pass means each export implements the same noise model as the NoiseVault reference on"
    " these circuits. It is not a measure of how well the model matches the hardware."
)
_EXTRAS = {"qiskit": "qiskit", "cirq": "cirq", "pennylane": "pennylane", "stim": "stim"}
# Gate angles for check circuits: pi/2 unless listed. r keeps a phase off 0 so Cirq does not
# turn it into an X rotation. rzz at pi/2 equals zz, which Cirq and PennyLane must still
# charge rzz's noise unless the profile defines zz (see _two_qubit_ops).
_ANGLES: dict[str, tuple[float, ...]] = {"r": (pi / 2, pi / 4), "ms": (0.0, 0.0)}
_MAX_ORDER = 8


@dataclass(frozen=True)
class Circuit:
    """A check circuit on circuit qubits ``0..num_qubits-1`` (layout qubits in order)."""

    name: str
    num_qubits: int
    ops: tuple[Op, ...]


@dataclass(frozen=True)
class CircuitCheck:
    circuit: str
    num_qubits: int
    gates: tuple[str, ...]  # the gates this framework's version of the circuit uses
    tvd: float
    tolerance: float
    passed: bool
    sampled: bool = False  # True: shots through the framework's own measurement, 5 sigma


@dataclass(frozen=True)
class FrameworkCheck:
    """One framework's export run on the check circuits it can express."""

    framework: str
    version: str
    method: str
    circuits: tuple[CircuitCheck, ...]
    not_run: tuple[tuple[str, str], ...]  # (circuit, why this framework cannot express it)
    approximated: int
    omitted: int
    unknown: int
    report: str  # the export report's summary

    @property
    def passed(self) -> bool:
        return bool(self.circuits) and all(c.passed for c in self.circuits)

    @property
    def worst(self) -> CircuitCheck:
        """The circuit closest to (or furthest past) its tolerance."""
        return max(self.circuits, key=lambda c: c.tvd / c.tolerance)

    @property
    def max_tvd(self) -> float:
        return max(c.tvd for c in self.circuits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "version": self.version,
            "method": self.method,
            "passed": self.passed,
            "max_tvd": self.max_tvd,
            "tolerance": self.worst.tolerance,
            "circuits": [{**c.__dict__, "gates": list(c.gates)} for c in self.circuits],
            "not_run": [{"circuit": c, "reason": r} for c, r in self.not_run],
            "report": {
                "approximated": self.approximated,
                "omitted": self.omitted,
                "unknown": self.unknown,
                "summary": self.report,
            },
        }


@dataclass(frozen=True)
class CheckResult:
    profile_id: str
    fingerprint: str
    layout: dict[int, int]
    shots: int
    seed: int | None
    circuits: tuple[Circuit, ...]
    frameworks: tuple[FrameworkCheck, ...]
    skipped: tuple[tuple[str, str], ...] = ()  # (framework, reason)

    @property
    def passed(self) -> bool:
        """At least one framework ran, and every one that ran passed."""
        return bool(self.frameworks) and all(f.passed for f in self.frameworks)

    def summary(self) -> str:
        chain = "-".join(str(self.layout[i]) for i in range(len(self.layout)))
        lines = [
            f"check {self.profile_id} nv:{self.fingerprint[:12]}: {len(self.circuits)} circuits"
            f" on qubits {chain}"
        ]
        for f in self.frameworks:
            verdict = "pass" if f.passed else "FAIL"
            lines.append(
                f"  {f.framework:<10} {verdict}  max TVD {f.max_tvd:.2g} (tolerance"
                f" {f.worst.tolerance:.2g}), {len(f.circuits)} circuits, {f.method}"
            )
            lines += [f"    not run: {c}: {why}" for c, why in f.not_run]
        lines += [f"  {name:<10} skipped: {why}" for name, why in self.skipped]
        lines.append(NOTE)
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "fingerprint": self.fingerprint,
            "passed": self.passed,
            "layout": {str(k): v for k, v in self.layout.items()},
            "shots": self.shots,
            "seed": self.seed,
            "circuits": [
                {
                    "name": c.name,
                    "num_qubits": c.num_qubits,
                    "ops": [[op.name, list(op.qubits), list(op.params)] for op in c.ops],
                }
                for c in self.circuits
            ],
            "frameworks": [f.to_dict() for f in self.frameworks],
            "skipped": [{"framework": n, "reason": r} for n, r in self.skipped],
            "note": NOTE,
        }


def check(
    profile: Profile,
    *,
    frameworks: Sequence[str] | None = None,
    layout: Mapping[Hashable, int] | Sequence[int] | None = None,
    shots: int = 20_000,
    seed: int | None = 0,
) -> CheckResult:
    """Run the check circuits through every installed export (or ``frameworks``).

    ``layout`` gives the chain of physical qubits to use (1 to 4 qubits, neighbors connected);
    by default ``profile.suggest_layout(4)``. ``shots`` and ``seed`` apply to sampled
    frameworks (Stim). A framework that is not installed, or cannot express any check circuit,
    is listed in ``skipped`` with the reason.
    """
    names = list(FRAMEWORKS if frameworks is None else frameworks)
    unknown = [n for n in names if n not in FRAMEWORKS]
    if unknown:
        raise ValueError(f"unknown framework {unknown[0]!r}; choose from {', '.join(FRAMEWORKS)}")
    if not isinstance(shots, int) or shots < 1:
        raise ValueError(f"shots={shots!r}: give a positive number of shots")
    chain = _chain(profile, layout)
    circuits = build_circuits(profile, chain)
    if not circuits:
        raise NoiseVaultError(
            f"{profile.id} has no calibrated native gate with a known unitary on qubits {chain},"
            " so there is nothing to check; pass layout= with other qubits"
        )
    expected = _Expected(profile, chain)
    results, skipped = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)  # each report records it
        for name in names:
            try:
                runner = _RUNNERS[name](profile, chain)
            except ImportError:
                skipped.append((name, f"not installed: {install_hint(_EXTRAS[name])}"))
                continue
            except NoiseVaultError as exc:
                skipped.append((name, f"the export refused this profile: {exc}"))
                continue
            outcome = _run(runner, circuits, expected, shots, seed)
            if isinstance(outcome, str):
                skipped.append((name, outcome))
            else:
                results.append(outcome)
    return CheckResult(
        profile_id=profile.id,
        fingerprint=profile.fingerprint,
        layout=dict(enumerate(chain)),
        shots=shots,
        seed=seed,
        circuits=circuits,
        frameworks=tuple(results),
        skipped=tuple(skipped),
    )


# circuits -----------------------------------------------------------------------------------


def _chain(profile: Profile, layout: Mapping[Hashable, int] | Sequence[int] | None) -> list[int]:
    if layout is None:
        n = min(MAX_QUBITS, profile.device.num_qubits)
        chain = list(profile.suggest_layout(n).values())
    else:
        n = len(layout)
        if not 1 <= n <= MAX_QUBITS:
            raise LayoutError(f"a check layout has 1 to {MAX_QUBITS} qubits, got {n}")
        mapping = normalize_layout(range(n), layout, profile)
        chain = [mapping[i] for i in range(n)]
    if _unitary_natives(profile, 2):
        for i in range(n - 1):
            if not _two_qubit_ops(profile, chain, i):
                raise LayoutError(
                    f"qubits {chain[i]} and {chain[i + 1]} share no calibrated 2-qubit native"
                    " gate; pass a layout whose neighbors are connected"
                    " (profile.suggest_layout(n) gives one)"
                )
    return chain


def build_circuits(
    profile: Profile, chain: Sequence[int], expressible: Callable[[Op], bool] = lambda op: True
) -> tuple[Circuit, ...]:
    """Check circuits from the calibrated natives on ``chain`` (circuit qubit i is ``chain[i]``)
    that ``expressible`` accepts: an entangling chain, a mirror circuit, a single-qubit
    sequence, and every 2-qubit native on one pair when there are several."""
    table = profile.table
    n = len(chain)
    ones = [
        name
        for name in _unitary_natives(profile, 1)
        if all(_calibrated(table.gate(name, (p,))) for p in chain)
        and all(expressible(_op(name, (q,))) for q in range(n))
    ]
    mix = next((g for g in ones if _mixes(_op(g, (0,)))), None)
    pairs = [
        [op for op in _two_qubit_ops(profile, chain, i) if expressible(op)] for i in range(n - 1)
    ]
    entangles = n >= 2 and all(pairs)

    def layer(k: int) -> list[Op]:
        return [_op(mix, (q,)) for q in range(k)]  # type: ignore[arg-type]

    def entangle(k: int) -> list[Op]:
        return [pairs[i][0] for i in range(k - 1)]

    # Without a gate that makes a superposition, the entangling and mirror circuits act only
    # on |0...0>, where neither a 2-qubit gate's unitary nor coherent noise has any effect.
    circuits = []
    if mix is not None and entangles:
        circuits.append(Circuit("ghz_chain", n, (*layer(n), *entangle(n), *layer(n))))
    k = min(3, n)
    forward = []
    if mix is not None:
        forward = [*layer(k), *(entangle(k) if entangles else [])]
        forward += [_op(g, (q,)) for q in range(k) for g in ones]
    inverses = {op: _inverse(op) for op in forward}
    forward = [op for op in forward if inverses[op] is not None]
    if forward and (entangles or n == 1 or not _unitary_natives(profile, 2)):
        backward = [inv for op in reversed(forward) for inv in inverses[op]]  # type: ignore[union-attr]
        circuits.append(Circuit("mirror", k, (*forward, *backward)))
    if ones:
        k = min(2, n)
        sequence = [_op(g, (q,)) for q in range(k) for _ in range(2) for g in ones]
        circuits.append(Circuit("single_qubit", k, tuple(sequence)))
    if mix is not None and n >= 2 and len(pairs[0]) > 1:
        circuits.append(Circuit("two_qubit_natives", 2, (*layer(2), *pairs[0], *layer(2))))
    return tuple(circuits)


def _unitary_natives(profile: Profile, arity: int) -> list[str]:
    """Natives of ``arity`` with a registry unitary, most calibration records first."""
    records = Counter(r.gate for r in profile.calibrations)
    names = [n for n in profile.table.natives(arity) if gates.lookup(n) is not None]
    return sorted(names, key=lambda n: (-records[n], n))


def _two_qubit_ops(profile: Profile, chain: Sequence[int], i: int) -> list[Op]:
    """Each calibrated 2-qubit native on circuit pair (i, i+1), in an operand order it allows."""
    table = profile.table
    out = []
    for name in _unitary_natives(profile, 2):
        for a, b in ((i, i + 1), (i + 1, i)):
            found = table.gate(name, (chain[a], chain[b]))
            if _calibrated(found):
                op = _op(name, (a, b))
                if name == "rzz" and "zz" in profile.gates:
                    op = Op(name, op.qubits, (pi / 4,))  # pi/2 would be the zz gate
                out.append(op)
                break
    return out


def _calibrated(found: Any) -> bool:
    return isinstance(found, GateNoise) and found.state == "calibrated"


def _op(name: str, qubits: tuple[int, ...]) -> Op:
    info = gates.GATES[name]
    return Op(name, qubits, _ANGLES.get(name, (pi / 2,) * len(info.params)))


def _unitary(op: Op) -> np.ndarray:
    return gates.GATES[op.name].unitary(*op.params)  # type: ignore[misc]


def _mixes(op: Op) -> bool:
    """True when the gate takes |0> to a superposition."""
    return 0.1 < abs(_unitary(op)[0, 0]) ** 2 < 0.9


def _inverse(op: Op) -> list[Op] | None:
    """The gate repeated until the product is the identity up to phase, or None."""
    u = _unitary(op)
    power = u
    for k in range(1, _MAX_ORDER + 1):
        phase = power[0, 0]
        if abs(abs(phase) - 1) < 1e-9 and np.allclose(power, phase * np.eye(len(u)), atol=1e-9):
            return [op] * (k - 1)
        power = u @ power
    return None


# expected probabilities ---------------------------------------------------------------------


class _Expected:
    """Reference probabilities per circuit, plain and Pauli-twirled, computed once each."""

    def __init__(self, profile: Profile, chain: Sequence[int]) -> None:
        self.profile = profile
        self.chain = list(chain)
        self._cache: dict[tuple[int, tuple[Op, ...], bool], np.ndarray] = {}

    def __call__(self, circuit: Circuit, *, twirled: bool) -> np.ndarray:
        key = (circuit.num_qubits, circuit.ops, twirled)
        if key not in self._cache:
            layout = self.chain[: circuit.num_qubits]
            if twirled:
                self._cache[key] = _twirled(self.profile, circuit, layout)
            else:
                self._cache[key] = reference_probabilities(
                    self.profile, circuit.ops, circuit.num_qubits, layout=layout, readout=True
                )
        return self._cache[key]


def _twirled(profile: Profile, circuit: Circuit, layout: list[int]) -> np.ndarray:
    """The reference with each gate's channel replaced by its Pauli twirl (Stim's model)."""
    n, table = circuit.num_qubits, profile.table
    report = Report.start(profile, "reference", None)
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1.0
    for op in circuit.ops:
        rho = _apply(rho, [_unitary(op)], op.qubits, n)
        wires = tuple(layout[q] for q in op.qubits)
        built = resolve_op(table, op.name, wires, unknown_gates="error", report=report)
        if built.channels:
            rho = _apply(rho, pauli_kraus(pauli_twirl(built.channels, wires)), op.qubits, n)
    probs = np.real(np.diagonal(rho.reshape(2**n, 2**n)))
    return _with_readout(probs, [readout_matrix(table.qubit(q)) for q in layout])


def _with_readout(probs: np.ndarray, matrices: Sequence[np.ndarray | None]) -> np.ndarray:
    """Apply M[measured, prepared] per qubit to big-endian probabilities (None: no error)."""
    probs = np.asarray(probs, dtype=float).reshape((2,) * len(matrices))
    for axis, matrix in enumerate(matrices):
        if matrix is not None:
            probs = np.moveaxis(np.tensordot(matrix, probs, axes=([1], [axis])), 0, axis)
    return probs.reshape(-1)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# running ------------------------------------------------------------------------------------


class _Runner:
    """One framework: which ops it can express, and how it runs a circuit through its export."""

    framework: str
    version: str
    sampled = False
    twirled = False

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        self.profile = profile
        self.chain = chain

    def cannot_express(self, op: Op) -> str | None:
        raise NotImplementedError

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        """Exact probabilities, or sampled frequencies, big-endian over circuit qubits."""
        raise NotImplementedError

    def sample_measured(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray | None:
        """Frequencies through the framework's own measurement, when ``run`` bypasses it."""
        return None

    def reports(self) -> list[Report]:
        raise NotImplementedError


def _run(
    runner: _Runner,
    plan: Sequence[Circuit],
    expected: _Expected,
    shots: int,
    seed: int | None,
) -> FrameworkCheck | str:
    """The framework's check on the circuits it can express, or why it ran none."""
    circuits = build_circuits(
        runner.profile, runner.chain, lambda op: not runner.cannot_express(op)
    )
    built = {c.name for c in circuits}
    not_run = [
        (c.name, "; ".join(dict.fromkeys(filter(None, map(runner.cannot_express, c.ops)))))
        for c in plan
        if c.name not in built
    ]
    checks = [
        _compare(
            c,
            runner.run(c, shots, seed),
            expected(c, twirled=runner.twirled),
            shots,
            runner.sampled,
        )
        for c in circuits
    ]
    if not checks:
        reasons = dict.fromkeys(part for _, why in not_run for part in why.split("; "))
        return "no check circuit can be expressed: " + "; ".join(reasons)
    widest = max(circuits, key=lambda c: c.num_qubits)
    measured = runner.sample_measured(widest, shots, seed)
    if measured is not None:
        checks.append(_compare(widest, measured, expected(widest, twirled=False), shots, True))
    reports = runner.reports()

    def count(attr: str) -> int:
        return len({str(item) for r in reports for item in getattr(r, attr)})

    method = (
        f"sampled {shots} shots vs the twirled reference, {SIGMAS:g} sigma"
        if runner.sampled
        else "exact"
        if measured is None
        else f"exact, and {widest.name} sampled {shots} shots through the framework's"
        f" measurement, {SIGMAS:g} sigma"
    )
    return FrameworkCheck(
        framework=runner.framework,
        version=runner.version,
        method=method,
        circuits=tuple(checks),
        not_run=tuple(not_run),
        approximated=count("approximated"),
        omitted=count("omitted"),
        unknown=count("unknown"),
        report=reports[0].summary(),
    )


def _compare(
    circuit: Circuit, got: np.ndarray, want: np.ndarray, shots: int, sampled: bool
) -> CircuitCheck:
    tvd = _tvd(got, want)
    if sampled:
        bound = SIGMAS * np.sqrt(want * (1 - want) / shots) + SIGMAS / shots
        passed = bool(np.all(np.abs(got - want) <= bound))
        tolerance = min(1.0, 0.5 * float(bound.sum()))
    else:
        passed, tolerance = tvd <= EXACT_TOLERANCE, EXACT_TOLERANCE
    names = tuple(sorted({op.name for op in circuit.ops}))
    return CircuitCheck(circuit.name, circuit.num_qubits, names, tvd, tolerance, passed, sampled)


def _registry_missing(op: Op, framework: str, column: str | None) -> str | None:
    return None if column else f"{framework} has no {op.name} gate"


class _Qiskit(_Runner):
    framework = "qiskit"

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        super().__init__(profile, chain)
        import qiskit
        import qiskit_aer
        from qiskit.circuit import library

        from .frameworks.qiskit import ALIASES, gate_class, to_qiskit

        self.version = f"{qiskit.__version__} (qiskit-aer {qiskit_aer.__version__})"
        self._library, self._aliases, self._gate_class = library, ALIASES, gate_class
        self.sim = to_qiskit(profile)
        self.sim.set_options(method="density_matrix")
        self._readout = {
            e["gate_qubits"][0][0]: np.array(e["probabilities"]).T  # Aer rows: prepared state
            for e in self.sim.noise_model.to_dict()["errors"]
            if e["type"] == "roerror"
        }

    def _gate(self, op: Op) -> Any:
        alias = self._aliases.get(op.name)
        if alias is not None:
            if alias in self.profile.gates or op.params not in ((), (0.0, 0.0)):
                return None
            return getattr(self._library, gates.GATES[alias].qiskit_class)(pi / 2)
        cls = self._gate_class(op.name)
        return None if cls is None else cls(*op.params)

    def cannot_express(self, op: Op) -> str | None:
        gate = self._gate(op)
        if gate is None:
            return f"Qiskit has no {op.name} gate"
        qargs = tuple(self.chain[q] for q in op.qubits)
        if not self.sim.target.instruction_supported(gate.name, qargs):
            return f"the Qiskit export has no {gate.name} on qubits {list(qargs)}"
        return None

    def _circuit(self, circuit: Circuit, clbits: int = 0) -> Any:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.sim.target.num_qubits, clbits)
        for op in circuit.ops:
            qc.append(self._gate(op), [self.chain[q] for q in op.qubits])
        return qc

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        # Exact: Aer's probabilities before measurement, then the exported ReadoutError
        # matrices. sample_measured covers Aer applying them at a real measurement.
        qc = self._circuit(circuit)
        measured = self.chain[: circuit.num_qubits]
        qc.save_probabilities(measured)
        little = np.asarray(self.sim.run(qc).result().data()["probabilities"])
        n = circuit.num_qubits
        big = little.reshape((2,) * n).transpose(range(n - 1, -1, -1)).reshape(-1)
        return _with_readout(big, [self._readout.get(q) for q in measured])

    def sample_measured(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        n = circuit.num_qubits
        qc = self._circuit(circuit, n)
        qc.measure(self.chain[:n], range(n))
        counts = self.sim.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
        freq = np.zeros(2**n)
        for bits, count in counts.items():
            freq[int(bits[::-1], 2)] = count / shots  # clbit 0 is the rightmost character
        return freq

    def reports(self) -> list[Report]:
        return [self.sim.report]


class _Cirq(_Runner):
    framework = "cirq"

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        super().__init__(profile, chain)
        import cirq

        from .frameworks.cirq import ECRGate, to_cirq

        self.cirq, self.version = cirq, cirq.__version__
        self.qubits = cirq.LineQubit.range(len(chain))
        self.model = to_cirq(profile, layout=dict(zip(self.qubits, chain, strict=True)))
        c = cirq
        self._gates: dict[str, Callable[..., Any]] = {
            "id": lambda: c.I,
            "x": lambda: c.X,
            "y": lambda: c.Y,
            "z": lambda: c.Z,
            "h": lambda: c.H,
            "s": lambda: c.S,
            "sdg": lambda: c.S**-1,
            "t": lambda: c.T,
            "tdg": lambda: c.T**-1,
            "sx": lambda: c.X**0.5,
            "sxdg": lambda: c.X**-0.5,
            "rx": c.rx,
            "ry": c.ry,
            "rz": c.rz,
            "r": lambda t, p: c.PhasedXPowGate(phase_exponent=p / pi, exponent=t / pi),
            "cx": lambda: c.CNOT,
            "cz": lambda: c.CZ,
            "ecr": ECRGate,
            "iswap": lambda: c.ISWAP,
            "sqrt_iswap": lambda: c.ISWAP**0.5,
            "zz": lambda: c.ZZ**0.5,
            "rzz": lambda t: c.ZZPowGate(exponent=t / pi),
            "rxx": lambda t: c.XXPowGate(exponent=t / pi),
            "ryy": lambda t: c.YYPowGate(exponent=t / pi),
            "ms": lambda p0, p1: c.ms(pi / 4) if p0 == p1 == 0 else None,
        }

    def cannot_express(self, op: Op) -> str | None:
        make = self._gates.get(op.name)
        return (
            None
            if make is not None and make(*op.params) is not None
            else (f"Cirq has no {op.name} gate")
        )

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        cirq = self.cirq
        measured = self.qubits[: circuit.num_qubits]
        ops = [
            self._gates[op.name](*op.params).on(*(self.qubits[q] for q in op.qubits))
            for op in circuit.ops
        ]
        noisy = cirq.Circuit([*ops, cirq.measure(*measured, key="m")]).with_noise(self.model)
        # Nothing after the measurement can change its outcome, so the state just before it,
        # readout channels included, gives the exact outcome probabilities.
        end = next(i for i, m in enumerate(noisy) if any(cirq.is_measurement(op) for op in m))
        quantum = noisy[:end]
        simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
        rho = simulator.simulate(quantum, qubit_order=measured).final_density_matrix
        return np.real(np.diag(rho))

    def reports(self) -> list[Report]:
        return [self.model.report]


class _PennyLane(_Runner):
    framework = "pennylane"

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        super().__init__(profile, chain)
        import pennylane as qml

        from .frameworks.pennylane import operation_for, to_pennylane

        self.qml, self.version = qml, qml.__version__
        self.model = to_pennylane(profile, layout=chain)
        self._operation_for = operation_for

    def _cls(self, op: Op) -> Any:
        return self._operation_for(op.name)

    def cannot_express(self, op: Op) -> str | None:
        return None if self._cls(op) is not None else f"PennyLane has no {op.name} gate"

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        qml, n = self.qml, circuit.num_qubits

        @qml.qnode(qml.device("default.mixed", wires=n))
        def run() -> Any:
            for op in circuit.ops:
                self._cls(op)(*op.params, wires=list(op.qubits))
            return qml.probs(wires=range(n))

        return np.asarray(qml.add_noise(run, self.model)(), dtype=float)

    def reports(self) -> list[Report]:
        return [self.model.report]


class _Stim(_Runner):
    framework = "stim"
    sampled = True
    twirled = True

    def __init__(self, profile: Profile, chain: list[int]) -> None:
        super().__init__(profile, chain)
        import stim

        from .frameworks.stim import sample_with_readout, to_stim

        self.version = stim.__version__
        self._to_stim, self._sample = to_stim, sample_with_readout
        self._reports: list[Report] = []

    def cannot_express(self, op: Op) -> str | None:
        info = gates.GATES[op.name]
        if info.params:
            return f"{op.name} takes an angle; Stim circuits hold only fixed Clifford gates"
        return None if info.stim else f"Stim has no {op.name} instruction"

    def run(self, circuit: Circuit, shots: int, seed: int | None) -> np.ndarray:
        n = circuit.num_qubits
        lines = [
            f"{gates.GATES[op.name].stim[0]} {' '.join(map(str, op.qubits))}" for op in circuit.ops
        ]
        lines.append("M " + " ".join(map(str, range(n))))
        noisy = self._to_stim(
            self.profile, "\n".join(lines), layout=dict(enumerate(self.chain)), readout="exact"
        )
        self._reports.append(noisy.report)
        bits = self._sample(noisy, shots, seed=seed)
        index = bits.astype(int) @ (1 << np.arange(n)[::-1])
        return np.bincount(index, minlength=2**n) / shots

    def reports(self) -> list[Report]:
        return self._reports


_RUNNERS: dict[str, type[_Runner]] = {
    "qiskit": _Qiskit,
    "cirq": _Cirq,
    "pennylane": _PennyLane,
    "stim": _Stim,
}
