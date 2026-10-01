"""Circuit-to-device qubit mappings: validation shared by every framework, and a chain helper."""

from __future__ import annotations

import operator
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from . import gates
from .errors import LayoutError, NoiseVaultWarning
from .report import warn_from_caller
from .table import GateNoise

if TYPE_CHECKING:
    from .profile import Profile
    from .table import NoiseTable

_BEAMS = (32, 128, 512)


@dataclass(frozen=True, order=True)
class _Cost:
    """A chain's score: qubits lacking a 1-qubit gate their usable gates cannot make first,
    then qubits with no usable 1-qubit gate, then qubits with unknown readout, then the summed
    error."""

    incomplete: int
    no_gate: int
    no_readout: int
    error: float

    def __add__(self, other: _Cost) -> _Cost:
        return _Cost(
            self.incomplete + other.incomplete,
            self.no_gate + other.no_gate,
            self.no_readout + other.no_readout,
            self.error + other.error,
        )


def _int_label(label: Hashable) -> int | None:
    if isinstance(label, bool):
        return None
    try:
        return operator.index(label)  # type: ignore[arg-type]
    except TypeError:
        return None


def normalize_layout(
    labels: Iterable[Hashable],
    layout: Mapping[Hashable, int] | Sequence[int] | None,
    profile: Profile,
    *,
    index_of: Callable[[Hashable], int | None] = _int_label,
) -> dict[Hashable, int]:
    """Map every circuit qubit label to a usable physical qubit, or raise LayoutError.

    With no ``layout``, labels that ``index_of`` turns into integers map to themselves
    (integers by default; adapters pass their own rule, e.g. for Cirq ``LineQubit``). A sequence
    layout maps label ``i`` to ``layout[i]``. The result must be complete and injective and use
    in-range, enabled qubits.
    """
    labels = list(dict.fromkeys(labels))
    table = profile.table
    if layout is None:
        mapping: dict[Hashable, int] = {}
        for label in labels:
            index = index_of(label)
            if index is None:
                raise LayoutError(
                    f"qubit {label!r} has no integer index; pass layout={{{label!r}: <physical"
                    " qubit>, ...} covering every circuit qubit"
                )
            mapping[label] = index
    elif isinstance(layout, Mapping):
        mapping = dict(layout)
    else:
        mapping = dict(enumerate(layout))

    missing = [label for label in labels if label not in mapping]
    if missing:
        raise LayoutError(f"layout has no physical qubit for {missing!r}; map every circuit qubit")
    result: dict[Hashable, int] = {}
    owner: dict[int, Hashable] = {}
    for label in labels:
        physical = _int_label(mapping[label])
        if physical is None:
            raise LayoutError(f"layout maps {label!r} to {mapping[label]!r}, not a qubit index")
        if not 0 <= physical < table.num_qubits:
            raise LayoutError(
                f"layout maps {label!r} to qubit {physical}, but {profile.id} has qubits"
                f" 0..{table.num_qubits - 1}"
            )
        if table.qubit(physical).disabled:
            raise LayoutError(
                f"layout maps {label!r} to qubit {physical}, which {profile.id} marks disabled;"
                " choose another qubit (profile.suggest_layout(n) proposes a usable chain)"
            )
        if physical in owner:
            raise LayoutError(
                f"layout maps both {owner[physical]!r} and {label!r} to qubit {physical}"
            )
        owner[physical] = label
        result[label] = physical
    return result


def suggest_layout(profile: Profile, n: int) -> dict[int, int]:
    """A connected chain of ``n`` well-calibrated qubits as ``{0: p0, 1: p1, ...}``.

    Deterministic beam search minimizing the summed average infidelity of the typical 1-qubit
    gate, mean readout error and typical 2-qubit gate along the chain. A pair with no usable
    2-qubit gate is never a link; a pair with one is, whether connectivity lists it or only a
    calibration record does.

    Every qubit in the chain can run what the profile's 1-qubit gates can, when such a chain
    exists. A qubit falls short when a 1-qubit gate the profile defines is disabled on it and
    the gates still usable there cannot make it exactly. That is judged from the registry
    unitaries: the usable gates' continuous rotations, closed under commutators and under
    conjugation by every usable gate, must contain the disabled gate's rotations. So rz and sx
    make any rotation and an IBM qubit without x still qualifies, while rz and x make only z
    rotations and flips and one without sx does not. A transpiler cannot place an arbitrary
    1-qubit gate on such a qubit. When no chain avoids them, the chain uses as few as it can
    and a NoiseVaultWarning names them.

    Missing calibration ranks next: a chain with fewer qubits lacking a calibrated 1-qubit gate
    always wins, then one with fewer unknown readout errors, whatever the calibrated errors. On
    an all-to-all device it takes the ``n`` qubits with the lowest 1-qubit and readout cost
    when every consecutive pair among them is usable, and otherwise searches as on any other
    device. It is a starting point for small experiments, not a circuit placer.
    """
    table = profile.table
    if not 1 <= n <= table.num_qubits:
        raise LayoutError(f"cannot choose {n} qubits on {profile.id} ({table.num_qubits} qubits)")
    usable = [q for q in range(table.num_qubits) if not table.qubit(q).disabled]
    if len(usable) < n:
        raise LayoutError(f"{profile.id} has only {len(usable)} usable qubits, not {n}")
    lacking = {q: _lacking(table, q) for q in usable}
    qubit_cost = {q: _qubit_cost(table, q, lacking[q]) for q in usable}
    edge_costs: dict[tuple[int, int], _Cost | None] = {}

    def edge(a: int, b: int) -> _Cost | None:
        key = (min(a, b), max(a, b))
        if key not in edge_costs:
            edge_costs[key] = _edge_cost(table, *key)
        return edge_costs[key]

    complete = [q for q in usable if not lacking[q]]
    path = _chain(table, complete, qubit_cost, edge, n) or _chain(
        table, usable, qubit_cost, edge, n
    )
    if path is None:
        raise LayoutError(f"{profile.id} has no connected chain of {n} usable qubits")
    short = [f"{q} ({', '.join(lacking[q])} disabled)" for q in path if lacking[q]]
    if short:
        warn_from_caller(
            f"{profile.id} has no connected chain of {n} qubits that can each make every"
            f" 1-qubit gate, so this one includes qubit{'s' if len(short) > 1 else ''}"
            f" {', '.join(short)}; a transpiler may fail to place 1-qubit gates there",
            NoiseVaultWarning,
        )
    return dict(enumerate(path))


def _chain(
    table: NoiseTable,
    pool: list[int],
    qubit_cost: dict[int, _Cost],
    edge: Callable[[int, int], _Cost | None],
    n: int,
) -> tuple[int, ...] | None:
    if len(pool) < n:
        return None
    if table.all_to_all:
        chain = tuple(sorted(pool, key=lambda q: (qubit_cost[q], q))[:n])
        if all(edge(a, b) is not None for a, b in zip(chain, chain[1:], strict=False)):
            return chain
    neighbors = _neighbors(table, pool)
    for width in _BEAMS:  # a wider beam only when a narrow one walks into dead ends
        path = _beam_search(pool, qubit_cost, neighbors, edge, n, width)
        if path is not None:
            return path
    return None


def _beam_search(
    usable: list[int],
    qubit_cost: dict[int, _Cost],
    neighbors: dict[int, list[int]],
    edge: Callable[[int, int], _Cost | None],
    n: int,
    width: int,
) -> tuple[int, ...] | None:
    beam = sorted((qubit_cost[q], (q,)) for q in usable)[:width]
    for _ in range(n - 1):
        grown: dict[tuple[int, ...], _Cost] = {}
        for cost, path in beam:
            members = set(path)
            for at_tail, end in ((True, path[-1]), (False, path[0])):
                for nb in neighbors[end]:
                    step = None if nb in members else edge(end, nb)
                    if step is None:
                        continue
                    new = path + (nb,) if at_tail else (nb, *path)
                    key = min(new, new[::-1])
                    total = cost + step + qubit_cost[nb]
                    if key not in grown or total < grown[key]:
                        grown[key] = total
        if not grown:
            return None
        beam = sorted((cost, path) for path, cost in grown.items())[:width]
    return beam[0][1]


def _qubit_cost(table: NoiseTable, q: int, lacking: tuple[str, ...]) -> _Cost:
    one = table.typical(1, (q,))
    readout = table.qubit(q).readout
    gate_error = one.avg_infidelity if isinstance(one, GateNoise) else None
    readout_error = None if readout is None else sum(readout) / 2
    return _Cost(
        int(bool(lacking)),
        int(gate_error is None),
        int(readout_error is None),
        (gate_error or 0.0) + (readout_error or 0.0),
    )


def _edge_cost(table: NoiseTable, a: int, b: int) -> _Cost | None:
    costs = [
        found.avg_infidelity
        for found in (table.typical(2, (a, b)), table.typical(2, (b, a)))
        if isinstance(found, GateNoise) and found.avg_infidelity is not None
    ]
    return _Cost(0, 0, 0, min(costs)) if costs else None


def _neighbors(table: NoiseTable, usable: list[int]) -> dict[int, list[int]]:
    allowed = set(usable)
    out: dict[int, set[int]] = {q: set() for q in usable}
    pairs = combinations(usable, 2) if table.all_to_all else table.listed_pairs()
    for a, b in pairs:
        if a in allowed and b in allowed:
            out[a].add(b)
            out[b].add(a)
    return {q: sorted(nbs) for q, nbs in out.items()}


def _lacking(table: NoiseTable, q: int) -> tuple[str, ...]:
    """The profile's 1-qubit gates disabled on ``q`` that its usable 1-qubit gates cannot make."""
    defined = [
        name
        for name, spec in table.profile.gates.items()
        if not spec.disabled and table.arity(name) == 1 and _maybe_unitary(name)
    ]
    usable = tuple(name for name in defined if table.allowed(name, (q,)))
    off = tuple(name for name in defined if name not in usable)
    return _unreachable(usable, off) if off else ()


def _maybe_unitary(name: str) -> bool:
    info = gates.lookup(name)
    return info is None or info.unitary is not None


# One-qubit gates as rotations: su(2) is R^3 with the cross product as commutator, and
# conjugating by a unitary rotates that space.
_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
_AT = (0.7, 1.1, 1.9)  # generic parameters, where no rotation's tangent vanishes
_STEP = 1e-6
_TOLERANCE = 1e-6


@cache
def _unreachable(usable: tuple[str, ...], off: tuple[str, ...]) -> tuple[str, ...]:
    reach = _algebra(usable)
    return tuple(name for name in off if len(_span([*reach, *_needs(name)])) > len(reach))


def _algebra(names: tuple[str, ...]) -> np.ndarray:
    """The rotations the gates ``names`` make with continuous parameters, as orthonormal rows."""
    known = [info for info in map(gates.lookup, names) if info and info.arity == 1]
    adjoints = [_adjoint(_at(info)) for info in known]
    basis = _span(v for info in known for v in _tangents(info))
    while True:
        brackets = (np.cross(a, b) for a, b in combinations(basis, 2))
        turned = (r @ v for r in adjoints for v in basis)
        grown = _span([*basis, *brackets, *turned])
        if len(grown) == len(basis):
            return grown
        basis = grown


def _needs(name: str) -> list[np.ndarray]:
    """The rotations a transpiler must make to stand in for gate ``name``."""
    info = gates.lookup(name)
    if info is None or info.arity != 1:
        return list(np.eye(3))
    return [*_tangents(info), _rotation(_at(info))]


def _at(info: gates.GateInfo) -> np.ndarray:
    return info.unitary(*_AT[: len(info.params)])  # type: ignore[misc]


def _tangents(info: gates.GateInfo) -> list[np.ndarray]:
    here = _at(info).conj().T
    out = []
    for k in range(len(info.params)):
        moved = list(_AT[: len(info.params)])
        moved[k] += _STEP
        out.append(_rotation(here @ info.unitary(*moved)) / _STEP)  # type: ignore[misc]
    return out


def _rotation(u: np.ndarray) -> np.ndarray:
    """``u`` up to phase as sin(angle/2) times its rotation axis."""
    special = u / np.sqrt(np.linalg.det(u))
    return np.real(0.5j * np.einsum("ij,kji->k", special, _PAULI))


def _adjoint(u: np.ndarray) -> np.ndarray:
    """The rotation of su(2) that conjugating by ``u`` performs."""
    return np.real(np.einsum("kij,jl,mlo,oi->km", _PAULI, u, _PAULI, u.conj().T)) / 2


def _span(vectors: Iterable[np.ndarray]) -> np.ndarray:
    rows = np.array(list(vectors), dtype=float).reshape(-1, 3)
    if not len(rows):
        return rows
    _, sizes, directions = np.linalg.svd(rows)
    return directions[: int((sizes > _TOLERANCE).sum())]
