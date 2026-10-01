"""Circuit-to-device qubit mappings: validation shared by every framework, and a chain helper."""

from __future__ import annotations

import operator
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import LayoutError
from .table import GateNoise

if TYPE_CHECKING:
    from .profile import Profile
    from .table import NoiseTable

_BEAMS = (32, 128, 512)


@dataclass(frozen=True, order=True)
class _Cost:
    """A chain's score: values with no calibration (compared first), then the summed error."""

    missing: int
    error: float

    def __add__(self, other: _Cost) -> _Cost:
        return _Cost(self.missing + other.missing, self.error + other.error)


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
    gate, mean readout error and typical 2-qubit gate along the chain. A qubit with no usable
    1-qubit calibration or an unknown readout error ranks last: a chain with fewer such gaps always
    beats one with more, whatever the calibrated errors. A pair with no usable 2-qubit gate is
    never a link. On an all-to-all device it takes the ``n`` qubits with the lowest 1-qubit and
    readout cost when every consecutive pair among them is usable, and otherwise searches as on
    any other device. It is a starting point for small experiments, not a circuit placer.
    """
    table = profile.table
    if not 1 <= n <= table.num_qubits:
        raise LayoutError(f"cannot choose {n} qubits on {profile.id} ({table.num_qubits} qubits)")
    usable = [q for q in range(table.num_qubits) if not table.qubit(q).disabled]
    qubit_cost = {q: _qubit_cost(table, q) for q in usable}
    edge_costs: dict[tuple[int, int], _Cost | None] = {}

    def edge(a: int, b: int) -> _Cost | None:
        key = (min(a, b), max(a, b))
        if key not in edge_costs:
            edge_costs[key] = _edge_cost(table, *key)
        return edge_costs[key]

    if table.all_to_all:
        if len(usable) < n:
            raise LayoutError(f"{profile.id} has only {len(usable)} usable qubits, not {n}")
        chain = sorted(usable, key=lambda q: (qubit_cost[q], q))[:n]
        if all(edge(a, b) is not None for a, b in zip(chain, chain[1:], strict=False)):
            return dict(enumerate(chain))
    neighbors = _neighbors(table, usable)
    for width in _BEAMS:  # a wider beam only when a narrow one walks into dead ends
        path = _beam_search(usable, qubit_cost, neighbors, edge, n, width)
        if path is not None:
            return dict(enumerate(path))
    raise LayoutError(f"{profile.id} has no connected chain of {n} usable qubits")


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


def _qubit_cost(table: NoiseTable, q: int) -> _Cost:
    one = table.typical(1, (q,))
    readout = table.qubit(q).readout
    errors = [
        one.avg_infidelity if isinstance(one, GateNoise) else None,
        None if readout is None else sum(readout) / 2,
    ]
    known = [e for e in errors if e is not None]
    return _Cost(len(errors) - len(known), sum(known))


def _edge_cost(table: NoiseTable, a: int, b: int) -> _Cost | None:
    costs = [
        found.avg_infidelity
        for found in (table.typical(2, (a, b)), table.typical(2, (b, a)))
        if isinstance(found, GateNoise) and found.avg_infidelity is not None
    ]
    return _Cost(0, min(costs)) if costs else None


def _neighbors(table: NoiseTable, usable: list[int]) -> dict[int, list[int]]:
    allowed = set(usable)
    out: dict[int, set[int]] = {q: set() for q in usable}
    for a, b in table.edges():
        if a in allowed and b in allowed:
            out[a].add(b)
            out[b].add(a)
    return {q: sorted(nbs) for q, nbs in out.items()}
