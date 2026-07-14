from __future__ import annotations

from collections import defaultdict, deque

from .models import DeviceNoiseSnapshot


def _qubit_score(snapshot: DeviceNoiseSnapshot, index: int) -> float:
    qubit = next(q for q in snapshot.qubits if q.index == index)
    if not qubit.operational:
        return float("inf")
    score = float(qubit.readout_error or 0.0)
    one_qubit_errors = [
        gate.error
        for gate in snapshot.gates
        if gate.operational and gate.error is not None and gate.qubits == [index]
    ]
    if one_qubit_errors:
        score += min(one_qubit_errors)
    return score


def _edge_score(snapshot: DeviceNoiseSnapshot, a: int, b: int) -> float:
    errors = [
        gate.error
        for gate in snapshot.gates
        if gate.operational
        and gate.error is not None
        and len(gate.qubits) == 2
        and set(gate.qubits) == {a, b}
    ]
    return min(errors) if errors else 0.1


def best_connected_subset(snapshot: DeviceNoiseSnapshot, size: int) -> list[int]:
    if size < 1 or size > snapshot.num_qubits:
        raise ValueError(f"Invalid subset size {size} for {snapshot.num_qubits}-qubit snapshot.")

    operational = {q.index for q in snapshot.qubits if q.operational}
    graph: dict[int, set[int]] = defaultdict(set)
    for edge in snapshot.coupling_map:
        if len(edge) == 2 and edge[0] in operational and edge[1] in operational:
            graph[edge[0]].add(edge[1])
            graph[edge[1]].add(edge[0])

    if size == 1:
        return [min(operational, key=lambda q: _qubit_score(snapshot, q))]

    candidates: list[tuple[float, tuple[int, ...]]] = []
    for seed in sorted(operational):
        chosen = [seed]
        frontier = set(graph[seed])
        while len(chosen) < size and frontier:
            next_qubit = min(
                frontier,
                key=lambda q: _qubit_score(snapshot, q)
                + min(_edge_score(snapshot, q, existing) for existing in chosen if q in graph[existing]),
            )
            chosen.append(next_qubit)
            frontier.remove(next_qubit)
            frontier.update(graph[next_qubit] - set(chosen))
        if len(chosen) == size:
            chosen_tuple = tuple(sorted(chosen))
            qubit_cost = sum(_qubit_score(snapshot, q) for q in chosen_tuple)
            edge_cost = sum(
                _edge_score(snapshot, a, b)
                for a in chosen_tuple
                for b in chosen_tuple
                if a < b and b in graph[a]
            )
            candidates.append((qubit_cost + edge_cost, chosen_tuple))

    if not candidates:
        # Disconnected fallback is explicit and deterministic; caller records it in the report.
        ranked = sorted(operational, key=lambda q: (_qubit_score(snapshot, q), q))
        if len(ranked) < size:
            raise ValueError(f"Only {len(ranked)} operational qubits available.")
        return ranked[:size]
    return list(min(candidates)[1])


def connected(snapshot: DeviceNoiseSnapshot, qubits: list[int]) -> bool:
    if len(qubits) <= 1:
        return True
    allowed = set(qubits)
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in snapshot.coupling_map:
        if a in allowed and b in allowed:
            graph[a].add(b)
            graph[b].add(a)
    visited = {qubits[0]}
    queue = deque([qubits[0]])
    while queue:
        current = queue.popleft()
        for neighbour in graph[current] - visited:
            visited.add(neighbour)
            queue.append(neighbour)
    return visited == allowed
