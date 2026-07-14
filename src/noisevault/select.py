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
        and tuple(gate.qubits) == (a, b)
        and gate.name in {"cx", "ecr", "cz"}
    ]
    return min(errors) if errors else float("inf")


def _directed_entangling_graph(snapshot: DeviceNoiseSnapshot) -> dict[int, set[int]]:
    operational = {q.index for q in snapshot.qubits if q.operational}
    coupling = {tuple(edge) for edge in snapshot.coupling_map if len(edge) == 2}
    graph: dict[int, set[int]] = defaultdict(set)
    for gate in snapshot.gates:
        if (
            gate.operational
            and gate.name in {"cx", "ecr", "cz"}
            and len(gate.qubits) == 2
            and set(gate.qubits).issubset(operational)
            and (
                tuple(gate.qubits) in coupling
                or tuple(reversed(gate.qubits)) in coupling
            )
        ):
            graph[gate.qubits[0]].add(gate.qubits[1])
    return graph


def _path_score(snapshot: DeviceNoiseSnapshot, path: tuple[int, ...]) -> float:
    return sum(_qubit_score(snapshot, q) for q in path) + sum(
        _edge_score(snapshot, a, b) for a, b in zip(path, path[1:], strict=False)
    )


def _calibrated_path_candidates(
    snapshot: DeviceNoiseSnapshot, size: int
) -> list[tuple[float, tuple[int, ...]]]:
    operational = {q.index for q in snapshot.qubits if q.operational}
    if size == 1:
        return [(_qubit_score(snapshot, q), (q,)) for q in sorted(operational)]

    graph = _directed_entangling_graph(snapshot)
    paths: set[tuple[int, ...]] = set()

    def extend(path: tuple[int, ...]) -> None:
        if len(path) == size:
            paths.add(path)
            return
        for neighbour in sorted(graph[path[-1]] - set(path)):
            extend((*path, neighbour))

    for seed in sorted(operational):
        extend((seed,))
    return sorted((_path_score(snapshot, path), path) for path in paths)


def best_connected_subset(snapshot: DeviceNoiseSnapshot, size: int) -> list[int]:
    if size < 1 or size > snapshot.num_qubits:
        raise ValueError(f"Invalid subset size {size} for {snapshot.num_qubits}-qubit snapshot.")

    candidates = _calibrated_path_candidates(snapshot, size)
    if not candidates:
        raise ValueError(
            f"No directed, calibrated entangling path of {size} operational qubits is available."
        )
    return list(min(candidates)[1])


def best_common_connected_path(
    first: DeviceNoiseSnapshot, second: DeviceNoiseSnapshot, size: int
) -> list[int]:
    """Return a directed calibrated path valid in both dated snapshots."""
    second_paths = {path for _, path in _calibrated_path_candidates(second, size)}
    candidates = [
        (_path_score(first, path) + _path_score(second, path), path)
        for _, path in _calibrated_path_candidates(first, size)
        if path in second_paths
    ]
    if not candidates:
        raise ValueError(
            f"No shared directed, calibrated path of {size} operational qubits is available."
        )
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
