from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Operation:
    name: str
    wires: tuple[int, ...]
    params: tuple[float, ...] = ()

    def inverse(self) -> Operation:
        if self.name in {"h", "x", "cx"}:
            return self
        if self.name in {"rx", "rz"}:
            return Operation(self.name, self.wires, tuple(-value for value in self.params))
        raise ValueError(f"No inverse defined for {self.name}")


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    num_qubits: int
    operations: tuple[Operation, ...]
    family: str
    depth: int | None = None


def ghz(num_qubits: int) -> CircuitSpec:
    if num_qubits < 2:
        raise ValueError("GHZ benchmark requires at least two qubits.")
    operations = [Operation("h", (0,))]
    operations.extend(Operation("cx", (i, i + 1)) for i in range(num_qubits - 1))
    return CircuitSpec(f"ghz_n{num_qubits}", num_qubits, tuple(operations), "ghz")


def mirror(num_qubits: int, depth: int, seed: int) -> CircuitSpec:
    rng = np.random.default_rng(seed + 1009 * num_qubits + depth)
    forward: list[Operation] = []
    for layer in range(depth):
        for wire in range(num_qubits):
            if rng.random() < 0.5:
                forward.append(Operation("rx", (wire,), (float(rng.choice([np.pi / 2, -np.pi / 2])),)))
            else:
                forward.append(Operation("rz", (wire,), (float(rng.choice([np.pi / 2, -np.pi / 2])),)))
        start = layer % 2
        for wire in range(start, num_qubits - 1, 2):
            forward.append(Operation("cx", (wire, wire + 1)))
    inverse = [operation.inverse() for operation in reversed(forward)]
    return CircuitSpec(
        f"mirror_n{num_qubits}_d{depth}",
        num_qubits,
        tuple(forward + inverse),
        "mirror",
        depth,
    )


def rb_style(num_qubits: int, depth: int, seed: int) -> CircuitSpec:
    rng = np.random.default_rng(seed + 7919 * num_qubits + depth)
    forward: list[Operation] = []
    angles = [np.pi / 2, -np.pi / 2, np.pi]
    for layer in range(depth):
        for wire in range(num_qubits):
            gate_name = "rx" if rng.random() < 0.5 else "rz"
            forward.append(Operation(gate_name, (wire,), (float(rng.choice(angles)),)))
        if num_qubits > 1:
            control = layer % (num_qubits - 1)
            forward.append(Operation("cx", (control, control + 1)))
    inverse = [operation.inverse() for operation in reversed(forward)]
    return CircuitSpec(
        f"rbstyle_n{num_qubits}_d{depth}",
        num_qubits,
        tuple(forward + inverse),
        "rb_style",
        depth,
    )


def pilot_benchmarks(
    subset_sizes: list[int],
    mirror_depths: list[int],
    rb_depths: list[int],
    seed: int,
) -> list[CircuitSpec]:
    circuits: list[CircuitSpec] = []
    for size in subset_sizes:
        circuits.append(ghz(size))
        circuits.extend(mirror(size, depth, seed) for depth in mirror_depths)
        circuits.extend(rb_style(size, depth, seed) for depth in rb_depths)
    return circuits
