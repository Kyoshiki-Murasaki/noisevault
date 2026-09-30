"""Gate-error metrics: conversions, bounds and Pauli-vector ordering, all in one place.

For an n-qubit operation with d = 2**n:
  r      average gate infidelity, 1 - F_avg
  e_pro  process (entanglement) infidelity, r (d + 1) / d
  lambda Qiskit depolarizing parameter, (1 - lambda) rho + lambda I/d, so r = lambda (d - 1) / d
  pauli  non-identity Pauli probabilities; their sum is e_pro

Pauli vectors list labels in lexicographic I, X, Y, Z order without the identity. For two
qubits this is Stim's PAULI_CHANNEL_2 order IX, IY, IZ, XI, XX, ..., ZZ, and the first letter
acts on ``qubits[0]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product
from typing import Literal

MetricKind = Literal["avg_infidelity", "process_infidelity", "depolarizing_param", "pauli"]
METRIC_KEYS: tuple[MetricKind, ...] = (
    "avg_infidelity",
    "process_infidelity",
    "depolarizing_param",
    "pauli",
)
_TOL = 1e-12


def dim(num_qubits: int) -> int:
    return 2**num_qubits


def max_avg_infidelity(num_qubits: int) -> float:
    d = dim(num_qubits)
    return d / (d + 1)


def max_depolarizing_param(num_qubits: int) -> float:
    """Largest lambda of a CPTP depolarizing channel: d^2 / (d^2 - 1)."""
    d = dim(num_qubits)
    return d * d / (d * d - 1)


def process_from_avg(r: float, num_qubits: int) -> float:
    d = dim(num_qubits)
    return r * (d + 1) / d


def avg_from_process(e_pro: float, num_qubits: int) -> float:
    d = dim(num_qubits)
    return e_pro * d / (d + 1)


def avg_from_depolarizing(lam: float, num_qubits: int) -> float:
    d = dim(num_qubits)
    return lam * (d - 1) / d


def depolarizing_from_avg(r: float, num_qubits: int) -> float:
    d = dim(num_qubits)
    return r * d / (d - 1)


def pauli_arity(length: int) -> int:
    """Number of qubits of a Pauli vector with ``length`` non-identity entries."""
    n = round(math.log(length + 1, 4)) if length > 0 else 0
    if n < 1 or 4**n - 1 != length:
        raise ValueError(f"a Pauli vector has 3 (1 qubit) or 15 (2 qubits) entries, got {length}")
    return n


def avg_from_pauli(pauli: Sequence[float]) -> float:
    return avg_from_process(math.fsum(pauli), pauli_arity(len(pauli)))


def to_avg_infidelity(kind: MetricKind, value: float | Sequence[float], num_qubits: int) -> float:
    if kind == "pauli":
        return avg_from_pauli(value)  # type: ignore[arg-type]
    if kind == "process_infidelity":
        return avg_from_process(float(value), num_qubits)  # type: ignore[arg-type]
    if kind == "depolarizing_param":
        return avg_from_depolarizing(float(value), num_qubits)  # type: ignore[arg-type]
    return float(value)  # type: ignore[arg-type]


def check_metric(kind: MetricKind, value: float | Sequence[float], num_qubits: int) -> None:
    """Raise ValueError when ``value`` is outside the physical range of ``kind``."""
    d = dim(num_qubits)
    if kind == "pauli":
        entries = list(value)  # type: ignore[arg-type]
        if len(entries) != d * d - 1:
            raise ValueError(
                f"pauli for a {num_qubits}-qubit gate needs {d * d - 1} entries, got {len(entries)}"
            )
        if any(p < 0 for p in entries):
            raise ValueError("pauli entries must be >= 0")
        if math.fsum(entries) > 1 + _TOL:
            raise ValueError(f"pauli entries sum to {math.fsum(entries)!r} > 1")
        return
    upper = {
        "avg_infidelity": max_avg_infidelity(num_qubits),
        "process_infidelity": 1.0,
        "depolarizing_param": max_depolarizing_param(num_qubits),
    }[kind]
    if not 0 <= value <= upper + _TOL:  # type: ignore[operator]
        raise ValueError(f"{kind} of a {num_qubits}-qubit gate must be in [0, {upper:.6g}]")


def pauli_labels(num_qubits: int) -> tuple[str, ...]:
    """Non-identity Pauli labels in the order Pauli vectors use."""
    return tuple("".join(p) for p in product("IXYZ", repeat=num_qubits))[1:]


PAULI_1Q = pauli_labels(1)
PAULI_2Q = pauli_labels(2)


def uniform_pauli(r: float, num_qubits: int) -> tuple[float, ...]:
    """Pauli vector of the depolarizing channel with average infidelity ``r``."""
    d = dim(num_qubits)
    return (process_from_avg(r, num_qubits) / (d * d - 1),) * (d * d - 1)


_SWAPPED_2Q = tuple(PAULI_2Q.index(label[::-1]) for label in PAULI_2Q)


def swap_pauli_2q(pauli: Sequence[float]) -> tuple[float, ...]:
    """Pauli vector of the same channel with its two operands exchanged."""
    if len(pauli) != len(PAULI_2Q):
        raise ValueError(f"expected 15 entries, got {len(pauli)}")
    return tuple(pauli[i] for i in _SWAPPED_2Q)
