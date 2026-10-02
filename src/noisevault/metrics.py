"""Error metrics: conversions, bounds, Pauli-vector ordering and scaling, all in one place.

For an n-qubit operation with d = 2**n:
  r      average gate infidelity, 1 - F_avg
  e_pro  process (entanglement) infidelity, r (d + 1) / d
  lambda Qiskit depolarizing parameter, (1 - lambda) rho + lambda I/d, so r = lambda (d - 1) / d
  pauli  non-identity Pauli probabilities; their sum is e_pro

Pauli vectors list labels in lexicographic I, X, Y, Z order without the identity. For two
qubits this is Stim's PAULI_CHANNEL_2 order IX, IY, IZ, XI, XX, ..., ZZ, and the first letter
acts on ``qubits[0]``.

The scale_* rules raise a channel to a real power, so scaling twice multiplies the factors:
scale(scale(x, a), b) equals scale(x, a * b). Factor 1 keeps x, and factor 0 removes the error.
A value with no power at some factor >= 0 comes back unchanged at every factor. No rule raises.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache
from itertools import product
from typing import Literal

import numpy as np

MetricKind = Literal["avg_infidelity", "process_infidelity", "depolarizing_param", "pauli"]
METRIC_KEYS: tuple[MetricKind, ...] = (
    "avg_infidelity",
    "process_infidelity",
    "depolarizing_param",
    "pauli",
)
_TOL = 1e-12
_RATE_TOL = 1e-12


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
        raise ValueError(
            f"{kind} of a {num_qubits}-qubit gate must be in [0, {upper:.6g}], got {value!r}"
        )


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


def _anticommute(p: str, q: str) -> bool:
    return sum("I" not in (x, y) and x != y for x, y in zip(p, q, strict=True)) % 2 == 1


@cache
def _anticommutation(num_qubits: int) -> np.ndarray:
    labels = pauli_labels(num_qubits)
    return np.array([[_anticommute(p, q) for q in labels] for p in labels], dtype=float)


def scale_avg_infidelity(r: float, num_qubits: int, factor: float) -> float:
    """Average infidelity of the depolarizing channel with infidelity ``r``, raised to ``factor``.

    The channel's Pauli fidelity 1 - lambda becomes (1 - lambda)**factor, so the result equals
    the average infidelity of scale_pauli on ``uniform_pauli(r, num_qubits)``. At or past full
    depolarization the fidelity is <= 0 and has no real power, so ``r`` comes back unchanged.
    """
    lam = depolarizing_from_avg(r, num_qubits)
    if lam >= 1:
        return r
    return avg_from_depolarizing(-math.expm1(factor * math.log1p(-lam)), num_qubits)


def pauli_rates(pauli: Sequence[float]) -> tuple[float, ...] | None:
    """Pauli-Lindblad rates of a Pauli channel, or None when a Pauli fidelity is <= 0.

    A channel equals exp(sum_k lam_k (P_k rho P_k - rho)) exactly when its fidelities satisfy
    log f_a = -2 sum_k A[a, k] lam_k. A[a, k] is 1 when P_a and P_k anticommute and 0 otherwise.
    The fidelities are f_a = 1 - 2 sum_k A[a, k] p_k.
    """
    anti = _anticommutation(pauli_arity(len(pauli)))
    deficit = 2 * anti @ np.asarray(pauli, dtype=float)
    if deficit.max() >= 1:
        return None
    return tuple(np.linalg.solve(anti, -np.log1p(-deficit) / 2).tolist())


def pauli_embeddable(pauli: Sequence[float]) -> bool:
    """True when the Pauli channel has Pauli-Lindblad rates and every rate is >= -1e-12.

    An embeddable channel has a power that is a channel at every factor >= 0, so scale_pauli
    scales it. The channel with no error is embeddable. A channel at or past full
    depolarization has no rates and is not.
    """
    rates = pauli_rates(pauli)
    return rates is not None and min(rates) >= -_RATE_TOL


def scale_pauli(pauli: Sequence[float], factor: float) -> tuple[float, ...]:
    """The Pauli channel raised to ``factor``, which turns every Pauli fidelity f into f**factor.

    Only a channel that pauli_embeddable accepts scales. Its power has the Pauli-Lindblad rates
    times ``factor``, so the power is a channel at every factor >= 0. Any other channel has a
    power that is not a channel, or no real power at all, and comes back unchanged at every
    factor. With A from pauli_rates, the scaled probabilities are
    p_k = sum_a (2 A[k, a] - 1) (1 - f_a**factor) / 4**n.
    """
    if not pauli_embeddable(pauli):
        return tuple(pauli)
    anti = _anticommutation(pauli_arity(len(pauli)))
    log_fidelities = np.log1p(-2 * anti @ np.asarray(pauli, dtype=float))
    deficit = -np.expm1(factor * log_fidelities)
    scaled = (2 * anti - 1) @ deficit / (len(pauli) + 1)
    return tuple(np.maximum(scaled, 0.0).tolist())


def scale_readout(pair: tuple[float, float], factor: float) -> tuple[float, float]:
    """(P(1|0), P(0|1)) of M**factor, for the confusion matrix M = [[1 - a, b], [a, 1 - b]].

    M has eigenvalues 1 and 1 - a - b. Its power M**s keeps the ratio a : b and has
    a + b = 1 - (1 - a - b)**s. The identity (0, 0) and a pair no better than chance
    (a + b >= 1) come back unchanged.
    """
    a, b = pair
    total = a + b
    if total == 0 or total >= 1:
        return pair
    k = -math.expm1(factor * math.log1p(-total)) / total
    return (a * k, b * k)
