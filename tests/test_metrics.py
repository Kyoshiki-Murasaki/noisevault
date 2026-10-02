from __future__ import annotations

import numpy as np
import pytest
from conftest import require

from noisevault import metrics

_P = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def _pauli_matrix(label: str) -> np.ndarray:
    out = np.eye(1)
    for c in label:
        out = np.kron(out, _P[c])
    return out


def _pauli_superop(pauli: list[float]) -> np.ndarray:
    n = metrics.pauli_arity(len(pauli))
    labels = ["I" * n, *metrics.pauli_labels(n)]
    probs = [1 - sum(pauli), *pauli]
    mats = [_pauli_matrix(label) for label in labels]
    return sum(p * np.kron(m.conj(), m) for p, m in zip(probs, mats, strict=True))


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_average_and_process_infidelity_convert_like_qiskit(n: int, seed: int) -> None:
    qi = require("qiskit.quantum_info")
    d = 2**n
    raw = qi.SuperOp(qi.random_quantum_channel(d, rank=2, seed=seed)).data
    channel = qi.SuperOp(0.96 * np.eye(d * d) + 0.04 * raw)
    r = 1 - qi.average_gate_fidelity(channel)
    e = 1 - qi.process_fidelity(channel)
    assert metrics.process_from_avg(r, n) == pytest.approx(e, abs=1e-12)
    assert metrics.avg_from_process(e, n) == pytest.approx(r, abs=1e-12)
    assert metrics.to_avg_infidelity("process_infidelity", e, n) == pytest.approx(r, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("lam", [1e-3, 0.02, 0.5])
def test_depolarizing_param_matches_qiskit_aer(n: int, lam: float) -> None:
    noise = require("qiskit_aer.noise")
    qi = require("qiskit.quantum_info")
    r = 1 - qi.average_gate_fidelity(noise.depolarizing_error(lam, n))
    assert metrics.avg_from_depolarizing(lam, n) == pytest.approx(r, abs=1e-12)
    assert metrics.depolarizing_from_avg(r, n) == pytest.approx(lam, abs=1e-12)


@pytest.mark.parametrize("length", [3, 15])
def test_pauli_vector_average_infidelity_matches_qiskit(length: int) -> None:
    qi = require("qiskit.quantum_info")
    pauli = list(np.random.default_rng(length).uniform(0, 0.01, length))
    expected = 1 - qi.average_gate_fidelity(qi.SuperOp(_pauli_superop(pauli)))
    assert metrics.avg_from_pauli(pauli) == pytest.approx(expected, abs=1e-12)
    assert metrics.to_avg_infidelity("pauli", pauli, metrics.pauli_arity(length)) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize(
    ("kind", "value", "n"),
    [
        ("avg_infidelity", 2 / 3 + 1e-6, 1),
        ("avg_infidelity", 0.8 + 1e-6, 2),
        ("avg_infidelity", -1e-9, 1),
        ("process_infidelity", 1.01, 1),
        ("depolarizing_param", 4 / 3 + 1e-6, 1),
        ("pauli", [0.1, -0.01, 0.0], 1),
        ("pauli", [0.5, 0.4, 0.2], 1),
        ("pauli", [0.01] * 15, 1),
        ("pauli", [0.01] * 3, 2),
    ],
)
def test_check_metric_rejects_unphysical_values(kind, value, n) -> None:
    with pytest.raises(ValueError):
        metrics.check_metric(kind, value, n)


def test_an_out_of_range_metric_error_names_the_value() -> None:
    message = r"avg_infidelity of a 2-qubit gate must be in \[0, 0\.8\], got 0\.93$"
    with pytest.raises(ValueError, match=message):
        metrics.check_metric("avg_infidelity", 0.93, 2)


@pytest.mark.parametrize(
    ("kind", "value", "n"),
    [
        ("avg_infidelity", 2 / 3, 1),
        ("avg_infidelity", 0.8, 2),
        ("process_infidelity", 1.0, 2),
        ("depolarizing_param", 16 / 15, 2),
        ("pauli", [1 / 15] * 15, 2),
    ],
)
def test_check_metric_accepts_the_bounds(kind, value, n) -> None:
    metrics.check_metric(kind, value, n)


def test_pauli_label_order_is_stim_order() -> None:
    assert metrics.PAULI_1Q == ("X", "Y", "Z")
    assert metrics.PAULI_2Q[:4] == ("IX", "IY", "IZ", "XI")
    assert metrics.PAULI_2Q[-1] == "ZZ" and len(metrics.PAULI_2Q) == 15


def test_swap_pauli_2q_is_the_channel_with_operands_exchanged() -> None:
    pauli = list(np.random.default_rng(7).uniform(0, 0.01, 15))
    swapped = metrics.swap_pauli_2q(pauli)
    swap = np.eye(4)[[0, 2, 1, 3]]
    conj = np.kron(swap, swap)  # superoperator of conjugation by SWAP (real, self-inverse)
    assert np.allclose(_pauli_superop(list(swapped)), conj @ _pauli_superop(pauli) @ conj)
    assert swapped[metrics.PAULI_2Q.index("XI")] == pauli[metrics.PAULI_2Q.index("IX")]
    assert metrics.swap_pauli_2q(swapped) == tuple(pauli)


def test_uniform_pauli_has_the_stated_average_infidelity() -> None:
    for n, r in ((1, 1e-3), (2, 1e-2)):
        assert metrics.avg_from_pauli(metrics.uniform_pauli(r, n)) == pytest.approx(r, rel=1e-12)
