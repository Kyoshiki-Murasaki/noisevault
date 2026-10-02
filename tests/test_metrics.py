from __future__ import annotations

import math

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


def _pauli_probabilities(superop: np.ndarray) -> np.ndarray:
    d = math.isqrt(superop.shape[0])
    labels = metrics.pauli_labels(d.bit_length() - 1)
    mats = [_pauli_matrix(label) for label in labels]
    return np.array([np.trace(np.kron(m.conj(), m).conj().T @ superop).real / d**2 for m in mats])


def _lindblad_channel(rates: dict[str, float]) -> tuple[float, ...]:
    d2 = 4 ** len(next(iter(rates)))
    superop = np.eye(d2, dtype=complex)
    for label, rate in rates.items():
        flip = (1 - math.exp(-2 * rate)) / 2
        m = _pauli_matrix(label)
        superop = superop @ ((1 - flip) * np.eye(d2) + flip * np.kron(m.conj(), m))
    return tuple(_pauli_probabilities(superop).tolist())


_LINDBLAD_2Q = _lindblad_channel(
    dict(zip(metrics.PAULI_2Q, np.random.default_rng(3).uniform(0, 0.004, 15), strict=True))
)
_IX_IZ_XI = tuple(0.1 if label in ("IX", "IZ", "XI") else 0.0 for label in metrics.PAULI_2Q)
_RULES = {
    "avg_infidelity 1q": (lambda r, s: metrics.scale_avg_infidelity(r, 1, s), 1e-3),
    "avg_infidelity 2q": (lambda r, s: metrics.scale_avg_infidelity(r, 2, s), 2e-2),
    "pauli flips": (metrics.scale_pauli, (0.0099, 0.0001, 0.0099)),
    "pauli dephasing": (metrics.scale_pauli, (0.0, 0.0, 0.1)),
    "pauli 2q": (metrics.scale_pauli, _LINDBLAD_2Q),
    "readout": (metrics.scale_readout, (0.02, 0.05)),
    "readout one zero": (metrics.scale_readout, (0.0, 0.0093)),
}
_FACTORS = (0.0, 0.05, 0.3, 1.0, 1.7, 4.0, 12.0, 50.0)


@pytest.mark.parametrize(("scale", "value"), _RULES.values(), ids=_RULES)
def test_scaling_twice_multiplies_the_factors(scale, value) -> None:
    for a in _FACTORS:
        for b in _FACTORS:
            np.testing.assert_allclose(
                scale(scale(value, a), b), scale(value, a * b), rtol=0, atol=1e-12
            )


@pytest.mark.parametrize(("scale", "value"), _RULES.values(), ids=_RULES)
def test_factor_one_keeps_the_value_and_factor_zero_removes_the_error(scale, value) -> None:
    np.testing.assert_allclose(scale(value, 1.0), value, rtol=0, atol=1e-15)
    assert np.all(np.asarray(scale(value, 0.0)) == 0)


def test_independent_flips_and_dephasing_scale_in_closed_form() -> None:
    np.testing.assert_allclose(
        metrics.scale_pauli([0.09, 0.01, 0.09], 2), [0.1476, 0.0324, 0.1476], rtol=0, atol=1e-12
    )
    dephased = metrics.scale_pauli([0.0, 0.0, 0.1], 10)
    np.testing.assert_allclose(dephased, [0.0, 0.0, (1 - 0.8**10) / 2], rtol=0, atol=1e-12)
    assert dephased[2] == pytest.approx(0.446313, abs=5e-7)


@pytest.mark.parametrize(
    "pauli",
    [
        metrics.uniform_pauli(0.05, 1),
        metrics.uniform_pauli(0.2, 2),
        (0.0, 0.0, 0.1),
        (0.09, 0.01, 0.09),
        _LINDBLAD_2Q,
        _lindblad_channel({"IX": 0.004, "IZ": 0.003, "XI": 0.002}),
        (0.0, 0.0, 0.0),
        (0.0,) * 15,
    ],
    ids=[
        "depolarizing 1q",
        "depolarizing 2q",
        "dephasing",
        "independent flips",
        "lindblad 2q",
        "sparse lindblad 2q",
        "no error 1q",
        "no error 2q",
    ],
)
def test_embeddable_channels_stay_channels_at_every_factor(pauli) -> None:
    assert metrics.pauli_embeddable(pauli)
    n = metrics.pauli_arity(len(pauli))
    for factor in (0.0, *np.geomspace(1e-3, 1e3, 25)):
        metrics.check_metric("pauli", metrics.scale_pauli(pauli, float(factor)), n)


@pytest.mark.parametrize("labels", [metrics.PAULI_1Q, metrics.PAULI_2Q], ids=["1q", "2q"])
def test_pauli_rates_recover_the_generators_of_a_composed_channel(labels) -> None:
    rates = np.random.default_rng(11).uniform(0, 0.05, len(labels))
    rates[0] = 0.0
    pauli = _lindblad_channel(dict(zip(labels, rates, strict=True)))
    np.testing.assert_allclose(metrics.pauli_rates(pauli), rates, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("pauli", "min_rate"),
    [(_IX_IZ_XI, -0.0228), ((0.1, 0.0, 0.1), -0.0161), ((0.05, 0.2, 0.2), -0.0558)],
    ids=["IX IZ XI", "X and Z", "Y and Z heavy"],
)
def test_non_embeddable_channels_come_back_unchanged(pauli, min_rate) -> None:
    assert min(metrics.pauli_rates(pauli)) == pytest.approx(min_rate, abs=1e-4)
    assert not metrics.pauli_embeddable(pauli)
    for factor in (0.0, 0.5, 1.5, 2.0, 1e3):
        assert metrics.scale_pauli(pauli, factor) == pauli


@pytest.mark.parametrize(
    "pauli",
    [(0.25, 0.25, 0.25), (1 / 16,) * 15, (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)],
    ids=["full 1q", "full 2q", "past full", "certain Z"],
)
def test_pauli_rates_is_none_at_and_past_full_depolarization(pauli) -> None:
    assert metrics.pauli_rates(pauli) is None
    assert not metrics.pauli_embeddable(pauli)
    for factor in (0.0, 0.5, 2.0):
        assert metrics.scale_pauli(pauli, factor) == pauli


@pytest.mark.parametrize(("r", "n"), [(0.5, 1), (2 / 3, 1), (0.75, 2), (0.8, 2)])
def test_avg_infidelity_at_or_past_full_depolarization_is_unchanged(r: float, n: int) -> None:
    for factor in (0.0, 0.5, 2.0):
        assert metrics.scale_avg_infidelity(r, n, factor) == r


@pytest.mark.parametrize(("n", "r"), [(1, 1e-5), (1, 1e-3), (2, 1e-4), (2, 1e-2), (2, 0.3)])
def test_depolarizing_avg_infidelity_scales_like_its_pauli_vector(n: int, r: float) -> None:
    for factor in (0.01, 0.5, 1.84, 50.0):
        uniform = metrics.uniform_pauli(r, n)
        via_pauli = metrics.avg_from_pauli(metrics.scale_pauli(uniform, factor))
        assert metrics.scale_avg_infidelity(r, n, factor) == pytest.approx(via_pauli, rel=1e-13)


def test_scaled_avg_infidelity_at_factor_1_84() -> None:
    assert metrics.scale_avg_infidelity(0.001, 1, 1.84) == pytest.approx(0.001838454235, abs=1e-12)
    assert metrics.scale_avg_infidelity(0.01, 2, 1.84) == pytest.approx(0.01829688644, abs=1e-11)


@pytest.mark.parametrize("pair", [(0.02, 0.05), (0.0, 0.0093)])
@pytest.mark.parametrize("power", [2, 3, 7])
def test_readout_scales_as_a_power_of_its_confusion_matrix(pair, power: int) -> None:
    a, b = pair
    m = np.linalg.matrix_power(np.array([[1 - a, b], [a, 1 - b]]), power)
    np.testing.assert_allclose(
        metrics.scale_readout(pair, power), (m[1, 0], m[0, 1]), rtol=0, atol=1e-15
    )


def test_identity_readout_stays_the_identity() -> None:
    for factor in (0.0, 0.5, 1.0, 2.0, 1e3):
        assert metrics.scale_readout((0.0, 0.0), factor) == (0.0, 0.0)


@pytest.mark.parametrize("pair", [(0.5, 0.5), (0.6, 0.7), (1.0, 0.0)])
def test_readout_no_better_than_chance_is_unchanged(pair) -> None:
    for factor in (0.0, 0.5, 2.0):
        assert metrics.scale_readout(pair, factor) == pair
