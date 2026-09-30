from __future__ import annotations

import numpy as np
import pytest
from conftest import require

from noisevault.gates import GATES

_DEFAULT_PARAMS = (0.37, -1.1, 0.8)


def _unitary(row) -> np.ndarray:
    return row.unitary(*_DEFAULT_PARAMS[: len(row.params)])


def _reverse_qubits(u: np.ndarray, n: int) -> np.ndarray:
    axes = list(range(n))[::-1]
    return u.reshape([2] * 2 * n).transpose(axes + [n + a for a in axes]).reshape(2**n, 2**n)


@pytest.mark.parametrize("row", [r for r in GATES.values() if r.unitary], ids=lambda r: r.name)
def test_unitaries_are_unitary(row) -> None:
    u = _unitary(row)
    assert u.shape == (2**row.arity, 2**row.arity)
    assert np.allclose(u.conj().T @ u, np.eye(len(u)), atol=1e-12)


@pytest.mark.parametrize("row", [r for r in GATES.values() if r.qiskit], ids=lambda r: r.name)
def test_qiskit_column_names_the_same_gate(row) -> None:
    require("qiskit")
    from qiskit.circuit.library import get_standard_gate_name_mapping
    from qiskit.quantum_info import Operator

    gate = get_standard_gate_name_mapping()[row.qiskit]
    assert gate.base_class.__name__ == row.qiskit_class
    if row.unitary is None:
        return
    params = _DEFAULT_PARAMS[: len(row.params)]
    qiskit_matrix = Operator(gate.base_class(*params)).data  # little-endian
    assert np.allclose(_reverse_qubits(qiskit_matrix, row.arity), _unitary(row), atol=1e-12)


@pytest.mark.parametrize("row", [r for r in GATES.values() if r.stim], ids=lambda r: r.name)
def test_stim_column_names_the_same_gate_up_to_phase(row) -> None:
    stim = require("stim")
    if row.unitary is None:
        for name in row.stim:
            data = stim.gate_data(name)
            assert data.is_reset or data.produces_measurements, name
        return
    ours = _unitary(row)
    for name in row.stim:
        theirs = stim.gate_data(name).tableau.to_unitary_matrix(endian="big")
        overlap = np.trace(theirs.conj().T @ ours) / len(ours)
        assert abs(abs(overlap) - 1) < 1e-6, name  # stim matrices are single precision


@pytest.mark.parametrize("row", [r for r in GATES.values() if r.pennylane], ids=lambda r: r.name)
def test_pennylane_column_names_an_operation(row) -> None:
    qml = require("pennylane")
    name = row.pennylane.removeprefix("Adjoint(").removesuffix(")")
    assert hasattr(qml, name)


@pytest.mark.parametrize("row", [r for r in GATES.values() if r.cirq], ids=lambda r: r.name)
def test_cirq_column_names_a_gate_class(row) -> None:
    cirq = require("cirq")
    assert isinstance(getattr(cirq, row.cirq), type)


def test_direction_defaults() -> None:
    symmetric = {n for n, r in GATES.items() if r.symmetric}
    assert {"cz", "rzz", "rxx", "ryy", "zz", "ms", "iswap", "swap", "sqrt_iswap"} == symmetric
    assert {n for n, r in GATES.items() if r.multi_entangler} == {"swap", "cswap", "ccx"}
