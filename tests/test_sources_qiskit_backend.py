from __future__ import annotations

from itertools import permutations

from conftest import require

import noisevault as nv

require("qiskit")

from qiskit.circuit import Measure  # noqa: E402
from qiskit.circuit.library import (  # noqa: E402
    CXGate,
    CZGate,
    ECRGate,
    SwapGate,
    SXGate,
    XGate,
    YGate,
)
from qiskit.providers import BackendV2, Options  # noqa: E402
from qiskit.transpiler import InstructionProperties, Target  # noqa: E402


class _Backend(BackendV2):
    def __init__(self, target: Target, name: str = "toy") -> None:
        super().__init__(name=name)
        self._target = target

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    def run(self, run_input, **options):
        raise NotImplementedError


def _props(error: float, duration: float) -> InstructionProperties:
    return InstructionProperties(error=error, duration=duration)


def _target(num_qubits: int, *gates: tuple) -> Target:
    target = Target(num_qubits=num_qubits)
    for gate, loci in gates:
        target.add_instruction(gate, loci)
    target.add_instruction(Measure(), {(q,): _props(0.02, 1e-6) for q in range(num_qubits)})
    return target


def _assert_allows_exactly_what_the_target_allows(target: Target, profile: nv.Profile) -> None:
    table = profile.table
    for name in target.operation_names:
        if name == "measure":
            continue
        arity = target.operation_from_name(name).num_qubits
        for qubits in permutations(range(target.num_qubits), arity):
            supported = target.instruction_supported(name, qubits)
            if arity == 2 and table.symmetric(name):
                supported = supported or target.instruction_supported(name, qubits[::-1])
            assert table.allowed(name, qubits) == supported, (name, qubits)


def test_a_one_qubit_gate_on_some_qubits_is_disabled_on_the_others() -> None:
    target = _target(
        3,
        (SXGate(), {(0,): _props(0.001, 35e-9)}),
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(3)}),
        (CZGate(), {(0, 1): _props(0.01, 60e-9), (1, 2): _props(0.01, 60e-9)}),
    )
    assert not target.instruction_supported("sx", (1,))
    profile = nv.from_qiskit_backend(_Backend(target))
    assert profile.table.gate("sx", (0,)).avg_infidelity == 0.001
    assert profile.table.gate("sx", (1,)).state == "disabled"
    assert profile.table.gate("sx", (2,)).state == "disabled"
    _assert_allows_exactly_what_the_target_allows(target, profile)
    require("qiskit_aer")
    exported = profile.to_qiskit().target
    assert set(exported["sx"]) == {(0,)}


def test_a_directed_gate_in_one_direction_of_a_pair_is_disabled_in_the_other() -> None:
    cz = {(a, b): _props(0.01, 60e-9) for a, b in ((0, 1), (1, 0), (1, 2), (2, 1))}
    target = _target(
        3,
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(3)}),
        (CXGate(), {(0, 1): _props(0.02, 300e-9)}),
        (CZGate(), cz),
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert profile.table.gate("cx", (0, 1)).avg_infidelity == 0.02
    assert profile.table.gate("cx", (1, 0)).state == "disabled"
    assert profile.table.gate("cx", (1, 2)).state == "disabled"
    _assert_allows_exactly_what_the_target_allows(target, profile)
    require("qiskit_aer")
    assert set(profile.to_qiskit().target["cx"]) == {(0, 1)}


def test_a_symmetric_gate_missing_from_a_pair_is_disabled_on_it() -> None:
    target = _target(
        3,
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(3)}),
        (ECRGate(), {(0, 1): _props(0.02, 500e-9), (2, 1): _props(0.02, 500e-9)}),
        (CZGate(), {(1, 2): _props(0.01, 60e-9)}),
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert profile.table.gate("cz", (0, 1)).state == "disabled"
    assert profile.table.gate("cz", (2, 1)).avg_infidelity == 0.01
    assert profile.table.gate("ecr", (1, 2)).state == "disabled"
    _assert_allows_exactly_what_the_target_allows(target, profile)


def test_a_gate_missing_from_an_undirected_pair_is_disabled_on_it() -> None:
    target = _target(
        3,
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(3)}),
        (CZGate(), {(0, 1): _props(0.01, 60e-9), (1, 2): _props(0.01, 60e-9)}),
        (SwapGate(), {(1, 0): _props(0.03, 180e-9)}),
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert not profile.connectivity.directed
    assert profile.table.gate("swap", (0, 1)).avg_infidelity == 0.03
    assert profile.table.gate("swap", (2, 1)).state == "disabled"
    _assert_allows_exactly_what_the_target_allows(target, profile)


def test_a_gate_with_no_locus_is_not_in_the_profile() -> None:
    target = _target(
        2,
        (XGate(), {(q,): _props(0.002, 35e-9) for q in range(2)}),
        (YGate(), {}),
        (CZGate(), {(0, 1): _props(0.01, 60e-9)}),
    )
    profile = nv.from_qiskit_backend(_Backend(target))
    assert "y" not in profile.gates
    _assert_allows_exactly_what_the_target_allows(target, profile)


def test_an_ionq_backend_is_trapped_ion() -> None:
    class IonQQPUBackend(_Backend):
        pass

    IonQQPUBackend.__module__ = "qiskit_ionq.ionq_backend"
    target = _target(
        2,
        (XGate(), {(q,): _props(0.0002, 130e-6) for q in range(2)}),
        (CXGate(), {(0, 1): _props(0.005, 970e-6), (1, 0): _props(0.005, 970e-6)}),
    )
    for name in ("ionq_qpu", "ionq_qpu.forte-1"):
        profile = nv.from_qiskit_backend(IonQQPUBackend(target, name=name))
        assert profile.device.technology == "trapped_ion", name
    assert nv.from_qiskit_backend(_Backend(target, name="ionq_qpu")).device.technology == (
        "trapped_ion"
    )


def test_a_backend_that_names_no_known_provider_is_other_technology() -> None:
    from qiskit.providers.fake_provider import GenericBackendV2

    profile = nv.from_qiskit_backend(GenericBackendV2(num_qubits=3, seed=7))
    assert profile.device.technology == "other"
