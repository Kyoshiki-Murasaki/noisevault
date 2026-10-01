from __future__ import annotations

import time
import tracemalloc

import pytest
from conftest import toy

import noisevault as nv
from noisevault import metrics
from noisevault.profile import Profile
from noisevault.table import GateNoise, Unavailable

ECR = {"qubits": 2, "avg_infidelity": 7e-3, "duration_ns": 400}


def table_of(**sections):
    return Profile.model_validate(toy(**sections)).table


def test_record_overrides_definition_field_by_field() -> None:
    table = table_of(calibrations=[{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02}])
    found = table.gate("cz", (0, 1))
    assert (found.state, found.origin, found.avg_infidelity) == ("calibrated", "record", 0.02)
    assert found.duration_ns == 70  # inherited from the definition
    default = table.gate("cz", (1, 2))
    assert (default.origin, default.avg_infidelity) == ("default", 1e-2)


def test_metric_group_is_replaced_whole() -> None:
    pauli = [1e-3] * 15
    table = table_of(calibrations=[{"gate": "cz", "qubits": [0, 1], "pauli": pauli}])
    found = table.gate("cz", (0, 1))
    assert found.spec.avg_infidelity is None
    assert found.pauli == tuple(pauli)
    assert found.avg_infidelity == pytest.approx(metrics.avg_from_pauli(pauli))


def test_symmetric_gate_uses_reversed_record_with_permuted_pauli() -> None:
    pauli = [0.0] * 15
    pauli[metrics.PAULI_2Q.index("IX")] = 3e-3  # X on the second operand, qubit 1
    table = table_of(calibrations=[{"gate": "cz", "qubits": [0, 1], "pauli": pauli}])
    found = table.gate("cz", (1, 0))
    assert found.origin == "reversed_record"
    assert found.pauli[metrics.PAULI_2Q.index("XI")] == 3e-3  # still on qubit 1, now first
    assert found.pauli[metrics.PAULI_2Q.index("IX")] == 0.0
    assert found.spec.pauli == found.pauli


def test_directed_gate_is_not_reversed() -> None:
    gates = {**toy()["gates"], "ecr": ECR}
    table = table_of(
        gates=gates,
        connectivity={"edges": [[1, 0], [1, 2]], "directed": True},
        calibrations=[{"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 6e-3}],
    )
    assert table.gate("ecr", (1, 0)).avg_infidelity == 6e-3
    assert isinstance(table.gate("ecr", (0, 1)), Unavailable)
    assert table.gate("ecr", (1, 2)).origin == "default"
    assert isinstance(table.gate("ecr", (2, 1)), Unavailable)


def test_connectivity_decides_where_defaults_apply() -> None:
    undirected = table_of()
    assert undirected.gate("cz", (1, 0)).origin == "default"
    assert isinstance(undirected.gate("cz", (0, 2)), Unavailable)
    everywhere = table_of(connectivity="all_to_all")
    assert everywhere.gate("cz", (0, 2)).origin == "default"


def test_disabled_record_terminates_lookup() -> None:
    table = table_of(
        calibrations=[
            {"gate": "cz", "qubits": [0, 1], "disabled": True},
            {"gate": "cz", "qubits": [1, 0], "avg_infidelity": 1e-2},
        ]
    )
    assert table.gate("cz", (0, 1)).state == "disabled"
    assert not table.allowed("cz", (0, 1))
    assert table.allowed("cz", (1, 0))


def test_gate_states() -> None:
    gates = {**toy()["gates"], "reset": {"duration_ns": 1000}}
    table = table_of(gates=gates)
    assert table.gate("rz", (0,)).state == "ideal"
    assert table.gate("sx", (0,)).state == "calibrated"
    assert table.gate("reset", (0,)).state == "uncalibrated"
    assert isinstance(table.gate("h", (0,)), Unavailable)


def test_z_family_is_free_only_through_a_virtual_rz() -> None:
    table = table_of()
    assert table.gate("s", (1,)).state == "ideal"
    assert table.gate("t", (1,)).state == "ideal"
    assert table.gate("u1", (1,)).state == "ideal"
    calibrated_rz = table_of(
        calibrations=[{"gate": "rz", "qubits": [1], "virtual": False, "avg_infidelity": 1e-4}]
    )
    assert calibrated_rz.gate("s", (0,)).state == "ideal"
    assert isinstance(calibrated_rz.gate("s", (1,)), Unavailable)


def test_defined_z_family_gates_without_their_own_metric_are_free_through_a_virtual_rz() -> None:
    table = table_of(
        gates={
            "rz": {"virtual": True},
            "z": {},
            "s": {"disabled": True},
            "t": {"avg_infidelity": 1e-4},
            "x": {"avg_infidelity": 0.01},
        },
        calibrations=[{"gate": "z", "qubits": [2], "avg_infidelity": 2e-3}],
    )
    free = table.gate("z", (0,))
    assert (free.state, free.avg_infidelity) == ("ideal", None)
    calibrated = table.gate("z", (2,))
    assert (calibrated.state, calibrated.avg_infidelity) == ("calibrated", 2e-3)
    assert table.gate("s", (0,)).state == "disabled"
    assert table.gate("t", (0,)).avg_infidelity == 1e-4


def test_rz_lookup_without_an_rz_definition_is_undefined() -> None:
    found = table_of(gates={"sx": {"avg_infidelity": 1e-3}}).gate("rz", (0,))
    assert isinstance(found, Unavailable) and found.kind == "undefined"


def test_bad_targets_and_disabled_qubits_are_unavailable() -> None:
    table = table_of(qubits=[{"index": 2, "disabled": True}])
    assert "acts on 2" in table.gate("cz", (0,)).reason
    assert "distinct" in table.gate("cz", (1, 1)).reason
    assert "outside" in table.gate("sx", (5,)).reason
    assert "disabled" in table.gate("cz", (1, 2)).reason
    assert table.qubit(2).disabled


def test_unavailable_says_why_as_data() -> None:
    table = table_of(qubits=[{"index": 2, "disabled": True}])
    kinds = {
        ("h", (0,)): "undefined",
        ("cz", (0, 2)): "qubit_disabled",
        ("cz", (0,)): "bad_target",
        ("cz", (1, 1)): "bad_target",
        ("sx", (5,)): "bad_target",
    }
    for (name, qubits), kind in kinds.items():
        assert table.gate(name, qubits).kind == kind, (name, qubits)
    assert table_of().gate("cz", (0, 2)).kind == "not_connected"
    assert table_of(gates={"rz": {"virtual": True}}).typical(1, (0,)).kind == "no_native"


def test_qubit_merge_rules() -> None:
    table = table_of(
        idle={"t1_us": 100, "t2_us": 80},
        readout={"p1_given_0": 0.01, "p0_given_1": 0.03},
        prep={"error": 1e-3},
        qubits=[{"index": 1, "t1_us": 40, "readout": {"error": 0.05}}],
    )
    q0, q1 = table.qubit(0), table.qubit(1)
    assert (q0.t1_ns, q0.t2_ns, q0.readout, q0.prep_error) == (1e5, 8e4, (0.01, 0.03), 1e-3)
    assert (q1.t1_ns, q1.t2_ns) == (4e4, 8e4)  # t1 overridden, t2 inherited
    assert q1.readout == (0.05, 0.05)  # readout replaced as a whole
    unknown = table_of().qubit(0)
    assert unknown.readout is None and unknown.prep_error is None and unknown.t1_ns is None


def test_dephasing_rate_merges_like_the_other_idle_fields() -> None:
    table = table_of(
        idle={"t1_us": 100, "dephasing_rate_per_s": 0.3},
        qubits=[{"index": 1, "dephasing_rate_per_s": 2.0}, {"index": 2, "t1_us": 50}],
    )
    assert [table.qubit(q).dephasing_rate_per_s for q in range(3)] == [0.3, 2.0, 0.3]
    assert table_of().qubit(0).dephasing_rate_per_s is None


def test_typical_prefers_most_records_then_name() -> None:
    gates = {**toy()["gates"], "x": {"avg_infidelity": 5e-4}, "id": {"avg_infidelity": 1e-4}}
    record = {"gate": "x", "qubits": [0], "avg_infidelity": 2e-4}
    table = table_of(gates=gates, calibrations=[record])
    assert table.typical(1, (1,)).gate == "x"  # most records
    tied = table_of(gates=gates)
    assert tied.typical(1, (1,)).gate == "sx"  # tie: alphabetical, after any real gate
    assert tied.typical(2, (0, 1)).gate == "cz"


def test_typical_takes_the_identity_only_when_no_real_gate_fits() -> None:
    gates = {**toy()["gates"], "id": {"avg_infidelity": 1e-4}}
    records = [{"gate": "id", "qubits": [q], "avg_infidelity": 1e-4} for q in range(3)]
    table = table_of(gates=gates, calibrations=records)
    assert table.typical(1, (0,)).gate == "sx"  # id has more records, but is an idle slot
    only_id = table_of(gates={**gates, "sx": {"disabled": True}}, calibrations=records)
    assert only_id.typical(1, (0,)).gate == "id"
    for q in range(3):
        assert nv.load("ibm_fez").table.typical(1, (q,)).gate == "sx"


def test_typical_skips_disabled_candidates_and_can_reverse_a_directed_record() -> None:
    gates = {**toy()["gates"], "ecr": ECR}
    records = [
        {"gate": "cz", "qubits": [0, 1], "disabled": True},
        {"gate": "cz", "qubits": [1, 0], "disabled": True},
        {"gate": "cz", "qubits": [2, 1], "avg_infidelity": 1e-2},
        {"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 6e-3},
        {"gate": "ecr", "qubits": [2, 1], "avg_infidelity": 8e-3},
    ]
    table = table_of(
        gates=gates,
        connectivity={"edges": [[1, 0], [2, 1]], "directed": True},
        calibrations=records,
    )
    chosen = table.typical(2, (0, 1))  # cz has most records but is disabled here
    assert (chosen.gate, chosen.origin, chosen.avg_infidelity) == ("ecr", "reversed_record", 6e-3)
    other = table.typical(2, (1, 2))
    assert (other.gate, other.origin) == ("cz", "reversed_record")


def test_typical_ignores_uncalibrated_and_virtual_gates() -> None:
    table = table_of(gates={"rz": {"virtual": True}, "sx": {}, "cz": {"avg_infidelity": 1e-2}})
    assert isinstance(table.typical(1, (0,)), Unavailable)
    assert table.natives(1) == ("sx",)


def test_all_to_all_is_never_expanded() -> None:
    profile = Profile.uniform(
        "big",
        technology="neutral_atom",
        num_qubits=2_000,
        one_qubit_error=1e-3,
        two_qubit_error=5e-3,
    )
    tracemalloc.start()
    start = time.perf_counter()
    table = profile.table
    assert table.gate("cz", (0, 1_999)).avg_infidelity == 5e-3
    assert table.typical(2, (123, 1567)).state == "calibrated"
    assert table.allowed("cz", (1_998, 3))
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 5_000_000 and elapsed < 1.0
    assert isinstance(table.gate("cz", (0, 1)), GateNoise)


def test_disabled_gates_carry_no_error_number() -> None:
    table = table_of(calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}])
    found = table.gate("cz", (0, 1))
    assert found.state == "disabled" and found.avg_infidelity is None and found.pauli is None


def test_typical_does_not_reverse_a_gate_disabled_on_the_pair() -> None:
    gates = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "ecr": ECR}
    table = table_of(
        gates=gates,
        connectivity={"edges": [[0, 1], [1, 0]], "directed": True},
        calibrations=[
            {"gate": "ecr", "qubits": [0, 1], "disabled": True},
            {"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 6e-3},
        ],
    )
    assert isinstance(table.typical(2, (0, 1)), Unavailable)
