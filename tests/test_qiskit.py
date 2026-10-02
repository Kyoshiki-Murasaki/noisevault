"""The Qiskit export: Target, per-locus channels, readout, reset, delays, validation, report."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pytest
from conftest import MANILA_V01, migrated, require, toy

import noisevault as nv
from noisevault.channels import ChannelSpec, gate_channels, superoperator
from noisevault.conversion import resolve_op
from noisevault.errors import (
    NoiseApproximationWarning,
    UnsupportedEffect,
)
from noisevault.profile import Profile
from noisevault.reference import Op
from noisevault.reference import probabilities as reference
from noisevault.report import Report

require("qiskit_aer")

from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.circuit.library import iSwapGate  # noqa: E402
from qiskit.quantum_info import SuperOp  # noqa: E402
from qiskit.transpiler import PassManager  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise.passes import RelaxationNoisePass  # noqa: E402

from noisevault import gates  # noqa: E402
from noisevault.frameworks.qiskit import (  # noqa: E402
    CircuitNotNativeError,
    NoiseVaultSimulator,
    SqrtISwapGate,
    UnsupportedDevice,
    gate_error,
    to_qiskit,
)


def quiet_export(profile: Profile, **options: Any) -> NoiseVaultSimulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        return to_qiskit(profile, **options)


def ops_of(circuit: QuantumCircuit, layout: list[int]) -> list[Op]:
    """A transpiled circuit's gates as reference ops on circuit qubits ``0..len(layout)-1``."""
    position = {q: i for i, q in enumerate(layout)}
    return [
        Op(
            i.operation.name,
            tuple(position[circuit.find_bit(q).index] for q in i.qubits),
            tuple(float(p) for p in i.operation.params),
        )
        for i in circuit.data
        if i.operation.name != "barrier"
    ]


def aer_probabilities(sim: AerSimulator, circuit: QuantumCircuit, layout: list[int]) -> np.ndarray:
    """Exact pre-measurement probabilities, big-endian over ``layout`` like the reference."""
    circuit = circuit.copy()
    circuit.save_probabilities(layout)
    little = np.asarray(sim.run(circuit).result().data()["probabilities"])
    n = len(layout)
    return little.reshape((2,) * n).transpose(range(n - 1, -1, -1)).reshape(-1)


def with_exported_readout(
    probs: np.ndarray, sim: NoiseVaultSimulator, layout: list[int]
) -> np.ndarray:
    """Apply the simulator's readout errors as Aer defines them (rows = prepared state)."""
    rows = {
        tuple(e["gate_qubits"][0]): np.array(e["probabilities"])
        for e in sim.noise_model.to_dict()["errors"]
        if e["type"] == "roerror"
    }
    probs = probs.reshape((2,) * len(layout))
    for axis, q in enumerate(layout):
        probs = np.moveaxis(np.tensordot(rows[(q,)].T, probs, axes=([1], [axis])), 0, axis)
    return probs.reshape(-1)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


@pytest.fixture(scope="module")
def manila() -> Profile:
    return migrated(MANILA_V01)


def ring() -> Profile:
    """Four qubits on a ring with distinct coherence, a Pauli record and an uncalibrated x."""
    return Profile.model_validate(
        toy(
            device={
                "name": "ring",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 4,
            },
            connectivity={"edges": [[0, 1], [1, 2], [2, 3], [1, 3]]},
            gates={
                "rz": {"virtual": True},
                "sx": {"avg_infidelity": 1e-3, "duration_ns": 35},
                "x": {},
                "cz": {"avg_infidelity": 1e-2, "duration_ns": 70},
                "ecr": {"qubits": 2, "avg_infidelity": 2e-2, "duration_ns": 300},
            },
            idle={"t1_us": 80, "t2_us": 60},
            qubits=[
                {"index": 1, "t1_us": 30, "t2_us": 50, "dephasing_rate_per_s": 2000},
                {"index": 3, "t1_us": 120, "t2_us": 20},
            ],
            calibrations=[
                {"gate": "cz", "qubits": [1, 3], "avg_infidelity": 3e-2, "duration_ns": 90},
                {"gate": "cz", "qubits": [2, 3], "pauli": [1e-3 * (k + 1) for k in range(15)]},
                {"gate": "ecr", "qubits": [0, 1], "avg_infidelity": 1.5e-2},
            ],
            readout={"p1_given_0": 0.01, "p0_given_1": 0.04, "duration_ns": 800},
            prep={"error": 0.02},
        )
    )


# per-gate channels ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "qargs"),
    [
        ("sx", (2,)),
        ("x", (0,)),  # uncalibrated: the typical 1-qubit native's channels
        ("cz", (1, 3)),
        ("cz", (3, 1)),  # reversed operands of a record, non-contiguous qubits
        ("cz", (3, 2)),  # reversed Pauli record: labels must follow the operands
    ],
)
def test_gate_superoperator_equals_the_shared_channels(name: str, qargs: tuple) -> None:
    profile = ring()
    sim = quiet_export(profile)
    sim.set_options(method="superop", enable_truncation=False)
    circuit = QuantumCircuit(4)
    gate = sim.target.operation_from_name(name)
    circuit.append(gate, list(qargs))
    circuit.save_superop()
    actual = np.asarray(sim.run(circuit).result().data()["superop"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        built = resolve_op(
            profile.table,
            name,
            qargs,
            unknown_gates="typical",
            report=Report.start(profile, "t", None),
        )
    ideal = ChannelSpec("pauli", qargs, (gates.lookup(name).unitary(),))
    expected = superoperator([ideal, *built.channels], wires=(3, 2, 1, 0))  # Qiskit order
    assert built.channels
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_multi_qubit_kraus_channels_are_reordered_for_qiskit() -> None:
    # No built-in channel is a 2-qubit non-Pauli map yet; a CX written as one Kraus operator on
    # wires (5, 2) must act with 5 as control once placed on qargs (2, 5).
    cx = gates.lookup("cx").unitary()
    channel = ChannelSpec("pauli", (5, 2), (cx,))
    actual = SuperOp(gate_error([channel], (2, 5))).data
    np.testing.assert_allclose(actual, superoperator([channel], wires=(5, 2)), atol=1e-12)


# circuits against the reference --------------------------------------------------------------


@pytest.mark.parametrize("layout", [[0, 1, 2], [4, 3, 2], [2, 1, 0]])
def test_native_circuits_with_readout_match_the_reference(manila: Profile, layout: list) -> None:
    sim = quiet_export(manila)
    sim.set_options(method="density_matrix")
    circuit = ghz(3)
    circuit.rx(0.7, 2)
    circuit.cx(2, 1)
    compiled = transpile(circuit, sim, initial_layout=layout, seed_transpiler=5)
    ours = with_exported_readout(aer_probabilities(sim, compiled, layout), sim, layout)
    ref = reference(manila, ops_of(compiled, layout), 3, layout=layout, readout=True)
    assert tvd(ours, ref) <= 1e-9


def test_directed_reversed_and_typical_gates_match_the_reference() -> None:
    profile = ring()
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(4)
    circuit.sx(0)
    circuit.sx(3)
    circuit.x(2)  # uncalibrated: typical noise
    circuit.ecr(1, 0)  # directed gate against its record's direction: the definition
    circuit.cz(3, 2)  # Pauli record on the reversed pair
    circuit.rz(0.3, 3)
    circuit.cz(3, 1)
    layout = [0, 1, 2, 3]
    ours = with_exported_readout(aer_probabilities(sim, circuit, layout), sim, layout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        ref = reference(profile, ops_of(circuit, layout), 4, readout=True)
    assert tvd(ours, ref) <= 1e-9


def test_asymmetric_readout_is_exported_exactly(manila: Profile) -> None:
    sim = quiet_export(manila)
    rows = {
        tuple(e["gate_qubits"][0]): e["probabilities"]
        for e in sim.noise_model.to_dict()["errors"]
        if e["type"] == "roerror"
    }
    for q in range(5):
        a, b = manila.table.qubit(q).readout
        assert a != b
        np.testing.assert_array_equal(rows[(q,)], [[1 - a, a], [b, 1 - b]])


def test_aer_reads_the_readout_rows_as_prepared_states(manila: Profile) -> None:
    # Aer's from_backend replaces qubit 0's P(0|1) = 0.0548 and P(1|0) = 0.0158 by their mean;
    # sampling separates the right orientation from the transposed one by about 80 sigma.
    sim = quiet_export(manila)
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    shots = 200_000
    counts = sim.run(circuit, shots=shots, seed_simulator=7).result().get_counts()
    expected = reference(manila, [Op("x", (0,))], 1, layout=[0], readout=True)[0]
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert abs(counts.get("0", 0) / shots - expected) < 5 * sigma


def test_all_to_all_trapped_ion_device_transpiles_and_runs() -> None:
    profile = Profile.uniform(
        "ion8",
        technology="trapped_ion",
        num_qubits=8,
        one_qubit_error=2e-4,
        two_qubit_error=4e-3,
        readout_error=3e-3,
        t1_us=1e8,
        t2_us=1e6,
        one_qubit_ns=10_000,
        two_qubit_ns=200_000,
    )
    sim = quiet_export(profile)
    assert len(sim.target["rxx"]) == 8 * 7  # every ordered pair of an all-to-all device

    circuit = ghz(8)
    circuit.measure_all()
    compiled = transpile(circuit, sim, seed_transpiler=2)
    counts = sim.run(compiled, shots=4000, seed_simulator=3).result().get_counts()
    assert counts.get("0" * 8, 0) + counts.get("1" * 8, 0) > 0.9 * 4000

    sim.set_options(method="density_matrix")
    small = transpile(ghz(4), sim, initial_layout=[6, 1, 4, 0], seed_transpiler=2)
    layout = [6, 1, 4, 0]
    ours = aer_probabilities(sim, small, layout)
    ref = reference(profile, ops_of(small, layout), 4, layout=layout, readout=False)
    assert tvd(ours, ref) <= 1e-9


@pytest.mark.timing
def test_large_all_to_all_exports_fast() -> None:
    # Eight entanglers on 48 qubits: building every ordered pair's channels took 47 s.
    profile = Profile.uniform(
        "ion48",
        technology="trapped_ion",
        num_qubits=48,
        one_qubit_error=1e-4,
        two_qubit_error=2e-3,
        t1_us=1e7,
        t2_us=1e6,
        two_qubit_ns=200_000,
    )
    start = time.perf_counter()
    sim = quiet_export(profile)
    assert time.perf_counter() - start < 10
    assert len(sim.target["cz"]) == 48 * 47


@pytest.mark.slow
@pytest.mark.parametrize("ref", [info.id for info in nv.profiles()])
def test_every_bundled_profile_exports_transpiles_and_runs(ref: str) -> None:
    profile = nv.load(ref)
    sim = quiet_export(profile)
    circuit = ghz(4)
    circuit.measure_all()
    compiled = transpile(circuit, sim, seed_transpiler=2)
    counts = sim.run(compiled, shots=1000, seed_simulator=1).result().get_counts()
    assert counts.get("0000", 0) + counts.get("1111", 0) > 800


def test_natives_aer_lacks_run_as_labelled_unitaries_with_their_noise() -> None:
    profile = Profile.uniform(
        "grid",
        technology="superconducting",
        num_qubits=2,
        one_qubit_error=1e-3,
        two_qubit_error=2e-2,
        t1_us=50,
        two_qubit_ns=100,
    )
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(2)
    circuit.sx(0)
    circuit.append(iSwapGate(), [0, 1])
    ours = aer_probabilities(sim, circuit, [0, 1])
    ref = reference(profile, [Op("sx", (0,)), Op("iswap", (0, 1))], 2, readout=False)
    assert tvd(ours, ref) <= 1e-9


def test_ms_exports_as_rxx_with_the_ms_noise() -> None:
    profile = Profile.model_validate(
        toy(
            device={"name": "ions", "vendor": "test", "technology": "trapped_ion", "num_qubits": 3},
            connectivity="all_to_all",
            gates={
                "rz": {"virtual": True},
                "rx": {"avg_infidelity": 5e-4},
                "ry": {"avg_infidelity": 5e-4},
                "ms": {"avg_infidelity": 1e-2},
            },
        )
    )
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    compiled = transpile(ghz(3), sim, seed_transpiler=1)
    assert "rxx" in compiled.count_ops()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        ref = reference(profile, ops_of(compiled, [0, 1, 2]), 3, readout=False)
    assert tvd(aer_probabilities(sim, compiled, [0, 1, 2]), ref) <= 1e-9
    assert any(a.what == "gate ms" for a in sim.report.approximated)


# target and validation -----------------------------------------------------------------------


def test_transpile_routes_around_a_disabled_pair() -> None:
    profile = Profile.model_validate(
        toy(
            device={
                "name": "sq",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 4,
            },
            connectivity={"edges": [[0, 1], [1, 2], [2, 3], [3, 0]]},
            calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}],
        )
    )
    sim = quiet_export(profile)
    assert {(0, 1), (1, 0)}.isdisjoint(sim.target["cz"])
    circuit = ghz(2)
    circuit.measure_all()
    compiled = transpile(circuit, sim, initial_layout=[0, 1], seed_transpiler=4)
    used = {
        tuple(sorted(compiled.find_bit(q).index for q in i.qubits))
        for i in compiled.data
        if i.operation.num_qubits == 2
    }
    assert used and (0, 1) not in used
    assert sim.run(compiled, shots=10).result().success


_LINE_GATES = {
    "rz": {"virtual": True},
    "sx": {"avg_infidelity": 1e-3, "duration_ns": 35},
    "x": {"avg_infidelity": 1e-3, "duration_ns": 35},
    "cz": {"avg_infidelity": 1e-2, "duration_ns": 70},
}
_QUBIT_0_UNUSABLE = {
    "qubit off": {"qubits": [{"index": 0, "disabled": True}]},
    "1q gates off": {
        "calibrations": [
            {"gate": "sx", "qubits": [0], "disabled": True},
            {"gate": "x", "qubits": [0], "disabled": True},
        ]
    },
    "only pair off": {"calibrations": [{"gate": "cz", "qubits": [0, 1], "disabled": True}]},
    "sx off": {"calibrations": [{"gate": "sx", "qubits": [0], "disabled": True}]},
}


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("case", list(_QUBIT_0_UNUSABLE))
def test_suggested_layout_transpiles_around_disabled_parts_at_every_level(
    case: str, level: int
) -> None:
    # Level 0 places circuit qubit i on physical qubit i whatever the Target allows (Qiskit's
    # own Targets fail the same way), so the supported route there is an explicit layout.
    profile = Profile.model_validate(
        toy(
            device={
                "name": "line",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 4,
            },
            connectivity={"edges": [[0, 1], [1, 2], [2, 3]]},
            gates=_LINE_GATES,
            **_QUBIT_0_UNUSABLE[case],
        )
    )
    sim = quiet_export(profile)
    circuit = QuantumCircuit(2)
    circuit.h([0, 1])
    circuit.cx(0, 1)
    circuit.measure_all()
    layout = list(profile.suggest_layout(2).values())
    compiled = transpile(
        circuit, sim, initial_layout=layout, optimization_level=level, seed_transpiler=3
    )
    used = {compiled.find_bit(q).index for i in compiled.data for q in i.qubits}
    assert used and 0 not in used
    assert sim.run(compiled, shots=10).result().success


_QUBIT_0_SHORT = {
    "sx re-enabled elsewhere": {
        "gates": {**_LINE_GATES, "sx": {"avg_infidelity": 2e-3, "disabled": True}},
        "calibrations": [{"gate": "sx", "qubits": [q], "disabled": False} for q in (1, 2)],
    },
    "rx off, sx and ry left": {
        "gates": {
            "sx": {"avg_infidelity": 2e-3},
            "ry": {"avg_infidelity": 2e-3},
            "rx": {"avg_infidelity": 1e-3},
            "cz": {"avg_infidelity": 1e-2},
        },
        "calibrations": [{"gate": "rx", "qubits": [0], "disabled": True}],
        "qubits": [{"index": q, "readout": {"error": e}} for q, e in enumerate([1e-3, 0.2, 0.3])],
    },
}


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("case", list(_QUBIT_0_SHORT))
def test_suggested_layout_transpiles_around_a_qubit_missing_a_native(case: str, level: int) -> None:
    profile = Profile.model_validate(
        toy(
            device={
                "name": "tri",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 3,
            },
            connectivity="all_to_all",
            **_QUBIT_0_SHORT[case],
        )
    )
    sim = quiet_export(profile)
    circuit = QuantumCircuit(2)
    circuit.h([0, 1])
    circuit.cx(0, 1)
    circuit.measure_all()
    layout = list(profile.suggest_layout(2).values())
    compiled = transpile(
        circuit, sim, initial_layout=layout, optimization_level=level, seed_transpiler=3
    )
    assert sim.run(compiled, shots=10).result().success


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_a_qubit_without_x_is_used_only_when_needed_and_still_transpiles(level: int) -> None:
    profile = Profile.model_validate(
        toy(
            device={
                "name": "tri",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 3,
            },
            connectivity="all_to_all",
            gates=_LINE_GATES,
            calibrations=[{"gate": "x", "qubits": [0], "disabled": True}],
            qubits=[{"index": 0, "readout": {"error": 1e-4}}],
        )
    )
    sim = quiet_export(profile)
    assert 0 not in profile.suggest_layout(2).values()
    with pytest.warns(nv.NoiseVaultWarning, match=r"qubit 0 \(x disabled\)"):
        layout = list(profile.suggest_layout(3).values())
    circuit = QuantumCircuit(3)
    circuit.h([0, 1, 2])
    circuit.x(range(3))
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_all()
    compiled = transpile(
        circuit, sim, initial_layout=layout, optimization_level=level, seed_transpiler=3
    )
    assert sim.run(compiled, shots=10).result().success


def test_target_carries_errors_and_durations_per_locus(manila: Profile) -> None:
    sim = quiet_export(manila)
    target = sim.target
    record = next(r for r in manila.calibrations if (r.gate, r.qubits) == ("cx", (1, 2)))
    assert target["cx"][(1, 2)].duration == pytest.approx(record.duration_ns * 1e-9)
    assert target["cx"][(1, 2)].error == pytest.approx(record.avg_infidelity, rel=1e-9)
    assert (0, 2) not in target["cx"]
    a, b = manila.table.qubit(3).readout
    assert target["measure"][(3,)].error == pytest.approx((a + b) / 2)
    assert target["rz"][(0,)].error == 0.0


_TRANSPILE_FIRST = (
    "transpile it for this simulator first: from qiskit import transpile;"
    " sim.run(transpile(circuit, sim))"
)


def test_run_rejects_an_untranspiled_circuit_with_the_fix(manila: Profile) -> None:
    sim = quiet_export(manila)
    with pytest.raises(
        CircuitNotNativeError, match=r"h on qubits \[0\].*transpile\(circuit, sim\)"
    ) as caught:
        sim.run(ghz(2))
    assert caught.value.hint == _TRANSPILE_FIRST
    wrong_pair = QuantumCircuit(3, name="wrong_pair")
    wrong_pair.cx(0, 2)
    with pytest.raises(CircuitNotNativeError) as caught:
        sim.run(wrong_pair)
    assert caught.value.message == (
        "circuit 'wrong_pair': cx on qubits [0, 2] is not available on ibm_manila (the device"
        " does not provide cx on that locus)"
    )
    assert caught.value.hint == _TRANSPILE_FIRST
    with pytest.raises(CircuitNotNativeError) as caught:
        sim.run(QuantumCircuit(6, name="wide"))
    assert caught.value.message == "circuit 'wide' has 6 qubits but ibm_manila has 5"
    assert caught.value.hint == _TRANSPILE_FIRST
    assert sim.run(transpile(ghz(2), sim), shots=10).result().success


# reset and delays ----------------------------------------------------------------------------


def test_reset_flips_with_the_preparation_error() -> None:
    profile = ring()
    sim = quiet_export(profile, readout=False)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(4)
    circuit.reset(2)
    assert aer_probabilities(sim, circuit, [2]) == pytest.approx([0.98, 0.02], abs=1e-12)


def _delayed(prepare: str, ns: float) -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    getattr(circuit, prepare)(1)
    if ns:
        circuit.delay(ns, 1, unit="ns")
    circuit.save_density_matrix([1])
    return circuit


def test_delays_relax_and_dephase() -> None:
    profile = ring()
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    q1 = profile.table.qubit(1)
    ns = 4000.0

    def rho(prepare: str, delay: float) -> np.ndarray:
        result = sim.run(_delayed(prepare, delay)).result()
        return np.asarray(result.data()["density_matrix"])

    excited = rho("x", ns)[1, 1] / rho("x", 0)[1, 1]
    assert excited == pytest.approx(np.exp(-ns / q1.t1_ns), rel=1e-9)
    coherence = abs(rho("sx", ns)[0, 1]) / abs(rho("sx", 0)[0, 1])
    dephasing = 1 - 2 * q1.dephasing_rate_per_s * ns * 1e-9
    assert coherence == pytest.approx(np.exp(-ns / q1.t2_ns) * dephasing, rel=1e-9)


def test_delays_match_aers_relaxation_pass_without_extra_dephasing(manila: Profile) -> None:
    sim = quiet_export(manila)
    sim.set_options(method="density_matrix")
    circuit = _delayed("sx", 3000)
    t1s = [manila.table.qubit(q).t1_ns * 1e-9 for q in range(4)]
    t2s = [
        min(manila.table.qubit(q).t2_ns, 2 * manila.table.qubit(q).t1_ns) * 1e-9 for q in range(4)
    ]
    from qiskit.circuit import Delay

    relaxed = PassManager([RelaxationNoisePass(t1s, t2s, op_types=Delay)]).run(circuit)
    plain = AerSimulator(method="density_matrix", noise_model=sim.noise_model)
    ours = np.asarray(sim.run(circuit).result().data()["density_matrix"])
    aers = np.asarray(plain.run(relaxed).result().data()["density_matrix"])
    np.testing.assert_allclose(ours, aers, rtol=0, atol=1e-12)
    assert abs(ours[0, 1]) < abs(
        np.asarray(plain.run(circuit).result().data()["density_matrix"])[0, 1]
    )


def test_scheduled_circuits_get_idle_relaxation() -> None:
    # Scheduling against this Target (no dt) writes delays in seconds: they must relax too.
    sim = quiet_export(ring(), readout=False)
    sim.set_options(method="density_matrix")
    layout = [0, 1, 3]
    plain = transpile(ghz(3), sim, initial_layout=layout, seed_transpiler=1)
    scheduled = transpile(
        ghz(3), sim, initial_layout=layout, scheduling_method="alap", seed_transpiler=1
    )
    delays = [i.operation for i in scheduled.data if i.operation.name == "delay"]
    assert delays and {d.unit for d in delays} == {"s"}

    def purity(circuit: QuantumCircuit) -> float:
        circuit = circuit.copy()
        circuit.save_density_matrix(layout)
        rho = np.asarray(sim.run(circuit).result().data()["density_matrix"])
        return float(np.real(np.trace(rho @ rho)))

    assert purity(scheduled) < purity(plain) - 1e-3


def test_delay_in_device_ticks_says_how_to_fix_it() -> None:
    sim = quiet_export(ring())
    circuit = QuantumCircuit(1)
    circuit.delay(100, 0)
    with pytest.raises(CircuitNotNativeError) as caught:
        sim.run(circuit)
    assert caught.value.message == "delay on qubit 0 has duration 100 dt"
    assert caught.value.hint == (
        "the profile has no sample time, so give delays a time unit (s, ms, us, ns, ps), e.g."
        " qc.delay(100, q, unit='ns')"
    )


def test_unbound_delay_duration_says_to_bind_it_first() -> None:
    sim = quiet_export(ring())
    t = Parameter("t")
    circuit = QuantumCircuit(1)
    circuit.delay(t, 0, unit="ns")
    with pytest.raises(CircuitNotNativeError) as caught:
        sim.run(circuit, parameter_binds=[{t: [100.0]}])
    assert caught.value.message == "delay on qubit 0 has the unbound duration t"
    assert caught.value.hint == (
        "delays relax before parameter_binds apply, so bind delay durations before run:"
        " sim.run(circuit.assign_parameters({...}))"
    )
    bound = circuit.assign_parameters({t: 100.0})
    assert sim.run(bound, shots=1).result().success


def _coherence_on_qubit_0(**coherence: float) -> Profile:
    return Profile.model_validate(
        toy(
            gates={
                "rz": {"virtual": True},
                "x": {"avg_infidelity": 0},
                "sx": {"avg_infidelity": 0},
                "cz": {"avg_infidelity": 1e-2, "duration_ns": 70},
            },
            qubits=[{"index": 0, **coherence}],
            prep={"error": 0},
        )
    )


def test_delays_report_each_qubit_without_relaxation_data_as_unknown() -> None:
    sim = quiet_export(_coherence_on_qubit_0(t1_us=50, t2_us=40), readout=False)
    sim.set_options(method="density_matrix")
    assert sim.report.unknown == []
    circuit = QuantumCircuit(3)
    circuit.x([0, 1])
    circuit.delay(100_000, [0, 1], unit="ns")
    decayed = np.exp(-100_000 / 50_000)
    assert aer_probabilities(sim, circuit, [0, 1]) == pytest.approx(
        [0, 1 - decayed, 0, decayed], abs=1e-12
    )
    sim.run(circuit).result()
    sim.run([circuit, circuit]).result()
    unknown_1 = "T1 and T2 of qubit 1 (no delay relaxation)"
    assert sim.report.unknown == [unknown_1]
    assert "delay: thermal relaxation and dephasing over its duration" in sim.report.exact
    other = QuantumCircuit(3)
    other.delay(500, 2, unit="ns")
    sim.run(other).result()
    assert sim.report.unknown == [unknown_1, "T1 and T2 of qubit 2 (no delay relaxation)"]
    assert f"unknown (no noise applied): {unknown_1}, T1 and T2 of qubit 2" in (
        sim.report.summary()
    )


@pytest.mark.parametrize(
    ("coherence", "factor"),
    [
        ({"t1_us": 50}, np.exp(-10_000 / (2 * 50_000))),
        ({"t2_us": 40}, np.exp(-10_000 / 40_000)),
        ({"dephasing_rate_per_s": 2000}, 1 - 2 * 2000 * 10_000e-9),
    ],
    ids=["t1-only", "t2-only", "dephasing-only"],
)
def test_delays_relax_with_partial_coherence_data_and_report_nothing_unknown(
    coherence: dict, factor: float
) -> None:
    sim = quiet_export(_coherence_on_qubit_0(**coherence), readout=False)
    sim.set_options(method="density_matrix")
    circuit = QuantumCircuit(3)
    circuit.sx(0)
    circuit.delay(10_000, 0, unit="ns")
    circuit.save_density_matrix([0])
    rho = np.asarray(sim.run(circuit).result().data()["density_matrix"])
    assert abs(rho[0, 1]) == pytest.approx(0.5 * factor, rel=1e-9)
    assert sim.report.unknown == []


# report --------------------------------------------------------------------------------------


def _reported_profile(**extra: Any) -> Profile:
    return Profile.model_validate(
        toy(
            gates={
                "rz": {"virtual": True},
                "sx": {"avg_infidelity": 1e-3, "duration_ns": 35},
                "x": {},
                "cz": {"avg_infidelity": 1e-4, "duration_ns": 400},
                "zz": {"avg_infidelity": 1e-2},
            },
            idle={"t1_us": 20, "t2_us": 30},
            **extra,
        )
    )


def test_typical_noise_warnings_point_at_the_callers_line() -> None:
    with pytest.warns(NoiseApproximationWarning, match="x on qubits") as caught:
        to_qiskit(_reported_profile())
    assert [w.filename for w in caught] == [__file__] * len(caught)


def test_report_lists_what_the_export_did() -> None:
    profile = _reported_profile(effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4}])
    with pytest.warns(NoiseApproximationWarning, match="x on qubits") as caught:
        sim = to_qiskit(profile)
    assert len([w for w in caught if "x on qubits" in str(w.message)]) == 1
    report = sim.report
    assert report.framework == "qiskit" and report.options == {
        "unknown_gates": "typical",
        "readout": True,
    }
    assert any(e.startswith("gate noise") for e in report.exact)
    approximated = {a.what: a.how for a in report.approximated}
    assert approximated["gate x"] == "noise of the typical 1-qubit native gate"
    assert approximated["gate zz"] == "exported as Qiskit rzz"
    assert "effect leakage on cz" in report.omitted
    assert any(u.startswith("readout error of qubits [0, 1, 2]") for u in report.unknown)
    assert {(c.gate, c.qubits) for c in report.clamped} >= {("cz", (0, 1)), ("cz", (1, 2))}
    assert report.events == {}

    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.x(0)
    circuit.x(2)
    sim.run(circuit, shots=1)
    assert report.events["typical_noise_used"]["x"] == 3


def test_unknown_gates_error_leaves_uncalibrated_natives_to_transpile_around() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseApproximationWarning)
        sim = to_qiskit(ring(), unknown_gates="error")
    assert "x" not in sim.target.operation_names
    assert (
        "native x: no error metric on qubits [0, 1, 2, 3] and unknown_gates='error', so"
        " transpile does not use it there (unknown_gates='typical' gives it the typical"
        " native's noise)"
    ) in sim.report.omitted
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    compiled = transpile(circuit, sim, seed_transpiler=1)
    assert "x" not in compiled.count_ops()
    assert sim.run(compiled, shots=10).result().success
    assert sim.report.events == {}  # no guessed noise anywhere


def test_unknown_gates_error_refuses_when_no_entangler_is_calibrated() -> None:
    profile = Profile.model_validate(
        toy(gates={"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "cz": {}})
    )
    with pytest.raises(UnsupportedDevice, match=r"no two-qubit native.*cz: no error metric"):
        to_qiskit(profile, unknown_gates="error")


_ONE_QUBIT_NATIVES = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}}
_NO_EDGES = {"edges": []}


_UNCALIBRATED = (
    " and unknown_gates='error', so transpile does not use it there (unknown_gates='typical'"
    " gives it the typical native's noise))"
)


@pytest.mark.parametrize(
    ("sections", "unknown_gates", "ending", "hint"),
    [
        (
            {
                "gates": {**_ONE_QUBIT_NATIVES, "ms": {"avg_infidelity": 1e-2}},
                "connectivity": _NO_EDGES,
            },
            "typical",
            "compile to. Its connectivity allows no pair of enabled qubits, so profile.to_cirq()"
            " cannot run a two-qubit gate either",
            None,
        ),
        (
            {
                "gates": {
                    **_ONE_QUBIT_NATIVES,
                    "cz": {"avg_infidelity": 1e-2},
                    "ms": {"avg_infidelity": 1e-2},
                },
                "calibrations": [{"gate": "cz", "qubits": [0, 1], "disabled": True}],
                "connectivity": _NO_EDGES,
            },
            "typical",
            "compile to (cz: disabled on every locus; ms: ms has no calibration on (0, 1) and"
            " connectivity does not allow it). The profile allows no two-qubit native on any pair"
            " of enabled qubits, so profile.to_cirq() cannot run one either",
            None,
        ),
        (
            {"gates": {"sx": {"disabled": True}, "cz": {"avg_infidelity": 1e-2}}},
            "typical",
            "compile to (sx: disabled on every locus). The profile allows no one-qubit native on"
            " any enabled qubit, so profile.to_cirq() cannot run one either",
            None,
        ),
        (
            {"gates": {**_ONE_QUBIT_NATIVES, "cz": {}}},
            "error",
            "compile to (cz: no error metric on qubits [(0, 1), (1, 0), (1, 2), (2, 1)]"
            + _UNCALIBRATED,
            "simulate it with profile.to_cirq() instead, or give the profile a calibrated"
            " two-qubit native that Qiskit provides",
        ),
        (
            {"gates": {"sx": {}, "cz": {"avg_infidelity": 1e-2}}},
            "error",
            "compile to (sx: no error metric on qubits [0, 1, 2]" + _UNCALIBRATED,
            "simulate it with profile.to_cirq() instead, or give the profile a calibrated"
            " one-qubit native that Qiskit provides",
        ),
    ],
    ids=[
        "no pair",
        "disabled or unconnected",
        "one-qubit disabled",
        "uncalibrated",
        "one-qubit uncalibrated",
    ],
)
def test_a_refusal_says_why_and_whether_cirq_can_run_the_gate(
    sections, unknown_gates, ending, hint
) -> None:
    with pytest.raises(UnsupportedDevice) as refused:
        to_qiskit(Profile.model_validate(toy(**sections)), unknown_gates=unknown_gates)
    message = refused.value.message
    assert message.endswith(ending), message
    assert refused.value.hint == hint


def test_the_report_names_a_native_that_no_listed_pair_allows() -> None:
    gates = {**_ONE_QUBIT_NATIVES, "cz": {}, "ms": {"avg_infidelity": 1e-2}}
    cz = [{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 2e-2}]
    profile = Profile.model_validate(toy(gates=gates, calibrations=cz, connectivity=_NO_EDGES))
    sim = to_qiskit(profile)
    assert sorted(sim.target.operation_names) == ["cz", "delay", "measure", "reset", "rz", "sx"]
    omission = "native ms: ms has no calibration on (0, 1) and connectivity does not allow it"
    assert omission in sim.report.omitted


def test_readout_must_be_a_bool() -> None:
    with pytest.raises(ValueError, match="readout='symmetrize': pass True or False"):
        to_qiskit(ring(), readout="symmetrize")  # type: ignore[arg-type]


def test_effects_that_demand_modelling_are_refused() -> None:
    effect = {"type": "leakage", "gate": "cz", "prob": 1e-4, "allow": "approximate"}
    with pytest.raises(UnsupportedEffect, match="allow"):
        to_qiskit(_reported_profile(effects=[effect]))


def test_profile_method_forwards_options(manila: Profile) -> None:
    sim = manila.to_qiskit(readout=False)
    assert isinstance(sim, NoiseVaultSimulator) and sim.profile is manila
    assert "readout error (readout=False)" in sim.report.omitted
    assert not [e for e in sim.noise_model.to_dict()["errors"] if e["type"] == "roerror"]


# sqrt_iswap ----------------------------------------------------------------------------------


def test_sqrt_iswap_gate_and_its_cx_rule_are_exact() -> None:
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
    from qiskit.circuit.library import CXGate
    from qiskit.quantum_info import Operator

    registry = gates.GATES["sqrt_iswap"].unitary()
    assert np.allclose(Operator(SqrtISwapGate()).data, registry, rtol=0, atol=1e-12)
    assert np.allclose(Operator(SqrtISwapGate().definition).data, registry, rtol=0, atol=1e-12)
    (rule,) = [
        c for c in SessionEquivalenceLibrary.get_entry(CXGate()) if "sqrt_iswap" in c.count_ops()
    ]
    assert rule.count_ops()["sqrt_iswap"] == 2
    assert np.allclose(Operator(rule).data, Operator(CXGate()).data, rtol=0, atol=1e-12)


def _sqrt_iswap_line() -> Profile:
    """Google's gate set on three qubits; sqrt_iswap has an asymmetric Pauli error."""
    pauli = [0.0] * 15
    pauli[0], pauli[14] = 0.015, 0.004
    return Profile.model_validate(
        toy(
            gates={
                "rz": {"virtual": True},
                "r": {"avg_infidelity": 1e-3, "duration_ns": 25},
                "sqrt_iswap": {"pauli": pauli, "duration_ns": 32},
            },
            idle={"t1_us": 20, "t2_us": 15},
            readout={"p1_given_0": 0.01, "p0_given_1": 0.04},
        )
    )


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_circuits_transpile_to_sqrt_iswap_and_match_the_reference(level: int) -> None:
    from qiskit.quantum_info import Operator, random_unitary

    profile = _sqrt_iswap_line()
    sim = quiet_export(profile)
    sim.set_options(method="density_matrix")
    circuit = ghz(3)
    circuit.rzz(0.4, 1, 2)
    circuit.unitary(random_unitary(4, seed=3), [0, 1])
    layout = [0, 1, 2]
    compiled = transpile(
        circuit, sim, initial_layout=layout, optimization_level=level, seed_transpiler=4
    )
    assert set(compiled.count_ops()) <= {"r", "rz", "sqrt_iswap"}
    assert Operator.from_circuit(compiled).equiv(Operator(circuit))
    ours = with_exported_readout(aer_probabilities(sim, compiled, layout), sim, layout)
    ref = reference(profile, ops_of(compiled, layout), 3, layout=layout, readout=True)
    assert tvd(ours, ref) <= 1e-9


def test_google_profiles_export_sqrt_iswap_and_report_the_gate_count_cost() -> None:
    sim = quiet_export(nv.load("google_weber"))
    assert {"r", "rz", "sqrt_iswap"} <= set(sim.target.operation_names)
    assert "sycamore" not in sim.target.operation_names
    assert any(a.what == "gate count of transpiled circuits" for a in sim.report.approximated)
    assert "native sycamore: no Qiskit instruction for it in this export" in sim.report.omitted
    (qiskit,) = nv.load("google_weber").check(frameworks=["qiskit"]).frameworks
    assert qiskit.passed and not qiskit.not_run


def _t2_above_2_t1(**gates: dict) -> Profile:
    """Qubit 0 states T2 above 2*T1; qubit 1 does not."""
    return Profile.model_validate(
        toy(
            device={
                "name": "t2",
                "vendor": "test",
                "technology": "superconducting",
                "num_qubits": 2,
            },
            connectivity={"edges": []},
            gates={"rz": {"virtual": True}, **gates},
            qubits=[
                {"index": 0, "t1_us": 10, "t2_us": 100},
                {"index": 1, "t1_us": 10, "t2_us": 15},
            ],
        )
    )


def _gate_path_t2_entries() -> list:
    profile = _t2_above_2_t1(sx={"avg_infidelity": 1e-3, "duration_ns": 35})
    report = Report.start(profile, "test", None)
    report.record_channels(gate_channels(profile.table.gate("sx", (0,)), [profile.table.qubit(0)]))
    return report.approximated


def test_delay_relaxation_reports_the_t2_clamp_like_the_gate_path() -> None:
    sim = quiet_export(_t2_above_2_t1())
    circuit = QuantumCircuit(2)
    circuit.delay(1000, [0, 1], unit="ns")
    sim.run(circuit).result()
    t2 = [a for a in sim.report.approximated if a.what.startswith("T2")]
    assert t2 == _gate_path_t2_entries()


@pytest.mark.parametrize("directed", [False, True])
def test_the_target_holds_every_pair_the_table_allows_off_the_connectivity(directed) -> None:
    profile = Profile.model_validate(
        toy(
            connectivity={"edges": [[0, 1]], "directed": directed},
            calibrations=[{"gate": "cz", "qubits": [2, 1], "avg_infidelity": 0.03}],
        )
    )
    target = quiet_export(profile).target
    pairs = [(a, b) for a in range(3) for b in range(3) if a != b]
    assert [p for p in pairs if p in target["cz"]] == [
        p for p in pairs if profile.table.allowed("cz", p)
    ]
    assert (1, 2) in target["cz"]


@pytest.mark.parametrize(
    "extra", [{}, {"cz": {"avg_infidelity": 2e-2}}], ids=["uniform", "plus-cz"]
)
def test_a_one_qubit_profile_runs_a_one_qubit_circuit(extra: dict) -> None:
    data = Profile.uniform(
        "u",
        technology="superconducting",
        num_qubits=1,
        one_qubit_error=1e-2,
        two_qubit_error=2e-2,
        readout_error=0.05,
    ).to_dict()
    data["gates"].update(extra)
    sim = Profile.model_validate(data).to_qiskit()
    assert "cz" not in sim.target.operation_names
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    counts = sim.run(transpile(qc, sim), shots=4000, seed_simulator=1).result().get_counts()
    assert 0.03 < counts.get("0", 0) / 4000 < 0.09
