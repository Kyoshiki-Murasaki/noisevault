from __future__ import annotations

import os
import warnings
from importlib.util import find_spec
from math import pi

import numpy as np
import pytest
from conftest import require, toy

from noisevault import gates
from noisevault.channels import ChannelSpec, superoperator
from noisevault.conversion import TYPICAL_FIX, resolve_op
from noisevault.errors import (
    DisabledGateError,
    LayoutError,
    MissingCalibrationError,
    NoiseApproximationWarning,
)
from noisevault.profile import Profile
from noisevault.report import Report

GATES = {
    **toy()["gates"],
    "x": {"duration_ns": 35},
    "ecr": {"qubits": 2, "avg_infidelity": 7e-3, "duration_ns": 400, "symmetric": False},
}


def _setup(**sections) -> tuple[Profile, Report]:
    profile = Profile.model_validate(toy(gates=GATES, **sections))
    return profile, Report.start(profile, "test", None)


def _resolve(profile: Profile, report: Report, name: str, qubits, unknown_gates="typical"):
    return resolve_op(profile.table, name, qubits, unknown_gates=unknown_gates, report=report)


def test_calibrated_and_ideal_gates_convert_without_approximation() -> None:
    profile, report = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve(profile, report, "sx", (0,)).requested == 1e-3
        assert _resolve(profile, report, "cz", (1, 0)).requested == 1e-2
        assert _resolve(profile, report, "rz", (2,)).channels == ()
        assert _resolve(profile, report, "t", (2,)).channels == ()  # z-family, virtual rz
    assert report.events == {} and report.approximated == []


def test_disabled_gate_raises() -> None:
    profile, report = _setup(calibrations=[{"gate": "cz", "qubits": [0, 1], "disabled": True}])
    for unknown_gates in ("typical", "error"):
        with pytest.raises(DisabledGateError):
            _resolve(profile, report, "cz", (0, 1), unknown_gates)


@pytest.mark.parametrize(
    ("name", "qubits", "typical"),
    [("h", (0,), "sx"), ("x", (1,), "sx"), ("cx", (1, 2), "cz"), ("fsim", (0, 1), "cz")],
    ids=["not-defined", "uncalibrated", "non-native", "unknown-name"],
)
def test_typical_noise_is_used_reported_and_warned_once(name, qubits, typical) -> None:
    profile, report = _setup()
    with pytest.warns(NoiseApproximationWarning) as caught:
        first = _resolve(profile, report, name, qubits)
        second = _resolve(profile, report, name, qubits)
    assert first.gate.gate == typical and first.gate.qubits == qubits
    assert second.requested == first.requested
    assert len(caught) == 1 and TYPICAL_FIX in str(caught[0].message)
    assert report.events["typical_noise_used"][name] == 2
    assert [a.what for a in report.approximated] == [f"gate {name}"]


def test_unknown_gates_error_raises_instead() -> None:
    profile, report = _setup()
    for name, qubits in (("h", (0,)), ("x", (1,)), ("cz", (0, 2))):
        with pytest.raises(MissingCalibrationError, match="unknown_gates='typical'"):
            _resolve(profile, report, name, qubits, "error")
    assert report.events == {}


@pytest.mark.parametrize(("name", "qubits"), [("swap", (0, 1)), ("ccx", (0, 1, 2))])
def test_multi_entanglers_must_be_decomposed(name, qubits) -> None:
    profile, report = _setup(connectivity="all_to_all")
    with pytest.raises(MissingCalibrationError, match="decompose"):
        _resolve(profile, report, name, qubits)
    assert report.events == {}


def test_three_qubit_unknown_gate_must_be_decomposed() -> None:
    profile, report = _setup(connectivity="all_to_all")
    with pytest.raises(MissingCalibrationError, match="decompose"):
        _resolve(profile, report, "mystery3", (0, 1, 2))


def test_pair_the_profile_does_not_allow_has_no_typical_noise() -> None:
    profile, report = _setup()
    with pytest.raises(MissingCalibrationError, match="no calibrated 2-qubit native"):
        _resolve(profile, report, "cz", (0, 2))


def test_typical_may_use_a_reversed_directed_record_and_reports_it() -> None:
    gates = {"rz": {"virtual": True}, "sx": GATES["sx"], "ecr": GATES["ecr"]}
    profile = Profile.model_validate(
        toy(
            gates=gates,
            connectivity={"edges": [[1, 0]], "directed": True},
            calibrations=[{"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 6e-3}],
        )
    )
    report = Report.start(profile, "test", None)
    with pytest.warns(NoiseApproximationWarning):
        built = _resolve(profile, report, "cx", (0, 1))
    assert built.gate.origin == "reversed_record" and built.requested == 6e-3
    assert report.events["reversed_record_used"]["ecr"] == 1


@pytest.mark.parametrize(
    ("qubits", "match"),
    [((0, 5), "qubits 0..2"), ((1, 1), "distinct"), ((1, 2), "disabled")],
)
def test_bad_physical_qubits_are_layout_errors(qubits, match) -> None:
    profile, report = _setup(qubits=[{"index": 2, "disabled": True}])
    with pytest.raises(LayoutError, match=match):
        _resolve(profile, report, "cz", qubits)


def test_misuse_by_an_adapter_is_a_plain_error() -> None:
    profile, report = _setup()
    with pytest.raises(ValueError, match="unknown_gates"):
        _resolve(profile, report, "sx", (0,), "sometimes")
    with pytest.raises(ValueError, match="not a unitary gate"):
        _resolve(profile, report, "measure", (0,))
    with pytest.raises(ValueError, match="acts on 2 qubits"):
        _resolve(profile, report, "cz", (0,))


# fixed-angle gates across exports --------------------------------------------------------------

# Calibrates the rotations only, each with its own error, so the noise shows which one a fixed
# gate took; rz is calibrated, so S is not free.
_ROTATIONS = toy(
    gates={
        "rz": {"avg_infidelity": 2e-3, "duration_ns": 35},
        "rx": {"avg_infidelity": 3e-2, "duration_ns": 35},
        "rzz": {"avg_infidelity": 6e-2, "duration_ns": 70},
    }
)


def _installed(module: str) -> bool:
    return os.environ.get("NOISEVAULT_REQUIRE_ALL") == "1" or find_spec(module) is not None


def _infidelity(channels: list[tuple[list[np.ndarray], tuple[int, ...]]]) -> float:
    """Process infidelity of the composed channels; Pauli twirling leaves it unchanged."""
    wires = tuple(sorted({w for _, ws in channels for w in ws}))
    specs = [ChannelSpec("pauli", ws, tuple(k)) for k, ws in channels]
    return 1 - np.trace(superoperator(specs, wires)).real / 4 ** len(wires)


def _cirq_noise(profile: Profile, case: str) -> float:
    cirq = require("cirq")
    gate = {"sx": cirq.X**0.5, "x": cirq.X, "s": cirq.S, "zz": cirq.ZZ**0.5}[case]
    qids = cirq.LineQubit.range(cirq.num_qubits(gate))
    noisy = profile.to_cirq(unknown_gates="error", readout=False).noisy_operation(gate.on(*qids))
    return _infidelity([(list(cirq.kraus(op)), tuple(q.x for q in op.qubits)) for op in noisy[1:]])


def _pennylane_noise(profile: Profile, case: str) -> float:
    qml = require("pennylane")
    make = {
        "sx": lambda: qml.SX(0),
        "x": lambda: qml.PauliX(0),
        "s": lambda: qml.S(0),
        "zz": lambda: qml.IsingZZ(pi / 2, wires=[0, 1]),
    }[case]
    tape = qml.tape.QuantumScript([make()], [qml.probs(wires=[0])])
    model = profile.to_pennylane(unknown_gates="error", readout=False)
    (noisy,), _ = qml.add_noise(tape, model)
    return _infidelity([(op.kraus_matrices(), tuple(op.wires)) for op in noisy.operations[1:]])


def _stim_noise(profile: Profile, case: str) -> float:
    require("stim")
    from noisevault.frameworks.stim import to_stim

    gate = {"sx": "SQRT_X 0", "x": "X 0", "s": "S 0", "zz": "SQRT_ZZ 0 1"}[case]
    noise = list(to_stim(profile, gate, unknown_gates="error", readout="none"))[1:]
    assert all(inst.name.startswith("PAULI_CHANNEL") for inst in noise)
    return sum(p for inst in noise for p in inst.gate_args_copy())


def _qiskit_noise(profile: Profile, case: str) -> float:
    require("qiskit_aer")
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator, SuperOp

    sim = profile.to_qiskit(unknown_gates="error", readout=False)
    sim.set_options(method="superop")
    circuit = QuantumCircuit(sim.target.num_qubits)
    {"sx": circuit.sx, "x": circuit.x, "s": circuit.s, "zz": lambda: circuit.rzz(pi / 2, 0, 1)}[
        case
    ](*(() if case == "zz" else (0,)))
    native = transpile(circuit, sim, initial_layout=list(range(circuit.num_qubits)))
    ideal = SuperOp(Operator(native)).data
    native.save_superop()
    noisy = np.asarray(sim.run(native).result().data()["superop"])
    noise = noisy @ ideal.conj().T
    return 1 - np.trace(noise).real / len(noise)


_EXPORTS = {
    "cirq": ("cirq", _cirq_noise),
    "pennylane": ("pennylane", _pennylane_noise),
    "stim": ("stim", _stim_noise),
    "qiskit": ("qiskit_aer", _qiskit_noise),
}


@pytest.mark.parametrize(
    ("case", "rotation", "qubits"),
    [("sx", "rx", (0,)), ("x", "rx", (0,)), ("s", "rz", (0,)), ("zz", "rzz", (0, 1))],
)
def test_every_export_runs_a_fixed_gate_as_the_only_rotation_it_equals(
    case, rotation, qubits
) -> None:
    profile = Profile.model_validate(_ROTATIONS)
    got = {
        name: noise(profile, case)
        for name, (module, noise) in _EXPORTS.items()
        if _installed(module)
    }
    report = Report.start(profile, "test", None)
    built = resolve_op(profile.table, rotation, qubits, unknown_gates="error", report=report)
    want = _infidelity([(list(c.kraus), c.wires) for c in built.channels])
    assert len(got) >= 2 and want > 0
    assert got == pytest.approx(dict.fromkeys(got, want), abs=1e-9)


# Each fixed-angle two-qubit gate as Stim, Cirq and PennyLane write it, and the registry gate
# and parameters it equals on a profile that defines only ms, only zz, or only the rotations.
# A profile without the gate's own native names it after the rotation, except that a fixed
# registry gate (zz) keeps its own name.
_TWO_QUBIT = {
    "SQRT_XX": (
        lambda c: c.XX**0.5,
        lambda q: q.IsingXX(pi / 2, wires=[0, 1]),
        {"ms": ("ms", (0.0, 0.0)), "zz": ("rxx", (pi / 2,)), "rotations": ("rxx", (pi / 2,))},
    ),
    "SQRT_XX_DAG": (
        lambda c: c.XX**-0.5,
        lambda q: q.IsingXX(-pi / 2, wires=[0, 1]),
        {"ms": ("ms", (pi, 0.0)), "zz": ("rxx", (-pi / 2,)), "rotations": ("rxx", (-pi / 2,))},
    ),
    "SQRT_YY": (
        lambda c: c.YY**0.5,
        lambda q: q.IsingYY(pi / 2, wires=[0, 1]),
        {
            "ms": ("ms", (pi / 2, pi / 2)),
            "zz": ("ryy", (pi / 2,)),
            "rotations": ("ryy", (pi / 2,)),
        },
    ),
    "SQRT_YY_DAG": (
        lambda c: c.YY**-0.5,
        lambda q: q.IsingYY(-pi / 2, wires=[0, 1]),
        {
            "ms": ("ms", (pi / 2, -pi / 2)),
            "zz": ("ryy", (-pi / 2,)),
            "rotations": ("ryy", (-pi / 2,)),
        },
    ),
    "SQRT_ZZ": (
        lambda c: c.ZZ**0.5,
        lambda q: q.IsingZZ(pi / 2, wires=[0, 1]),
        {"ms": ("zz", ()), "zz": ("zz", ()), "rotations": ("rzz", (pi / 2,))},
    ),
    "SQRT_ZZ_DAG": (
        lambda c: c.ZZ**-0.5,
        lambda q: q.IsingZZ(-pi / 2, wires=[0, 1]),
        {"ms": ("rzz", (-pi / 2,)), "zz": ("rzz", (-pi / 2,)), "rotations": ("rzz", (-pi / 2,))},
    ),
}
_DEFINED = {"ms": {"ms"}, "zz": {"zz"}, "rotations": {"rxx", "ryy", "rzz"}}


def _named(framework: str, stim_name: str, defined: set[str]) -> tuple[str, np.ndarray]:
    """The registry name the export gives the gate on a profile defining ``defined``, and the
    gate's own unitary in that framework."""
    cirq_gate, pennylane_op, _ = _TWO_QUBIT[stim_name]
    if framework == "stim":
        stim = require("stim")
        from noisevault.frameworks.stim import gate_name

        return gate_name(stim_name, defined), stim.gate_data(stim_name).unitary_matrix
    if framework == "cirq":
        cirq = require("cirq")
        from noisevault.frameworks.cirq import gate_name

        gate = cirq_gate(cirq)
        return gate_name(gate, defined), cirq.unitary(gate)
    qml = require("pennylane")
    from noisevault.frameworks.pennylane import gate_name

    op = pennylane_op(qml)
    return gate_name(op, defined), qml.matrix(op)


@pytest.mark.parametrize("profile", sorted(_DEFINED))
@pytest.mark.parametrize("stim_name", sorted(_TWO_QUBIT))
@pytest.mark.parametrize("framework", ["cirq", "pennylane", "stim"])
def test_fixed_angle_two_qubit_gates_are_named_by_the_registry_gate_they_equal(
    framework, stim_name, profile
) -> None:
    name, unitary = _named(framework, stim_name, _DEFINED[profile])
    want, params = _TWO_QUBIT[stim_name][2][profile]
    assert name == want
    native = gates.GATES[name].unitary(*params)
    assert abs(np.vdot(native, unitary)) == pytest.approx(4, abs=1e-6)  # equal up to phase
