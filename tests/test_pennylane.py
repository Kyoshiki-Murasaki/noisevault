from __future__ import annotations

import time
import warnings
from types import ModuleType

import numpy as np
import pytest
from conftest import MANILA_V01, migrated, require, toy

from noisevault.channels import ChannelSpec, superoperator
from noisevault.conversion import resolve_op
from noisevault.errors import (
    LayoutError,
    MissingCalibrationError,
    NoiseApproximationWarning,
    UnsupportedEffect,
)
from noisevault.profile import Profile
from noisevault.reference import Op, probabilities
from noisevault.report import Report

pytestmark = pytest.mark.filterwarnings("ignore::noisevault.errors.NoiseApproximationWarning")

GHZ = [Op("h", (0,)), Op("cx", (0, 1)), Op("cx", (1, 2))]
MIRROR_HALF = [
    Op("sx", (0,)),
    Op("rz", (0,), (0.7,)),
    Op("x", (1,)),
    Op("sx", (2,)),
    Op("cx", (0, 1)),
    Op("rz", (1,), (-1.1,)),
    Op("cx", (2, 1)),
    Op("sx", (1,)),
]


@pytest.fixture
def qml() -> ModuleType:
    return require("pennylane")


@pytest.fixture
def manila() -> Profile:
    return migrated(MANILA_V01)


def _ion() -> Profile:
    return Profile.uniform(
        "ion",
        technology="trapped_ion",
        num_qubits=5,
        one_qubit_error=3e-4,
        two_qubit_error=6e-3,
        readout_error=4e-3,
        t1_us=2e5,
        t2_us=5e3,
        one_qubit_ns=1e4,
        two_qubit_ns=2e5,
    )


def _asymmetric_toy() -> Profile:
    """Two qubits, ideal basis changes (virtual h, z family) and very different readouts."""
    data = toy(
        gates={
            "rz": {"virtual": True},
            "h": {"virtual": True},
            "sx": {"avg_infidelity": 2e-3, "duration_ns": 35},
            "cz": {"avg_infidelity": 2e-2, "duration_ns": 70},
        },
        idle={"t1_us": 20, "t2_us": 15},
        qubits=[
            {"index": 0, "readout": {"p1_given_0": 0.02, "p0_given_1": 0.11}},
            {"index": 1, "readout": {"p1_given_0": 0.07, "p0_given_1": 0.01}},
        ],
    )
    return Profile.model_validate(data)


def _mirror() -> list[Op]:
    """MIRROR_HALF then its inverse, in native gates, so the ideal output is |000>."""
    undo: list[Op] = []
    for op in reversed(MIRROR_HALF):
        if op.name == "rz":
            undo.append(Op("rz", op.qubits, (-op.params[0],)))
        elif op.name == "sx":  # sx^-1 = rz(pi) sx rz(pi) up to a global phase
            undo += [Op("rz", op.qubits, (np.pi,)), op, Op("rz", op.qubits, (np.pi,))]
        else:
            undo.append(op)
    return MIRROR_HALF + undo


def _pl_ops(qml: ModuleType, ops: list[Op], wires: list | None = None) -> None:
    classes = {
        "h": qml.Hadamard,
        "x": qml.PauliX,
        "sx": qml.SX,
        "rz": qml.RZ,
        "rx": qml.RX,
        "cx": qml.CNOT,
        "cz": qml.CZ,
    }
    for op in ops:
        targets = list(op.qubits) if wires is None else [wires[q] for q in op.qubits]
        classes[op.name](*op.params, wires=targets)


def _noisy_probs(qml, model, ops: list[Op], wires: list) -> np.ndarray:
    @qml.qnode(qml.device("default.mixed", wires=wires))
    def circuit():
        _pl_ops(qml, ops, wires)
        return qml.probs(wires=wires)

    return np.asarray(qml.add_noise(circuit, model)())


def _tvd(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


# per-gate channels ---------------------------------------------------------------------------


def _pauli_device() -> Profile:
    """A cx whose Pauli channel is not symmetric under swapping its operands."""
    pauli = [0.0] * 15
    pauli[0], pauli[11] = 0.02, 0.005  # IX and ZI: the first letter acts on qubits[0]
    data = toy(
        device={"name": "pauli", "technology": "other", "num_qubits": 5},
        connectivity="all_to_all",
        gates={"rz": {"virtual": True}, "cx": {"qubits": 2, "pauli": pauli}},
    )
    return Profile.model_validate(data)


@pytest.mark.parametrize(
    ("device", "gate", "wires", "layout"),
    [
        ("manila", "sx", (0,), None),
        ("manila", "cx", (0, 1), None),
        ("manila", "cx", (1, 0), None),
        ("manila", "cx", (0, 1), {0: 3, 1: 2}),
        ("manila", "cx", (1, 0), {0: 3, 1: 2}),
        ("manila", "h", (1,), {0: 3, 1: 4}),
        ("pauli", "cx", (0, 1), None),
        ("pauli", "cx", (1, 0), {0: 4, 1: 1}),
    ],
    ids=[
        "sx",
        "cx",
        "cx-reversed",
        "cx-layout",
        "cx-layout-reversed",
        "typical-layout",
        "pauli-cx",
        "pauli-cx-layout-reversed",
    ],
)
def test_gate_superoperators_equal_core_channels(qml, manila, device, gate, wires, layout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = manila if device == "manila" else _pauli_device()
    model = to_pennylane(profile, layout=layout, readout=False)
    pl_gate = {"sx": qml.SX, "cx": qml.CNOT, "h": qml.Hadamard}[gate](wires=list(wires))
    [tape], _ = qml.noise.add_noise(qml.tape.QuantumScript([pl_gate]), model)
    physical = tuple(wires if layout is None else (layout[w] for w in wires))
    to_physical = dict(zip(wires, physical, strict=True))

    inserted = [
        ChannelSpec("pauli", tuple(to_physical[w] for w in op.wires), tuple(op.kraus_matrices()))
        for op in tape.operations[1:]
    ]
    assert [op.name for op in tape.operations[1:]] == ["QubitChannel"] * len(inserted)
    report = Report.start(profile, "t", None)
    core = resolve_op(profile.table, gate, physical, unknown_gates="typical", report=report)
    assert core.channels
    expected = superoperator(core.channels, physical)
    assert np.abs(superoperator(inserted, physical) - expected).max() < 1e-10


def test_virtual_rz_gets_no_channels(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    tape = qml.tape.QuantumScript([qml.RZ(0.3, wires=0), qml.PhaseShift(0.2, wires=1)])
    [noisy], _ = qml.noise.add_noise(tape, to_pennylane(manila, readout=False))
    assert [op.name for op in noisy.operations] == ["RZ", "PhaseShift"]


# circuits against the reference --------------------------------------------------------------


@pytest.mark.parametrize("readout", [True, False], ids=["readout", "no-readout"])
@pytest.mark.parametrize("circuit", ["ghz", "mirror"])
@pytest.mark.parametrize("device", ["manila", "ion"])
def test_noisy_qnode_matches_reference(qml, manila, device, circuit, readout) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = manila if device == "manila" else _ion()
    ops = GHZ if circuit == "ghz" else _mirror()
    layout = [4, 3, 2] if device == "manila" else [1, 4, 0]
    model = to_pennylane(profile, layout=layout, readout=readout)

    got = _noisy_probs(qml, model, ops, [0, 1, 2])
    expected = probabilities(profile, ops, 3, layout=layout, readout=readout)
    ideal = probabilities(
        Profile.uniform(
            "ideal", technology="other", num_qubits=3, one_qubit_error=0.0, two_qubit_error=0.0
        ),
        ops,
        3,
    )
    assert _tvd(got, expected) <= 1e-9
    assert _tvd(got, ideal) > 1e-3  # the noise is really there


def test_string_wires_follow_the_layout(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    labels = ["anc", "data", "flag"]
    model = to_pennylane(manila, layout={"anc": 2, "data": 1, "flag": 0})
    got = _noisy_probs(qml, model, _mirror(), labels)
    expected = probabilities(manila, _mirror(), 3, layout=[2, 1, 0])
    assert _tvd(got, expected) <= 1e-9
    assert model.physical_qubit("data") == 1


def test_wires_the_layout_cannot_place_are_layout_errors(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    with pytest.raises(LayoutError, match="layout="):
        _noisy_probs(qml, to_pennylane(manila), GHZ, ["a", "b", "c"])
    with pytest.raises(LayoutError, match=r"'c' is not in the layout.*<physical qubit>}$"):
        _noisy_probs(qml, to_pennylane(manila, layout={"a": 0, "b": 1}), GHZ, ["a", "b", "c"])
    with pytest.raises(LayoutError, match="wires 0 to 1; extend the list"):
        _noisy_probs(qml, to_pennylane(manila, layout=[3, 4]), GHZ, [0, 1, 2])
    with pytest.raises(LayoutError, match="both"):
        to_pennylane(manila, layout={"a": 0, "b": 0})


# readout -------------------------------------------------------------------------------------


def test_asymmetric_readout_follows_the_profile_convention(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_asymmetric_toy())
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def prepared_01():
        return qml.probs(wires=[0, 1])

    # both qubits in |0>: qubit 0 reads 1 with P(1|0) = 0.02, qubit 1 with 0.07
    got = np.asarray(qml.add_noise(prepared_01, model)())
    assert got == pytest.approx(np.kron([0.98, 0.02], [0.93, 0.07]), abs=1e-12)


@pytest.mark.parametrize(("a", "b"), [(0.0, 0.0), (0.7, 0.6), (0.3, 0.0)])
def test_any_confusion_matrix_is_reproduced(qml, a, b) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(
        gates={"rz": {"virtual": True}, "rx": {"avg_infidelity": 0.0}},
        qubits=[{"index": 0, "readout": {"p1_given_0": a, "p0_given_1": b}}],
    )
    model = to_pennylane(Profile.model_validate(data))

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit(flip):
        qml.RX(np.pi * flip, wires=0)
        return qml.probs(wires=[0])

    noisy = qml.add_noise(circuit, model)
    assert np.asarray(noisy(0.0)) == pytest.approx([1 - a, a], abs=1e-12)
    assert np.asarray(noisy(1.0)) == pytest.approx([b, 1 - b], abs=1e-12)


def test_pauli_word_readout_is_applied_in_the_measured_basis(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,)), Op("rz", (1,), (0.4,)), Op("sx", (1,))]

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        _pl_ops(qml, ops)
        return qml.expval(qml.X(0) @ qml.Z(1)), qml.expval(qml.Y(1)), qml.var(qml.X(0))

    xz, y, var_x = qml.add_noise(circuit, to_pennylane(profile))()
    signs = np.array([1, -1, -1, 1])
    in_x0 = probabilities(profile, ops + [Op("h", (0,))], 2)
    in_y1 = probabilities(profile, ops + [Op("sdg", (1,)), Op("h", (1,))], 2)
    x0 = float(in_x0 @ np.array([1, 1, -1, -1]))
    assert float(xz) == pytest.approx(float(in_x0 @ signs), abs=1e-12)
    assert float(y) == pytest.approx(float(in_y1 @ np.array([1, -1, 1, -1])), abs=1e-12)
    assert float(var_x) == pytest.approx(1 - x0**2, abs=1e-12)


def test_sums_get_readout_in_their_shared_basis_and_conflicts_are_reported(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,))]
    model = to_pennylane(profile)

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit(observable):
        _pl_ops(qml, ops)
        return qml.expval(observable)

    noisy = qml.add_noise(circuit, model)
    probs = probabilities(profile, ops, 2)
    in_x0 = probabilities(profile, ops + [Op("h", (0,))], 2)
    x0, z0, z1 = in_x0 @ [1, 1, -1, -1], probs @ [1, 1, -1, -1], probs @ [1, -1, 1, -1]
    z_sum = float(probs @ np.array([2, 0, -2, 0]))  # Z0 + Z0 Z1
    assert float(noisy(qml.Z(0) + qml.Z(0) @ qml.Z(1))) == pytest.approx(z_sum, abs=1e-12)
    assert float(noisy(qml.X(0) + qml.Z(1))) == pytest.approx(x0 + z1, abs=1e-12)

    with pytest.warns(NoiseApproximationWarning, match="split_non_commuting"):
        mixed = float(noisy(qml.X(0) + qml.Z(0)))
    x0_ideal = probabilities(profile, ops + [Op("h", (0,))], 2, readout=False) @ [1, 1, -1, -1]
    z0_ideal = probabilities(profile, ops, 2, readout=False) @ [1, 1, -1, -1]
    assert mixed == pytest.approx(x0_ideal + z0_ideal, abs=1e-12)
    assert "readout on observables not measured in one product basis" in model.report.omitted

    split = qml.add_noise(qml.transforms.split_non_commuting(circuit), to_pennylane(profile))
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseApproximationWarning)
        fixed = float(split(qml.X(0) + qml.Z(0)))
    assert fixed == pytest.approx(x0 + z0, abs=1e-12)


def test_repeated_pauli_terms_read_out_like_the_scaled_word(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,)), Op("rz", (1,), (0.4,)), Op("sx", (1,))]

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        _pl_ops(qml, ops)
        return (
            qml.expval(qml.X(0) + qml.X(0)),
            qml.expval(qml.dot([0.5, 0.5], [qml.Y(1), qml.Y(1)])),
        )

    xx, yy = qml.add_noise(circuit, to_pennylane(profile))()
    x0 = probabilities(profile, ops + [Op("h", (0,))], 2) @ [1, 1, -1, -1]
    y1 = probabilities(profile, ops + [Op("sdg", (1,)), Op("h", (1,))], 2) @ [1, -1, 1, -1]
    assert float(xx) == pytest.approx(2 * x0, abs=1e-12)
    assert float(yy) == pytest.approx(y1, abs=1e-12)


def test_computational_basis_measurements_share_one_simulation(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    ops = [Op("sx", (w,)) for w in range(4)] + [Op("cx", (w, w + 1)) for w in range(3)]
    dev = qml.device("default.mixed", wires=4)

    @qml.qnode(dev)
    def circuit():
        _pl_ops(qml, ops)
        return [qml.expval(qml.Z(w)) for w in range(4)] + [qml.probs(wires=[2, 3])]

    noisy = qml.add_noise(circuit, to_pennylane(manila))
    with qml.Tracker(dev) as tracker:
        *z, probs_23 = noisy()
    assert tracker.totals["simulations"] == 1
    expected = probabilities(manila, ops, 4).reshape([2] * 4)
    for w in range(4):
        marginal = expected.sum(axis=tuple(a for a in range(4) if a != w))
        assert float(z[w]) == pytest.approx(marginal[0] - marginal[1], abs=1e-9)
    assert _tvd(probs_23, expected.sum(axis=(0, 1)).ravel()) <= 1e-9


def test_sampled_counts_include_readout(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = _asymmetric_toy()
    ops = [Op("sx", (0,)), Op("cz", (0, 1)), Op("sx", (1,))]
    shots = 200_000

    @qml.qnode(qml.device("default.mixed", wires=2, seed=11))
    def circuit():
        _pl_ops(qml, ops)
        return qml.counts(wires=[0, 1])

    noisy = qml.set_shots(qml.add_noise(circuit, to_pennylane(profile)), shots=shots)
    counts = noisy()
    got = np.array([counts.get(k, 0) for k in ("00", "01", "10", "11")]) / shots
    expected = probabilities(profile, ops, 2)
    assert np.abs(got - expected).max() < 5 * np.sqrt(0.25 / shots)
    assert np.abs(got - probabilities(profile, ops, 2, readout=False)).max() > 0.02


def test_probs_without_wires_equals_probs_on_every_wire(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=3))
    def circuit():
        _pl_ops(qml, MIRROR_HALF)
        return qml.probs(), qml.probs(wires=[0, 1, 2])

    model = to_pennylane(manila)
    implicit, explicit = qml.add_noise(circuit, model)()
    expected = probabilities(manila, MIRROR_HALF, 3)
    assert _tvd(np.asarray(explicit), expected) <= 1e-9
    assert _tvd(np.asarray(implicit), np.asarray(explicit)) <= 1e-12
    assert any(a.what == "readout of measurements without wires" for a in model.report.approximated)


@pytest.mark.parametrize("measure", ["counts", "sample"])
def test_sampled_measurements_without_wires_include_readout(qml, manila, measure) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    shots = 100_000
    results = {}
    for wires in (None, [0, 1, 2]):

        @qml.qnode(qml.device("default.mixed", wires=3, seed=5))
        def circuit(wires=wires):
            _pl_ops(qml, MIRROR_HALF)
            return getattr(qml, measure)(wires=wires)

        out = qml.set_shots(qml.add_noise(circuit, to_pennylane(manila)), shots=shots)()
        results[str(wires)] = _frequencies(out, 3, measure)
    expected = probabilities(manila, MIRROR_HALF, 3)
    without_readout = probabilities(manila, MIRROR_HALF, 3, readout=False)
    sigma = np.sqrt(expected * (1 - expected) / shots)
    for got in results.values():
        assert np.all(np.abs(got - expected) <= 5 * sigma + 5 / shots)
        assert _tvd(got, without_readout) > 0.02
    assert results["None"] == pytest.approx(results["[0, 1, 2]"], abs=1e-12)


def test_device_wires_no_operation_touches_read_out_ideally(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=3))
    def circuit():
        qml.PauliX(0)
        qml.PauliX(1)
        return qml.probs()

    probs = np.asarray(qml.add_noise(circuit, to_pennylane(manila))()).reshape(2, 2, 2)
    assert probs[:, :, 1].sum() == 0
    expected = probabilities(manila, [Op("x", (0,)), Op("x", (1,))], 2)
    assert _tvd(probs[:, :, 0].ravel(), expected) <= 1e-9


def _frequencies(out, n: int, measure: str) -> np.ndarray:
    if measure == "counts":
        keys = [format(i, f"0{n}b") for i in range(2**n)]
        total = sum(out.values())
        return np.array([out.get(k, 0) for k in keys]) / total
    index = np.asarray(out, dtype=int) @ (1 << np.arange(n)[::-1])
    return np.bincount(index, minlength=2**n) / len(index)


def test_unknown_readout_is_reported_not_invented(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    profile = Profile.model_validate(toy())
    model = to_pennylane(profile)

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit():
        qml.PauliX(0)
        return qml.probs(wires=[0])

    got = np.asarray(qml.add_noise(circuit, model)())
    assert got == pytest.approx(
        probabilities(profile, [Op("x", (0,))], 1, readout=False), abs=1e-12
    )
    assert "readout on qubit 0" in model.report.unknown


def test_reset_gets_the_preparation_error(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(gates={"rz": {"virtual": True}, "x": {"avg_infidelity": 0.0}}, prep={"error": 0.05})
    model = to_pennylane(Profile.model_validate(data), readout=False)

    @qml.qnode(qml.device("default.mixed", wires=2))  # deferred measurement needs a spare wire
    def circuit():
        qml.PauliX(0)
        qml.measure(0, reset=True)
        return qml.probs(wires=[0])

    assert np.asarray(qml.add_noise(circuit, model)()) == pytest.approx([0.95, 0.05], abs=1e-12)

    unknown = to_pennylane(Profile.model_validate(toy()), readout=False)
    assert np.asarray(qml.add_noise(circuit, unknown)()) == pytest.approx([1, 0], abs=1e-12)
    assert "reset error on qubit 0" in unknown.report.unknown


# unknown gates, report -----------------------------------------------------------------------


def test_unknown_gate_warns_once_and_counts_every_use(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(manila)

    @qml.qnode(qml.device("default.mixed", wires=2))
    def circuit():
        qml.Hadamard(0)
        qml.Hadamard(0)
        qml.Hadamard(1)
        return qml.probs(wires=[0, 1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qml.add_noise(circuit, model)()
    messages = [str(w.message) for w in caught if issubclass(w.category, NoiseApproximationWarning)]
    assert len(messages) == 1 and "h on qubits [0]" in messages[0]
    assert model.report.events["typical_noise_used"]["h"] == 3


def test_unknown_gates_error_raises(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit():
        qml.Hadamard(0)
        return qml.probs(wires=[0])

    with pytest.raises(MissingCalibrationError, match="h on qubits"):
        qml.add_noise(circuit, to_pennylane(manila, unknown_gates="error"))()
    with pytest.raises(ValueError, match="choose 'typical' or 'error'"):
        to_pennylane(manila, unknown_gates="ignore")


def test_conditional_gates_get_the_noise_of_their_gate(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(manila)

    @qml.qnode(qml.device("default.mixed", wires=3))  # deferred measurement needs a spare wire
    def circuit():
        qml.PauliX(0)
        qml.cond(qml.measure(0), qml.PauliX)(1)
        return qml.probs(wires=[1])

    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseApproximationWarning)
        qml.add_noise(circuit, model)()
    assert "typical_noise_used" not in model.report.events
    assert "conditional gates" in [a.what for a in model.report.approximated]


def test_level_top_noises_adjoint_gates_as_written(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    @qml.qnode(qml.device("default.mixed", wires=1))
    def circuit():
        qml.adjoint(qml.SX(0))
        return qml.probs(wires=[0])

    user, top = to_pennylane(manila), to_pennylane(manila)
    qml.add_noise(circuit, user)()
    qml.add_noise(circuit, top, level="top")()
    assert "ry" in user.report.events["typical_noise_used"]
    assert "adjoint gates and templates" in [a.what for a in user.report.approximated]
    assert dict(top.report.events["typical_noise_used"]) == {"sxdg": 1}


def test_model_is_a_pennylane_noise_model_with_report(qml, manila) -> None:
    from noisevault.frameworks.pennylane import NoiseVaultPennyLaneModel

    model = manila.to_pennylane(layout=[0, 1])
    assert isinstance(model, qml.NoiseModel) and isinstance(model, NoiseVaultPennyLaneModel)
    assert model.profile is manila
    report = model.report.to_dict()
    assert report["framework"] == "pennylane" and report["framework_version"] == qml.__version__
    assert report["options"] == {"layout": [0, 1], "unknown_gates": "typical", "readout": True}


def test_effects_that_cannot_be_omitted_refuse_to_convert(qml) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    data = toy(effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4, "allow": "approximate"}])
    with pytest.raises(UnsupportedEffect, match="pennylane"):
        to_pennylane(Profile.model_validate(data))
    data["effects"][0]["allow"] = "omit"
    assert "effect leakage on cz" in to_pennylane(Profile.model_validate(data)).report.omitted


# gradients and scale -------------------------------------------------------------------------


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
def test_gradient_step_through_noisy_qnode(qml, diff_method) -> None:
    from pennylane import numpy as pnp

    from noisevault.frameworks.pennylane import to_pennylane

    model = to_pennylane(_ion(), layout=[2, 0])

    @qml.qnode(qml.device("default.mixed", wires=2), diff_method=diff_method)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(params[2], wires=1)
        return qml.expval(qml.Z(0) @ qml.Z(1))

    noisy = qml.add_noise(circuit, model)
    params = pnp.array([0.4, -0.9, 1.3], requires_grad=True)
    grad = qml.grad(noisy)(params)

    step = 1e-5
    finite = [(noisy(params + step * e) - noisy(params - step * e)) / (2 * step) for e in np.eye(3)]
    assert np.asarray(grad) == pytest.approx(np.asarray(finite, dtype=float), abs=1e-7)
    assert abs(float(noisy(params)) - float(circuit(params))) > 1e-4

    stepped = params - 0.1 * grad
    assert float(noisy(stepped)) < float(noisy(params))


def test_layered_circuit_converts_and_runs_quickly(qml, manila) -> None:
    from noisevault.frameworks.pennylane import to_pennylane

    rng = np.random.default_rng(7)
    angles = rng.uniform(0, 2 * np.pi, size=(6, 4, 2))
    start = time.perf_counter()
    model = to_pennylane(manila, layout=[1, 2, 3, 4])

    @qml.qnode(qml.device("default.mixed", wires=4))
    def circuit(theta):
        for layer in theta:
            for w in range(4):
                qml.RZ(layer[w, 0], wires=w)
                qml.SX(wires=w)
                qml.RZ(layer[w, 1], wires=w)
            for w in range(3):
                qml.CNOT(wires=[w, w + 1])
        return qml.probs(wires=range(4))

    noisy = qml.add_noise(circuit, model)
    first = np.asarray(noisy(angles))
    first_s = time.perf_counter() - start
    start = time.perf_counter()
    second = np.asarray(noisy(angles + 0.1))
    second_s = time.perf_counter() - start
    print(f"6-layer 4-qubit: first call {first_s:.3f} s, second call {second_s:.3f} s")

    assert first.sum() == pytest.approx(1) and not np.allclose(first, second)
    assert first_s < 10 and second_s < 5
