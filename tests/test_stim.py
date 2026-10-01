import copy
import pickle
import time
import warnings
from itertools import product

import numpy as np
import pytest
from conftest import MANILA_V01, migrated, require, toy

stim = require("stim")

from noisevault import gates  # noqa: E402
from noisevault.channels import (  # noqa: E402
    ChannelSpec,
    pauli_kraus,
    pauli_twirl,
    superoperator,
    thermal_relaxation_kraus,
)
from noisevault.conversion import resolve_op  # noqa: E402
from noisevault.errors import (  # noqa: E402
    LayoutError,
    MissingCalibrationError,
    NoiseApproximationWarning,
    UnsupportedEffect,
)
from noisevault.frameworks.stim import (  # noqa: E402
    ExistingNoiseError,
    NoiseVaultStimCircuit,
    layout_from_coords,
    sample_with_readout,
    to_stim,
)
from noisevault.profile import Profile  # noqa: E402
from noisevault.reference import _apply  # noqa: E402
from noisevault.report import Report  # noqa: E402

_STIM_TO_ROW = {name: row for row in gates.GATES.values() for name in row.stim}
_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}
_GHZ5 = [("H", (0,)), ("CX", (0, 1)), ("CX", (1, 2)), ("CX", (2, 3)), ("CX", (3, 4))]
_LAYER = [("SQRT_X", (0,)), ("CX", (0, 1)), ("CX", (2, 3)), ("S", (1,)), ("H", (4,))]
_LAYER += [("CX", (1, 2)), ("CX", (3, 4)), ("X", (2,))]
_INVERSE = {"SQRT_X": "SQRT_X_DAG", "S": "S_DAG"}
_MIRROR = _LAYER + [(_INVERSE.get(name, name), q) for name, q in reversed(_LAYER)]


@pytest.fixture(autouse=True)
def _quiet_typical_noise():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoiseApproximationWarning)
        yield


def _manila() -> Profile:
    return migrated(MANILA_V01)


def _readout_profile(p1_given_0: float = 0.02, p0_given_1: float = 0.1, **extra) -> Profile:
    data = toy(readout={"p1_given_0": p1_given_0, "p0_given_1": p0_given_1}, **extra)
    data["gates"] = {"rz": {"virtual": True}, "x": {"virtual": True}, "cz": {"virtual": True}}
    return Profile.model_validate(data)


def _grid(n: int, *, worse_rows: int = 0) -> Profile:
    """An n x n square-grid device with (row, col) coords; the first rows can be noisier."""
    edges = [(r * n + c, r * n + c + 1) for r in range(n) for c in range(n - 1)]
    edges += [(r * n + c, (r + 1) * n + c) for r in range(n - 1) for c in range(n)]
    base = Profile.uniform(
        f"grid{n}",
        technology="superconducting",
        num_qubits=n * n,
        one_qubit_error=1e-3,
        two_qubit_error=4e-3,
        readout_error=1e-2,
        t1_us=100,
        t2_us=80,
        one_qubit_ns=25,
        two_qubit_ns=40,
        connectivity=edges,
    )
    data = base.to_dict()
    data["qubits"] = [{"index": r * n + c, "coords": [r, c]} for r in range(n) for c in range(n)]
    bad = [e for e in edges if e[0] < worse_rows * n]
    data["calibrations"] = [{"gate": "cz", "qubits": list(e), "avg_infidelity": 0.05} for e in bad]
    return Profile.from_dict(data)


def _program(ops, n: int) -> stim.Circuit:
    lines = [f"{name} {' '.join(map(str, q))}" for name, q in ops]
    return stim.Circuit("\n".join([*lines, f"M {' '.join(map(str, range(n)))}"]))


def _twirled_reference(profile, ops, n, layout, *, asymmetric: bool) -> np.ndarray:
    """Density-matrix probabilities with each gate's resolve_op channel Pauli-twirled."""
    report = Report.start(profile, "test", None)
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1.0
    for name, qubits in ops:
        row = _STIM_TO_ROW[name]
        rho = _apply(rho, [row.unitary()], qubits, n)
        wires = tuple(layout[q] for q in qubits)
        built = resolve_op(profile.table, row.name, wires, unknown_gates="typical", report=report)
        if built.channels:
            rho = _apply(rho, pauli_kraus(pauli_twirl(built.channels, wires)), qubits, n)
    probs = np.real(np.diagonal(rho.reshape(2**n, 2**n))).reshape((2,) * n)
    for c in range(n):
        a, b = profile.table.qubit(layout[c]).readout
        s = (a + b) / 2
        matrix = [[1 - a, b], [a, 1 - b]] if asymmetric else [[1 - s, s], [s, 1 - s]]
        probs = np.moveaxis(np.tensordot(np.array(matrix), probs, axes=([1], [c])), 0, c)
    return probs.reshape(-1)


def _assert_within_5_sigma(bits: np.ndarray, expected: np.ndarray) -> float:
    shots, n = bits.shape
    index = bits.astype(int) @ (1 << np.arange(n)[::-1])
    observed = np.bincount(index, minlength=2**n) / shots
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert np.all(np.abs(observed - expected) <= 5 * sigma + 5 / shots)
    return 0.5 * float(np.abs(observed - expected).sum())


def _noise_after(out: stim.Circuit, gate: str) -> list[float]:
    items = list(out)
    at = next(i for i, inst in enumerate(items) if inst.name == gate)
    return items[at + 1].gate_args_copy()


# (a) twirled gate noise ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stim_gate", "canonical", "qubits", "layout"),
    [
        ("SQRT_X", "sx", (0,), {0: 3}),
        ("X", "x", (0,), {0: 1}),
        ("H", "h", (0,), {0: 2}),
        ("SQRT_X_DAG", "sxdg", (0,), {0: 0}),
        ("CX", "cx", (0, 1), {0: 2, 1: 1}),
        ("CX", "cx", (1, 0), {0: 3, 1: 4}),
        ("CZ", "cz", (0, 1), {0: 0, 1: 1}),
        ("ISWAP", "iswap", (0, 1), {0: 1, 1: 2}),
    ],
)
def test_gate_noise_is_the_twirl_of_the_shared_channel(stim_gate, canonical, qubits, layout):
    profile = _manila()
    out = to_stim(profile, f"{stim_gate} {' '.join(map(str, qubits))}", layout=layout)
    wires = tuple(layout[q] for q in qubits)
    report = Report.start(profile, "test", None)
    built = resolve_op(profile.table, canonical, wires, unknown_gates="typical", report=report)
    assert _noise_after(out, stim_gate) == pytest.approx(pauli_twirl(built.channels, wires))
    assert list(out)[1].name == f"PAULI_CHANNEL_{len(qubits)}"


def _direct_pauli_probabilities(channels, wires) -> np.ndarray:
    """Pauli error probabilities from the Pauli transfer matrix diagonal (Walsh transform)."""
    labels = ["".join(p) for p in product("IXYZ", repeat=len(wires))]
    d = 2 ** len(wires)
    s = superoperator(channels, wires)

    def op(label):
        m = np.eye(1)
        for c in label:
            m = np.kron(m, _PAULI[c])
        return m

    def apply(m):
        return (s @ m.flatten(order="F")).reshape(d, d, order="F")

    fidelity = {p: np.real(np.trace(op(p).conj().T @ apply(op(p)))) / d for p in labels}

    def commute(p, q):
        return sum(a != "I" and b != "I" and a != b for a, b in zip(p, q, strict=True)) % 2 == 0

    probs = [sum(fidelity[q] * (1 if commute(p, q) else -1) for q in labels) / d**2 for p in labels]
    return np.array(probs[1:])


def test_pauli_channel_2_order_matches_a_direct_pauli_transfer_computation():
    fast, slow = {"index": 0, "t1_us": 2, "t2_us": 1}, {"index": 1, "t1_us": 900, "t2_us": 700}
    profile = Profile.model_validate(toy(qubits=[fast, slow]))
    for order in [(0, 1), (1, 0)]:
        out = to_stim(profile, f"CZ {order[0]} {order[1]}")
        report = Report.start(profile, "test", None)
        built = resolve_op(profile.table, "cz", order, unknown_gates="typical", report=report)
        emitted = np.array(_noise_after(out, "CZ"))
        direct = _direct_pauli_probabilities(built.channels, order)
        assert emitted == pytest.approx(direct, abs=1e-9)
        # the short-T1 qubit carries the X errors: XI when it is the first target, IX otherwise
        xi, ix = emitted[3], emitted[0]
        assert (xi > 10 * ix) if order == (0, 1) else (ix > 10 * xi)


def test_stim_reads_pauli_channel_2_first_letter_on_the_first_target():
    args = [0.0] * 15
    args[3] = 1.0  # XI
    bits = stim.Circuit(f"PAULI_CHANNEL_2({','.join(map(str, args))}) 0 1\nM 0 1")
    assert bits.compile_sampler().sample(4).tolist() == [[True, False]] * 4


def test_z_family_gates_are_free_when_rz_is_virtual():
    out = to_stim(_manila(), "S 0\nZ 1\nS_DAG 2\nM 0 1 2", readout="none")
    assert str(out) == "S 0\nZ 1\nS_DAG 2\nM 0 1 2"


def test_unknown_gates_get_typical_noise_or_raise():
    profile = _manila()
    with pytest.warns(NoiseApproximationWarning, match="sqrt_y"):
        out = to_stim(profile, "SQRT_Y 0")
    assert list(out)[1].name == "PAULI_CHANNEL_1"
    assert out.report.events["typical_noise_used"]["sqrt_y"] == 1
    with pytest.raises(MissingCalibrationError, match="unknown_gates='typical'"):
        to_stim(profile, "SQRT_Y 0", unknown_gates="error")


@pytest.mark.parametrize("tick_ns", [None, 50.0])
def test_report_events_count_every_gate_application(tick_ns):
    circuit = "H 0\nH 0\nH 0\nREPEAT 100 {\n  H 1\n  TICK\n  REPEAT 2 {\n    H 1 2\n  }\n}"
    out = to_stim(_manila(), circuit, tick_ns=tick_ns)
    assert out.report.events["typical_noise_used"]["h"] == 3 + 100 * (1 + 2 * 2)


@pytest.mark.parametrize("gate", ["CXSWAP", "SWAPCX", "CZSWAP", "SWAP"])
def test_gates_needing_two_or_more_entanglers_must_be_decomposed(gate):
    with pytest.raises(MissingCalibrationError, match=f"`{gate} 0 1`.*decompose"):
        to_stim(_manila(), f"{gate} 0 1")


def test_two_qubit_identity_gets_no_noise():
    assert str(to_stim(_manila(), "II 0 1")) == "II 0 1"


def test_unusable_gate_errors_name_the_stim_instruction_and_the_fix():
    circuit = "H 0\nCX 0 1\nCX 1 2\nM 0 1 2"
    with pytest.raises(
        MissingCalibrationError, match=r"`CX 1 2` \(physical qubits \[1, 4\]\).*pass layout="
    ):
        to_stim(_manila(), circuit, layout={0: 0, 1: 1, 2: 4})
    with pytest.raises(MissingCalibrationError, match="`SQRT_Y 0` .*unknown_gates") as caught:
        to_stim(_manila(), "SQRT_Y 0", unknown_gates="error")
    assert "layout=" not in str(caught.value)


def test_repeated_targets_in_one_instruction_keep_gate_then_noise_order():
    out = to_stim(_manila(), "SQRT_X 0 0")
    assert [inst.name for inst in out] == ["SQRT_X", "PAULI_CHANNEL_1"] * 2


def test_measurement_feedback_gets_no_gate_noise():
    out = to_stim(_manila(), "M 0\nCX rec[-1] 1", readout="none")
    assert [inst.name for inst in out] == ["M", "CX"]
    assert any(a.what == "classically controlled Paulis" for a in out.report.approximated)


# (b) sampling matches the twirled density-matrix reference ------------------------------------


@pytest.mark.parametrize(
    ("ops", "layout"),
    [(_GHZ5, [0, 1, 2, 3, 4]), (_MIRROR, [4, 3, 2, 1, 0])],
    ids=["ghz5", "mirror"],
)
def test_sampling_matches_twirled_reference_on_manila(ops, layout):
    profile = _manila()
    out = to_stim(profile, _program(ops, 5), layout=layout)
    expected = _twirled_reference(profile, ops, 5, layout, asymmetric=False)
    bits = out.compile_sampler(seed=7).sample(200_000)
    _assert_within_5_sigma(bits, expected)


# (c) REPEAT blocks ---------------------------------------------------------------------------


def test_repeat_blocks_are_kept_with_the_noise_of_their_unrolled_body():
    circuit = stim.Circuit("""
        R 0 1
        REPEAT 3 {
            H 0
            CX 0 1
            REPEAT 2 {
                SQRT_X 1
            }
            MR 1
            DETECTOR(1, 0) rec[-1]
            TICK
        }
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    for tick_ns in (None, 80.0):
        out = to_stim(_manila(), circuit, tick_ns=tick_ns)
        blocks = [item for item in out if isinstance(item, stim.CircuitRepeatBlock)]
        assert [b.repeat_count for b in blocks] == [3]
        assert "PAULI_CHANNEL_2" in str(blocks[0].body_copy())
        assert out.flattened() == to_stim(_manila(), circuit.flattened(), tick_ns=tick_ns)
        assert (out.num_detectors, out.num_observables) == (3, 1)


def test_repeat_whose_first_pass_idles_differently_is_peeled_once():
    circuit = stim.Circuit("H 0\nREPEAT 3 {\n  TICK\n  H 1\n}\nTICK")
    out = to_stim(_manila(), circuit, tick_ns=100.0)
    assert [b.repeat_count for b in out if isinstance(b, stim.CircuitRepeatBlock)] == [2]
    assert out.flattened() == to_stim(_manila(), circuit.flattened(), tick_ns=100.0)


def _idle_twirl(profile: Profile, physical: int, tick_ns: float) -> list[float]:
    q = profile.table.qubit(physical)
    kraus = thermal_relaxation_kraus(q.t1_ns, q.t2_ns, tick_ns)
    return pauli_twirl([ChannelSpec("thermal_relaxation", (physical,), tuple(kraus))], (physical,))


@pytest.mark.parametrize("layout", [None, [2, 3, 4]], ids=["identity", "shifted"])
def test_idle_noise_at_tick_goes_to_qubits_left_idle_in_that_layer(layout):
    profile, circuit = _manila(), "H 0 1\nTICK\nCX 0 1\nTICK\nH 2\nTICK\nH 0"
    physical = layout or [0, 1, 2]
    plain = str(to_stim(profile, circuit, layout=layout)).split("TICK")
    timed = to_stim(profile, circuit, layout=layout, tick_ns=200.0)
    idle = []
    for with_idle, without in zip(str(timed).split("TICK"), plain, strict=True):
        extra = [line for line in with_idle.splitlines() if line not in without.splitlines()]
        idle.append(sorted(int(q) for line in extra for q in line.rsplit(")", 1)[1].split()))
    assert idle == [[2], [2], [0, 1], []]  # nothing after the last TICK
    names = [inst.name for inst in timed]
    emitted = list(timed)[names.index("TICK") - 1].gate_args_copy()
    assert emitted == pytest.approx(_idle_twirl(profile, physical[2], 200.0))
    assert emitted != pytest.approx(_idle_twirl(profile, 2 if layout else 4, 200.0))
    assert "idle noise" in [a.what for a in timed.report.approximated]
    assert any("idle noise (pass tick_ns=" in o for o in to_stim(profile, circuit).report.omitted)


def test_measured_and_reset_qubits_are_busy_in_their_tick_layer():
    out = to_stim(_manila(), "M 0\nR 1\nMR 2\nTICK\nH 0 1 2 3", readout="none", tick_ns=200.0)
    before_tick = str(out).split("TICK")[0].splitlines()
    assert before_tick[:3] == ["M 0", "R 1", "MR 2"]
    assert [line.rsplit(" ", 1)[1] for line in before_tick[3:]] == ["3"]


# (d) existing noise --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "noisy", ["DEPOLARIZE1(0.01) 0\nM 0", "H 0\nM(0.01) 0", "MPAD(0.3) 0\nM 0"]
)
def test_existing_noise_raises_with_the_fix(noisy):
    with pytest.raises(ExistingNoiseError, match="existing_noise='strip'.*existing_noise='keep'"):
        to_stim(_readout_profile(), noisy)


def test_existing_noise_keep_and_strip():
    profile = _readout_profile()
    kept = to_stim(profile, "DEPOLARIZE1(0.01) 0\nM(0.01) 0", existing_noise="keep")
    assert [inst.name for inst in kept] == ["DEPOLARIZE1", "M"]
    assert list(kept)[1].gate_args_copy() == pytest.approx([0.01 + 0.06 - 2 * 0.01 * 0.06])
    stripped = to_stim(profile, "DEPOLARIZE1(0.01) 0\nM(0.01) 0", existing_noise="strip")
    assert str(stripped) == "M(0.06) 0"
    assert "noise already in the circuit (stripped)" in stripped.report.omitted


def test_noisy_padding_is_kept_or_stripped_like_other_noise():
    profile = _readout_profile()
    assert str(to_stim(profile, "MPAD(0.3) 0 1\nM 0", existing_noise="keep")).startswith(
        "MPAD(0.3) 0 1\n"
    )
    assert str(to_stim(profile, "MPAD(0.3) 0 1\nM 0", existing_noise="strip")) == (
        "MPAD 0 1\nM(0.06) 0"
    )
    assert str(to_stim(profile, "MPAD 1\nM 0")) == "MPAD 1\nM(0.06) 0"


def test_heralded_noise_cannot_be_stripped_but_can_be_kept():
    circuit = "HERALDED_ERASE(0.1) 0\nM 0"
    with pytest.raises(ExistingNoiseError, match="measurement records"):
        to_stim(_readout_profile(), circuit, existing_noise="strip")
    out = to_stim(_readout_profile(), circuit, existing_noise="keep", readout="exact")
    assert out.readout_flips.tolist() == [[0.0, 0.0], [0.02, 0.1]]


# (e) readout and reset -----------------------------------------------------------------------


def test_symmetrized_readout_and_reset_errors_in_each_basis():
    profile = _readout_profile(prep={"error": 0.003})
    out = to_stim(profile, "RX 0\nMX 0\nMR 1\nMPP X0*Z1")
    assert str(out).splitlines() == [
        "RX 0",
        "Z_ERROR(0.003) 0",
        "MX(0.06) 0",
        "MR(0.06) 1",
        "X_ERROR(0.003) 1",
        "MPP(0.1128) X0*Z1",  # (1 - 0.88**2) / 2
    ]
    assert "readout error" in [a.what for a in out.report.approximated]


@pytest.mark.parametrize(
    ("prepare", "measure"),
    [("R", "M"), ("RX", "MX"), ("RY", "MY"), ("MR", "M"), ("MRX", "MX"), ("MRY", "MY")],
)
def test_failed_reset_leaves_the_orthogonal_state_in_every_basis(prepare, measure):
    profile = _readout_profile(prep={"error": 0.2})
    out = to_stim(profile, f"{prepare} 0\n{measure} 0", readout="none")
    shots = 40_000
    ones = out.compile_sampler(seed=2).sample(shots)[:, -1].mean()
    assert abs(ones - 0.2) < 5 * np.sqrt(0.2 * 0.8 / shots)


def test_reset_error_is_the_physical_qubits():
    qubits = [{"index": i, "prep": {"error": e}} for i, e in enumerate([0.01, 0.02, 0.03])]
    profile = Profile.model_validate(toy(qubits=qubits))
    out = to_stim(profile, "R 0 1\nRX 2", layout={0: 2, 1: 0, 2: 1})
    assert str(out).splitlines() == [
        "R 0 1",
        "X_ERROR(0.03) 0",
        "X_ERROR(0.01) 1",
        "RX 2",
        "Z_ERROR(0.02) 2",
    ]


def test_readout_none_adds_nothing_and_reports_it():
    out = to_stim(_readout_profile(), "X 0\nM 0 1", readout="none")
    assert str(out) == "X 0\nM 0 1"
    assert "readout error (readout='none')" in out.report.omitted


def test_unknown_readout_is_reported_not_zeroed_silently():
    out = to_stim(Profile.model_validate(toy()), "M 0 1")
    assert str(out) == "M 0 1"
    assert out.report.unknown == ["readout error of physical qubits 0, 1"]


def test_sample_with_readout_is_exact_for_asymmetric_readout():
    out = to_stim(_readout_profile(), "X 0\nM 0 !1 2", readout="exact")
    assert str(out) == "X 0\nM 0 !1 2"
    bits = sample_with_readout(out, 200_000, seed=3)
    zeros = 1 - bits.mean(axis=0)
    shots = len(bits)
    # prepared 1 reads 0 with P(0|1); an inverted prepared 0 records 0 with P(1|0)
    for observed, p in zip(zeros, [0.1, 0.02, 0.98], strict=True):
        assert abs(observed - p) < 5 * np.sqrt(p * (1 - p) / shots)
    again = sample_with_readout(out, 1000, seed=3)
    assert np.array_equal(again, sample_with_readout(out, 1000, seed=3))


def test_exact_readout_ghz_matches_asymmetric_reference_on_manila():
    profile = _manila()
    out = to_stim(profile, _program(_GHZ5, 5), readout="exact")
    expected = _twirled_reference(profile, _GHZ5, 5, [0, 1, 2, 3, 4], asymmetric=True)
    symmetric = _twirled_reference(profile, _GHZ5, 5, [0, 1, 2, 3, 4], asymmetric=False)
    assert 0.5 * np.abs(expected - symmetric).sum() > 0.01  # the test separates the two modes
    _assert_within_5_sigma(sample_with_readout(out, 200_000, seed=11), expected)


@pytest.mark.parametrize(
    ("circuit", "reason"),
    [("M 0\nCX rec[-1] 1\nM 1", "feeds a measurement"), ("MPP X0*X1", "multi-qubit product")],
)
def test_exact_readout_refuses_what_it_cannot_do_exactly(circuit, reason):
    with pytest.raises(ValueError, match=reason):
        to_stim(_manila(), circuit, readout="exact")


def test_sample_with_readout_needs_an_exact_export():
    with pytest.raises(ValueError, match="readout='exact'"):
        sample_with_readout(to_stim(_manila(), "M 0"), 10)


# result, options, layout ---------------------------------------------------------------------


def test_result_is_a_stim_circuit_with_report_and_profile():
    profile = _manila()
    out = profile.to_stim(stim.Circuit("H 0\nM 0"), layout={0: 3})
    assert isinstance(out, stim.Circuit) and isinstance(out, NoiseVaultStimCircuit)
    assert out.profile is profile and out.layout == {0: 3}
    assert out.report.framework == "stim" and out.report.options["layout"] == {0: 3}
    assert "Pauli twirl" in out.report.summary()


@pytest.mark.parametrize(
    "clone", [lambda c: pickle.loads(pickle.dumps(c)), copy.deepcopy, copy.copy]
)
def test_pickled_and_copied_results_keep_report_and_exact_readout(clone):
    out = to_stim(_readout_profile(), "X 0\nM 0 1", layout={0: 1, 1: 2}, readout="exact")
    again = clone(out)
    assert isinstance(again, NoiseVaultStimCircuit) and str(again) == str(out)
    assert again.report.summary() == out.report.summary()
    assert (again.profile, again.layout) == (out.profile, {0: 1, 1: 2})
    assert np.array_equal(again.readout_flips, out.readout_flips)
    assert np.array_equal(
        sample_with_readout(again, 50, seed=1), sample_with_readout(out, 50, seed=1)
    )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"readout": "asym"}, "readout='asym'"),
        ({"existing_noise": "drop"}, "existing_noise"),
        ({"tick_ns": 0}, "tick_ns"),
        ({"unknown_gates": "ideal"}, "unknown_gates"),
    ],
)
def test_bad_options_say_what_to_pass(options, match):
    with pytest.raises(ValueError, match=match):
        to_stim(_manila(), "H 0", **options)


def test_layout_errors_name_the_qubit():
    with pytest.raises(LayoutError, match="qubit 7"):
        to_stim(_manila(), "H 7")


def test_effects_are_omitted_or_refused():
    effect = {"type": "leakage", "gate": "cz", "prob": 1e-4}
    omitted = to_stim(Profile.model_validate(toy(effects=[effect])), "CZ 0 1")
    assert "effect leakage on cz" in omitted.report.omitted
    refused = Profile.model_validate(toy(effects=[{**effect, "allow": "approximate"}]))
    with pytest.raises(UnsupportedEffect):
        to_stim(refused, "CZ 0 1")


def test_layout_from_coords_places_a_surface_code_on_the_best_grid_patch():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=1)
    profile = _grid(9, worse_rows=3)
    layout = layout_from_coords(circuit, profile)
    edges = set(profile.table.edges())
    pairs = {(t[0].value, t[1].value) for i in circuit.flattened() if i.name == "CX"
             for t in i.target_groups()}  # fmt: skip
    assert all(tuple(sorted((layout[a], layout[b]))) in edges for a, b in pairs)
    assert min(layout.values()) >= 3 * 9  # avoids the noisy top rows
    assert len(set(layout.values())) == len(layout) == 17


def test_layout_from_coords_says_what_to_do_without_coords():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=1)
    with pytest.raises(LayoutError, match="layout="):
        layout_from_coords(circuit, _manila())
    with pytest.raises(LayoutError, match="no rotation or shift"):
        layout_from_coords(circuit, _grid(4))


@pytest.mark.slow
def test_layout_from_coords_is_fast_for_a_distance_11_code_on_a_32x32_grid():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=11, rounds=1)
    profile = _grid(32, worse_rows=5)
    start = time.perf_counter()
    layout = layout_from_coords(circuit, profile)
    elapsed = time.perf_counter() - start
    print(f"layout_from_coords d=11 on 32x32: {elapsed:.3f} s")
    assert len(set(layout.values())) == len(layout) == 241
    assert min(layout.values()) >= 5 * 32
    assert elapsed < 0.5


# (f) scale and (g) detector error model ------------------------------------------------------


@pytest.mark.slow
def test_1024_qubits_by_100_layers_converts_and_samples_in_under_5_seconds():
    n, layers = 1024, 100
    lines = [f"R {' '.join(map(str, range(n)))}", "TICK"]
    for layer in range(layers):
        lines += [f"H {' '.join(map(str, range(n)))}", "TICK"]
        pairs = " ".join(f"{i} {i + 1}" for i in range(layer % 2, n - 1, 2))
        lines += [f"CZ {pairs}", "TICK"]
    circuit = stim.Circuit("\n".join([*lines, f"M {' '.join(map(str, range(n)))}"]))
    profile = Profile.uniform(
        "chain1024",
        technology="superconducting",
        num_qubits=n,
        one_qubit_error=1e-3,
        two_qubit_error=5e-3,
        readout_error=1e-2,
        t1_us=100,
        t2_us=80,
        one_qubit_ns=25,
        two_qubit_ns=40,
        connectivity=[(i, i + 1) for i in range(n - 1)],
    )
    start = time.perf_counter()
    out = to_stim(profile, circuit, tick_ns=50.0)
    built = time.perf_counter()
    bits = out.compile_sampler(seed=1).sample(10_000)
    done = time.perf_counter()
    print(f"convert {built - start:.2f} s, sample 10k {done - built:.2f} s")
    assert bits.shape == (10_000, n)
    assert done - start < 5.0


def test_detector_error_model_builds_and_decodes_for_a_noisy_surface_code():
    pymatching = require("pymatching")
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    profile = _grid(7)
    out = to_stim(profile, circuit, layout=layout_from_coords(circuit, profile), tick_ns=50.0)
    dem = out.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == circuit.num_detectors and dem.num_errors > 100
    matching = pymatching.Matching.from_stim_circuit(out)
    detectors, observed = out.compile_detector_sampler(seed=5).sample(
        20_000, separate_observables=True
    )
    decoded = np.mean(matching.decode_batch(detectors)[:, 0] != observed[:, 0])
    raw = np.mean(observed[:, 0])
    assert 0 < decoded < raw / 3
