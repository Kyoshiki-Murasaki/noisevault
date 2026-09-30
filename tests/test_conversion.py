from __future__ import annotations

import warnings

import pytest
from conftest import toy

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
