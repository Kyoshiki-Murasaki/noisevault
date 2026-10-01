from __future__ import annotations

import json
import warnings

import pytest
from conftest import toy

from noisevault.errors import NoiseApproximationWarning, UnsupportedEffect
from noisevault.profile import Profile
from noisevault.report import HONESTY, Report


def _report(**effects) -> tuple[Profile, Report]:
    profile = Profile.model_validate(toy(**effects))
    report = Report.start(profile, "qiskit", "2.5.2", layout={"a": 0}, unknown_gates="typical")
    return profile, report


def test_effects_are_omitted_by_default() -> None:
    profile, report = _report(effects=[{"type": "leakage", "gate": "cz", "prob": 1e-4}])
    report.record_effects(profile.effects)
    assert report.omitted == ["effect leakage on cz"]


@pytest.mark.parametrize("allow", ["approximate", "exact"])
def test_effects_that_must_be_modeled_refuse_conversion(allow: str) -> None:
    profile, report = _report(
        effects=[{"type": "atom_loss", "on": "readout", "prob": 1e-3, "allow": allow}]
    )
    with pytest.raises(UnsupportedEffect, match="atom_loss"):
        report.record_effects(profile.effects)


def test_warn_once_per_key() -> None:
    _, report = _report()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            report.warn_once("h", "h is not native; typical noise used")
        report.warn_once("ry", "ry is not native; typical noise used")
    assert [w.category for w in caught] == [NoiseApproximationWarning] * 2


def test_report_serializes_and_summarizes() -> None:
    profile, report = _report()
    report.mark_exact("gate errors")
    report.approximate("readout", "symmetrized", "mean of P(1|0) and P(0|1)")
    report.approximate("readout", "symmetrized", "mean of P(1|0) and P(0|1)")
    report.mark_unknown("prep on qubit 0")
    report.count("typical_noise_used", "h", 3)
    report.count("typical_noise_used", "h")
    data = json.loads(json.dumps(report.to_dict()))
    assert data["fingerprint"] == profile.fingerprint
    assert data["options"] == {"layout": {"a": 0}, "unknown_gates": "typical"}
    assert data["events"] == {"typical_noise_used": {"h": 4}}
    assert len(data["approximated"]) == 1
    text = report.summary()
    assert profile.id in text and "typical_noise_used: h=4" in text and text.endswith(HONESTY)


def test_calibration_qualifiers_of_used_gates_are_reported() -> None:
    from noisevault.channels import gate_channels

    gates = toy()["gates"] | {
        "x": {
            "avg_infidelity": 1e-3,
            "scope": "cycle",
            "includes": ["spam", "leakage"],
            "statistic": "median",
            "assumption": "the vendor number is read as average gate fidelity",
        }
    }
    profile, report = _report(gates=gates)
    table = profile.table
    for name in ("x", "sx"):
        report.record_channels(gate_channels(table.gate(name, (0,)), [table.qubit(0)]))
    data = report.to_dict()
    assert {a["what"] for a in data["approximated"]} == {"x error"}
    text = json.dumps(data["approximated"])
    for expected in ("cycle", "spam", "leakage", "median", "read as average gate fidelity"):
        assert expected in text
    assert "x error" in report.summary()
