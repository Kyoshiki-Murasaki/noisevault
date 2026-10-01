from __future__ import annotations

import json
import warnings

import pytest
from conftest import require, toy

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
    assert profile.id in text and text.endswith(HONESTY)
    assert "used: h took the typical native gate's noise 4 times" in text


def test_summary_states_each_count_as_a_sentence() -> None:
    _, report = _report()
    report.count("typical_noise_used", "cx", 40)
    report.count("typical_noise_used", "h")
    report.count("reversed_record_used", "cz", 20)
    report.count("circuit_channel_kept", "BitFlipChannel", 2)
    report.count("future_event", "q3", 5)
    used = [line for line in report.summary().splitlines() if line.startswith("used:")]
    assert used == [
        "used: cx took the typical native gate's noise 40 times;"
        " h took the typical native gate's noise once;"
        " cz took the calibration recorded for the opposite qubit order 20 times;"
        " the circuit's own BitFlipChannel was kept as written, with no noise added, 2 times;"
        " future event: q3 5 times"
    ]
    assert "=" not in used[0]
    assert report.to_dict()["events"]["reversed_record_used"] == {"cz": 20}


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
    for expected in ("cycle", "measurement", "leakage", "median", "read as average gate fidelity"):
        assert expected in text
    assert "x error" in report.summary()


def test_included_errors_are_named_in_plain_words() -> None:
    from noisevault.channels import gate_channels

    gates = toy()["gates"] | {
        "x": {"avg_infidelity": 1e-3, "includes": ["spam", "leakage", "1q_dressing"]}
    }
    profile, report = _report(gates=gates)
    report.record_channels(gate_channels(profile.table.gate("x", (0,)), [profile.table.qubit(0)]))
    approximated = [line for line in report.summary().splitlines() if "x error" in line]
    assert approximated == [
        "approximated: x error: the stated error already includes single-qubit gate error"
        " (explicit single-qubit gates in the circuit add their own error on top)",
        "approximated: x error: the stated error already includes leakage"
        " (applied as depolarizing noise; no population leaves the qubit)",
        "approximated: x error: the stated error already includes state preparation and"
        " measurement error (readout and preparation noise, where applied, add their own error"
        " on top)",
    ]


def _options(**options) -> dict:
    report = Report.start(Profile.model_validate(toy()), "pennylane", None, **options)
    return json.loads(json.dumps(report.to_dict()))["options"]


def test_keys_that_stay_distinct_as_strings_serialize_as_an_object() -> None:
    assert _options(layout={0: 2, 1: 0}) == {"layout": {"0": 2, "1": 0}}
    assert _options(layout={"a": 0, ("b", 1): 1}) == {"layout": {"a": 0, "('b', 1)": 1}}


def test_keys_that_collide_as_strings_serialize_as_pairs() -> None:
    assert _options(layout={0: 0, "0": 1}) == {"layout": [[0, 0], ["0", 1]]}
    nested = _options(extra={"layout": {1: 0, "1": 2}, "names": {"x": 1}})
    assert nested == {"extra": {"layout": [[1, 0], ["1", 2]], "names": {"x": 1}}}


def test_pennylane_report_keeps_integer_and_string_wires_apart() -> None:
    require("pennylane")
    model = Profile.model_validate(toy()).to_pennylane(layout={0: 0, "0": 1})
    data = json.loads(json.dumps(model.report.to_dict()))
    assert data["options"]["layout"] == [[0, 0], ["0", 1]]
