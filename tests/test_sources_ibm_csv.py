from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

import noisevault as nv
from noisevault.table import GateNoise, Unavailable

FIXTURES = Path(__file__).parent / "fixtures" / "ibm"
EAGLE = FIXTURES / "synthetic_eagle_2024.csv"  # 2023-2025 layout: a_b pairs, trailing spaces
HERON = FIXTURES / "synthetic_heron_2026.csv"  # 2026 layout: quoted, partner pairs, Yes/No


def _eagle() -> nv.Profile:
    return nv.from_ibm_csv(EAGLE, device="ibm_example", calibrated_at="2024-10-11T13:37:10Z")


def _heron() -> nv.Profile:
    return nv.from_ibm_csv(HERON, device="ibm_example", calibrated_at="2026-01-06")


def _record(profile: nv.Profile, gate: str, qubits: tuple[int, ...]):
    [record] = [r for r in profile.calibrations if (r.gate, r.qubits) == (gate, qubits)]
    return record


def test_older_layout_qubits_and_readout_convention() -> None:
    profile = _eagle()
    assert profile.id == "ibm_example" and profile.device.num_qubits == 4
    assert profile.device.calibrated_at.isoformat() == "2024-10-11T13:37:10+00:00"
    q1 = profile.table.qubit(1)
    assert q1.t1_ns == pytest.approx(180_000) and q1.t2_ns == pytest.approx(110_000)
    # 'Prob meas1 prep0' is P(1|0) and 'Prob meas0 prep1' is P(0|1)
    assert q1.readout == (0.015, 0.025)
    assert profile.qubits[1].readout.duration_ns == 1300
    assert profile.table.qubit(3).disabled  # Operational false, row listed first


def test_older_layout_packed_ecr_is_directed_with_sentinel() -> None:
    profile = _eagle()
    assert profile.connectivity.directed
    assert set(profile.connectivity.edges) == {(0, 1), (1, 0), (1, 2), (2, 3)}
    reverse = profile.table.gate("ecr", (1, 0))
    assert reverse.avg_infidelity == 0.0072 and reverse.duration_ns == 620
    assert profile.table.gate("ecr", (1, 2)).duration_ns == 560
    assert _record(profile, "ecr", (2, 3)).disabled  # IBM's 2_3:1 (on a disabled qubit, too)
    assert isinstance(profile.table.gate("ecr", (2, 1)), Unavailable)
    assert profile.gates["rz"].virtual
    assert profile.table.gate("sx", (0,)).duration_ns is None  # this layout has no 1q length
    assert profile.provenance.notes == ("Ignored columns: Lab note.",)


def test_2026_layout_partner_packing_symmetric_cz_and_units() -> None:
    profile = _heron()
    assert not profile.connectivity.directed
    assert set(profile.connectivity.edges) == {(0, 1), (1, 2), (2, 3)}
    forward, backward = profile.table.gate("cz", (0, 1)), profile.table.gate("cz", (1, 0))
    assert isinstance(forward, GateNoise) and isinstance(backward, GateNoise)
    assert forward.avg_infidelity == backward.avg_infidelity == 0.0013
    assert forward.duration_ns == 68
    assert _record(profile, "cz", (2, 3)).disabled
    # a partner cell belongs to the row's qubit: row 1 says 2:0.0016, row 2 says 1:0.0017
    assert profile.table.gate("cz", (1, 2)).avg_infidelity == 0.0016
    assert profile.table.gate("cz", (2, 1)).avg_infidelity == 0.0017
    cz_records = [r.qubits for r in profile.calibrations if r.gate == "cz"]
    assert cz_records == [(0, 1), (1, 2), (2, 1), (2, 3)]  # agreeing directions stored once
    assert profile.table.gate("rx", (1,)).avg_infidelity == 0.00017
    assert profile.table.gate("sx", (1,)).duration_ns == 32
    assert _record(profile, "sx", (3,)).disabled  # error 1 on a dead qubit
    assert profile.qubits[0].readout.duration_ns == 2400  # 'Readout length (us)' = 2.4
    assert profile.table.qubit(3).disabled
    assert profile.gates["rz"].virtual
    assert profile.device.calibrated_at.isoformat() == "2026-01-06T00:00:00+00:00"


def test_provenance_hashes_the_file() -> None:
    prov = _heron().provenance
    assert prov.source_kind == "user_file" and prov.data_kind == "measured"
    assert prov.redistributable == "unknown"
    assert prov.source_hash == "sha256:" + hashlib.sha256(HERON.read_bytes()).hexdigest()


def test_unknown_layout_names_the_columns_it_found(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("Qubit,Coherence,Fidelity\n0,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'T1 \\(us\\)'") as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert "'Coherence'" in str(info.value) and "'Fidelity'" in str(info.value)


def test_bad_packed_cell_names_line_and_column(tmp_path: Path) -> None:
    text = HERON.read_text(encoding="utf-8").replace('"2:0.0016;0:0.0013"', '"2=0.0016"')
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="line 3, 'CZ error'"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_long_header_list_is_cut_short(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text(",".join(f"col{i}" for i in range(500)) + "\n" + "1," * 499 + "1\n")
    with pytest.raises(ValueError, match=r"'col11', \.\.\.\)") as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert "'col12'" not in str(info.value) and len(str(info.value)) < 1000


def test_properties_json_passed_as_csv_points_to_the_right_reader() -> None:
    with pytest.raises(ValueError, match="looks like IBM properties JSON.*nv.pull") as info:
        nv.from_ibm_csv(
            FIXTURES / "manila_properties.json", device="ibm_x", calibrated_at="2025-01-01"
        )
    assert len(str(info.value)) < 300


def test_bad_calibrated_at_names_the_parameter() -> None:
    with pytest.raises(ValueError, match="calibrated_at='last week' is not an ISO 8601"):
        nv.from_ibm_csv(EAGLE, device="ibm_example", calibrated_at="last week")


def _edited(tmp_path: Path, source: Path, old: str, new: str) -> Path:
    text = source.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / source.name
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_conflicting_pair_in_one_cell_names_line_and_column(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"1:0.0013","1:68"', '"1:0.0013;1:0.009","1:68"')
    with pytest.raises(ValueError, match=r"line 2, 'CZ error'.*0\.0013.*0\.009"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_a_pair_repeated_with_the_same_value_is_read_once(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"1:0.0013","1:68"', '"1:0.0013;1:0.0013","1:68"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert profile.table.gate("cz", (0, 1)).avg_infidelity == 0.0013


def test_conflicting_pair_across_rows_names_both_lines(tmp_path: Path) -> None:
    path = _edited(tmp_path, EAGLE, "2_3:1,2_3:600", "2_3:1; 1_2:0.05,2_3:600; 1_2:560")
    with pytest.raises(ValueError, match=r"line 5.*ecr on qubits \(1, 2\).*line 4"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_pair_repeated_across_rows_with_the_same_values_is_read_once(tmp_path: Path) -> None:
    path = _edited(tmp_path, EAGLE, "2_3:1,2_3:600", "2_3:1; 1_2:0.008,2_3:600; 1_2:560")
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert _record(profile, "ecr", (1, 2)).avg_infidelity == 0.008


def test_a_qubit_listed_twice_is_an_error(tmp_path: Path) -> None:
    row = "0,200.0,120.0,4.8,-0.30,0.015,0.02,0.01,1300,0.0002,0,0.009,0.0002,,,true,\n"
    path = tmp_path / "twice.csv"
    path.write_text(EAGLE.read_text(encoding="utf-8") + row, encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 6: qubit 0 is already listed on twice.csv line 3"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_qubit_without_readout_stays_unknown(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"0.012","0.016","0.008","2.4"', '"","","","2.4"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert profile.table.qubit(0).readout is None
    assert profile.table.qubit(1).readout == (0.007, 0.013)
    assert profile.readout is None
    assert any("Qubits [0] have no readout" in note for note in profile.provenance.notes)


def test_qubit_without_t1_takes_the_median_and_says_so(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"0","300","250"', '"0","",""')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert profile.table.qubit(0).t1_ns == pytest.approx(270_000)  # qubits 1 and 2; 3 is off
    assert any("Qubits [0] have no T1" in note for note in profile.provenance.notes)


def test_a_blank_gate_error_takes_the_device_median_and_says_so(tmp_path: Path) -> None:
    row_0 = '"0","0.00015","0.00015","1:0.0013"'
    path = _edited(tmp_path, HERON, row_0, '"0","","0.00015","1:0.0013"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    sx = profile.table.gate("sx", (0,))
    assert (sx.state, sx.origin, sx.avg_infidelity) == ("calibrated", "record", 0.000185)
    assert profile.provenance.notes == (
        "Qubits [0] have no sx error; the device median applies to them.",
    )


def test_a_blank_gate_error_keeps_the_published_gate_length(tmp_path: Path) -> None:
    row_0 = '"0.00015","32","0.00015","0","0.00015"'
    path = _edited(tmp_path, HERON, row_0, '"0.00015","80","0.00015","",""')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    sx = profile.table.gate("sx", (0,))
    assert (sx.state, sx.avg_infidelity, sx.duration_ns) == ("calibrated", 0.000185, 80)
    assert profile.table.gate("x", (0,)).duration_ns == 80
    assert profile.gates["sx"].duration_ns == 32
    assert profile.gates["rz"].virtual
    assert profile.table.gate("rz", (0,)).state == "ideal"
    assert profile.provenance.notes == (
        "Qubits [0] have no sx error; the device median applies to them.",
    )


def test_a_disabled_qubit_stays_out_of_the_gate_medians(tmp_path: Path) -> None:
    blank_sx = _edited(tmp_path, HERON, '"0","0.00015","0.00015"', '"0","","0.00015"')
    path = _edited(tmp_path, blank_sx, '"0","1","1","2:1"', '"0","0.1","1","2:1"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert profile.table.gate("sx", (0,)).avg_infidelity == 0.000185  # qubits 1 and 2
    assert _record(profile, "sx", (3,)).avg_infidelity == 0.1


def test_a_disabled_qubit_stays_out_of_the_qubit_medians() -> None:
    profile = _heron()  # qubits 0 to 2 only: T1 300, 280, 260 us; T2 250, 260, 200 us
    assert (profile.idle.t1_us, profile.idle.t2_us) == (280, 250)
    assert (profile.readout.p1_given_0, profile.readout.p0_given_1) == (0.008, 0.016)


@pytest.mark.parametrize(
    ("cells", "label", "shown", "median_ns"),
    [
        ('"0","-1","250"', "T1", "-1", 270_000),
        ('"0","0","250"', "T1", "0", 270_000),
        ('"0","nan","250"', "T1", "nan", 270_000),
        ('"0","300","-250"', "T2", "-250", 230_000),
    ],
)
def test_invalid_coherence_takes_the_median_and_names_the_value(
    tmp_path: Path, cells: str, label: str, shown: str, median_ns: float
) -> None:
    path = _edited(tmp_path, HERON, '"0","300","250"', cells)
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    qubit = profile.table.qubit(0)
    assert (qubit.t1_ns if label == "T1" else qubit.t2_ns) == pytest.approx(median_ns)
    note = f"Qubit 0 reported {label} = {shown} us; treated as missing, the device median applies."
    assert note in profile.provenance.notes
    assert not any(f"have no {label}" in n for n in profile.provenance.notes)


def test_a_device_with_no_valid_t1_is_a_one_line_error(tmp_path: Path) -> None:
    text = HERON.read_text(encoding="utf-8")
    for qubit, t1 in (("0", "300"), ("1", "280"), ("2", "260"), ("3", "240")):
        assert f'"{qubit}","{t1}",' in text
        text = text.replace(f'"{qubit}","{t1}",', f'"{qubit}","0",')
    path = tmp_path / "dead.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"ibm_x reports no valid T1.*qubit 0: T1 = 0 us") as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert "\n" not in str(info.value)


@pytest.mark.parametrize("qubit", ["0.5", "-1", "nan", "inf", "x"])
def test_a_qubit_that_is_not_a_whole_number_names_its_line(tmp_path: Path, qubit: str) -> None:
    path = _edited(tmp_path, HERON, '"0","300"', f'"{qubit}","300"')
    message = f"line 2, 'Qubit': {qubit!r} is not a qubit number"
    with pytest.raises(ValueError, match=re.escape(message)):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_a_qubit_written_as_a_float_is_read_as_that_qubit(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"3","240"', '"3.0","240"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert profile.table.qubit(3).disabled == _heron().table.qubit(3).disabled
    assert profile.table.qubit(3).t1_ns == pytest.approx(240_000)


def test_a_negative_partner_names_its_line_and_column(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"1:0.0013","1:68"', '"-1:0.0013","1:68"')
    with pytest.raises(ValueError, match=r"line 2, 'CZ error': '-1:0.0013' is not"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
