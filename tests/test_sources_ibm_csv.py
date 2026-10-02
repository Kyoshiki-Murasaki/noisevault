from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import re
import statistics
from pathlib import Path

import pytest

import noisevault as nv
from noisevault.table import GateNoise, Unavailable

FIXTURES = Path(__file__).parent / "fixtures" / "ibm"
EAGLE = FIXTURES / "synthetic_eagle_2024.csv"
HERON = FIXTURES / "synthetic_heron_2026.csv"
FALCON = FIXTURES / "synthetic_falcon_2023.csv"


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
    with pytest.raises(nv.SourceDataError, match="no 'T1 \\(us\\)'") as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert info.value.message.endswith("column (found: 'Qubit', 'Coherence', 'Fidelity')")
    assert info.value.hint == (
        "download the calibration CSV from the device's page on the IBM Quantum platform"
    )


@pytest.mark.parametrize("extra", ["SX error", "√x (sx) error"], ids=["alias", "repeated"])
def test_two_columns_for_one_value_name_both_and_import_nothing(tmp_path: Path, extra: str) -> None:
    header, *rows = HERON.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "twice.csv"
    lines = [f'{header},"{extra}"', *(f'{row},"0.01"' for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert info.value.message == (
        f"twice.csv: '√x (sx) error' (column 12) and '{extra}' (column 18) are two columns for"
        " the same value"
    )
    assert info.value.hint == "delete one of them"


def test_repeated_columns_the_reader_ignores_still_import(tmp_path: Path) -> None:
    header, *rows = HERON.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "padded.csv"
    lines = [f'{header},"","","Note","Note"', *(f'{row},"","","a","b"' for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    profile = nv.from_ibm_csv(path, device="ibm_example", calibrated_at="2026-01-06")
    assert profile.fingerprint == _heron().fingerprint


def test_bad_packed_cell_names_line_and_column(tmp_path: Path) -> None:
    text = HERON.read_text(encoding="utf-8").replace('"2:0.0016;0:0.0013"', '"2=0.0016"')
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(nv.SourceDataError, match="line 3, 'CZ error'"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_long_header_list_is_cut_short(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text(",".join(f"col{i}" for i in range(500)) + "\n" + "1," * 499 + "1\n")
    with pytest.raises(nv.SourceDataError, match=r"'col11', \.\.\.\)") as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert "'col12'" not in str(info.value) and len(str(info.value)) < 1000


def test_properties_json_passed_as_csv_points_to_the_right_reader() -> None:
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(
            FIXTURES / "manila_properties.json", device="ibm_x", calibrated_at="2025-01-01"
        )
    assert info.value.message == (
        "manila_properties.json looks like IBM properties JSON, not a calibration CSV"
    )
    assert info.value.hint == (
        "use nv.pull(<name>) for a device, or from_qiskit_backend(<backend>) for a Qiskit backend"
    )


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
    with pytest.raises(nv.SourceDataError, match=r"line 2, 'CZ error'.*0\.0013.*0\.009"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_a_pair_repeated_with_the_same_value_is_read_once(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"1:0.0013","1:68"', '"1:0.0013;1:0.0013","1:68"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert profile.table.gate("cz", (0, 1)).avg_infidelity == 0.0013


def test_conflicting_pair_across_rows_names_both_lines(tmp_path: Path) -> None:
    path = _edited(tmp_path, EAGLE, "2_3:1,2_3:600", "2_3:1; 1_2:0.05,2_3:600; 1_2:560")
    with pytest.raises(nv.SourceDataError, match=r"line 5.*ecr on qubits \(1, 2\).*line 4"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_pair_repeated_across_rows_with_the_same_values_is_read_once(tmp_path: Path) -> None:
    path = _edited(tmp_path, EAGLE, "2_3:1,2_3:600", "2_3:1; 1_2:0.008,2_3:600; 1_2:560")
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert _record(profile, "ecr", (1, 2)).avg_infidelity == 0.008


def test_a_qubit_listed_twice_is_an_error(tmp_path: Path) -> None:
    row = "0,200.0,120.0,4.8,-0.30,0.015,0.02,0.01,1300,0.0002,0,0.009,0.0002,,,true,\n"
    path = tmp_path / "twice.csv"
    path.write_text(EAGLE.read_text(encoding="utf-8") + row, encoding="utf-8")
    with pytest.raises(
        nv.SourceDataError, match=r"line 6: qubit 0 is already listed on twice.csv line 3"
    ):
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
    assert profile.table.qubit(0).t1_ns == pytest.approx(270_000)
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


def test_a_pair_with_a_gate_length_and_no_error_keeps_both(tmp_path: Path) -> None:
    row_0 = _edited(tmp_path, HERON, '"1:0.0013","1:68"', '"","1:100"')
    path = _edited(tmp_path, row_0, '"2:0.0016;0:0.0013","2:68;0:68"', '"2:0.0016","2:68;0:100"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert set(profile.connectivity.edges) == {(0, 1), (1, 2), (2, 3)}
    cz = profile.table.gate("cz", (0, 1))
    median_error = statistics.median([0.0016, 0.0017])
    assert (cz.state, cz.avg_infidelity, cz.duration_ns) == ("calibrated", median_error, 100)
    assert profile.gates["cz"].duration_ns == statistics.median([68, 68, 100, 100])
    assert profile.provenance.notes == (
        "Pairs [(0, 1)] have no cz error; the device median applies to them.",
    )


def test_a_pair_with_an_error_one_way_only_takes_that_record(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"1:0.0013","1:68"', '"","1:100"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    for pair in ((0, 1), (1, 0)):
        cz = profile.table.gate("cz", pair)
        assert (cz.state, cz.avg_infidelity, cz.duration_ns) == ("calibrated", 0.0013, 68)
    assert profile.gates["cz"].duration_ns == 68
    assert profile.provenance.notes == ()


def test_a_disabled_qubit_stays_out_of_the_gate_medians(tmp_path: Path) -> None:
    blank_sx = _edited(tmp_path, HERON, '"0","0.00015","0.00015"', '"0","","0.00015"')
    path = _edited(tmp_path, blank_sx, '"0","1","1","2:1"', '"0","0.1","1","2:1"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert profile.table.gate("sx", (0,)).avg_infidelity == 0.000185
    assert _record(profile, "sx", (3,)).avg_infidelity == 0.1


def test_a_disabled_qubit_stays_out_of_the_qubit_medians() -> None:
    profile = _heron()
    assert profile.table.qubit(3).disabled
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
    with pytest.raises(
        nv.SourceDataError, match=r"ibm_x reports no valid T1.*qubit 0: T1 = 0 us"
    ) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert "\n" not in str(info.value)


def test_an_invalid_t1_on_a_disabled_qubit_counts_as_missing(tmp_path: Path) -> None:
    imported = {}
    for reading in ("0", ""):
        path = _edited(tmp_path, HERON, '"3","240",', f'"3","{reading}",')
        for qubit, t1 in (("0", "300"), ("1", "280"), ("2", "260")):
            path = _edited(tmp_path, path, f'"{qubit}","{t1}",', f'"{qubit}","",')
        imported[reading] = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    zero, blank = imported["0"], imported[""]
    assert zero.idle.t1_us is None and zero.table.qubit(3).disabled
    assert (zero.fingerprint, zero.provenance.notes) == (blank.fingerprint, blank.provenance.notes)


@pytest.mark.parametrize("qubit", ["0.5", "-1", "nan", "inf", "x"])
def test_a_qubit_that_is_not_a_whole_number_names_its_line(tmp_path: Path, qubit: str) -> None:
    path = _edited(tmp_path, HERON, '"0","300"', f'"{qubit}","300"')
    message = f"line 2, 'Qubit': {qubit!r} is not a qubit number"
    with pytest.raises(nv.SourceDataError, match=re.escape(message)):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('"0","300","250"', '"0","300","n/a"', "line 2, 'T2 (us)': 'n/a' is not a number"),
        (
            '"0.012","Yes"',
            '"0.012","Maybe"',
            "line 2, 'Operational': 'Maybe' is not yes/no or true/false",
        ),
    ],
    ids=["number", "flag"],
)
def test_a_cell_the_reader_cannot_read_names_its_line_and_column(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _edited(tmp_path, HERON, old, new)
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert str(info.value) == f"{path.name} {message}"


def test_a_header_with_no_qubit_rows_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(HERON.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert str(info.value) == "empty.csv has a header but no qubit rows"


@pytest.mark.parametrize(
    ("old", "new", "shown"),
    [
        (b'"T1 (us)"', b'"T1 (\xb5s)"', "line 1 has the byte 0xb5"),
        (b'"No"', b'"N\xe3o"', "line 5 has the byte 0xe3"),
    ],
    ids=["latin-1 header", "latin-1 cell"],
)
def test_a_file_that_is_not_utf8_names_the_line_and_the_byte(
    tmp_path: Path, old: bytes, new: bytes, shown: str
) -> None:
    path = tmp_path / "resaved.csv"
    path.write_bytes(HERON.read_bytes().replace(old, new))
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert str(info.value) == (
        f"resaved.csv is not UTF-8 text: {shown}; import the CSV as you downloaded it, not a"
        " copy a spreadsheet saved"
    )


def test_a_file_with_carriage_return_line_ends_reads_like_the_download(tmp_path: Path) -> None:
    path = tmp_path / HERON.name
    path.write_bytes(HERON.read_bytes().replace(b"\n", b"\r"))
    profile = nv.from_ibm_csv(path, device="ibm_example", calibrated_at="2026-01-06")
    assert profile.fingerprint == _heron().fingerprint


def test_a_cell_too_long_for_csv_names_its_line(tmp_path: Path) -> None:
    header, first, *_ = HERON.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "long.csv"
    path.write_text(f'{header}\n{first}\n"{"9" * 200_000}"\n', encoding="utf-8")
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert str(info.value) == (
        "long.csv line 3 is not valid CSV: field larger than field limit (131072)"
    )


@pytest.mark.parametrize(
    ("source", "edits", "message"),
    [
        pytest.param(
            EAGLE,
            [("false,\n", 'false,"\n')],
            "line 2 is not valid CSV: unexpected end of data",
            id="quote never closed",
        ),
        pytest.param(
            EAGLE,
            [("false,\n", 'false,"\n'), ("620,true,\n", '620,true,"\n')],
            "line 2 is not valid CSV: unexpected end of data",
            id="quote closed two lines down",
        ),
        pytest.param(
            HERON,
            [('"0","300"', '"0","300"0')],
            "line 2 is not valid CSV: ',' expected after '\"'",
            id="text after a closing quote",
        ),
    ],
)
def test_a_damaged_quote_names_its_line(
    tmp_path: Path, source: Path, edits: list[tuple[str, str]], message: str
) -> None:
    path = source
    for old, new in edits:
        path = _edited(tmp_path, path, old, new)
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert str(info.value) == f"{source.name} {message}"


_HERON_LAST_ROW = HERON.read_text(encoding="utf-8").splitlines()[-1]


@pytest.mark.parametrize(
    ("old", "new", "shown"),
    [
        pytest.param(_HERON_LAST_ROW, '"3","240"', "line 5 has 2 cells", id="cut short"),
        pytest.param(_HERON_LAST_ROW, '"3"', "line 5 has 1 cell", id="one cell"),
        pytest.param(
            '"0.010","Yes"', '"0.010","Yes","x"', "line 3 has 18 cells", id="one cell too many"
        ),
    ],
)
def test_a_line_with_missing_or_extra_cells_names_it(
    tmp_path: Path, old: str, new: str, shown: str
) -> None:
    path = _edited(tmp_path, HERON, old, new)
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert str(info.value) == f"{HERON.name} {shown}, but the header has 17 columns"


def test_a_line_number_counts_blank_lines(tmp_path: Path) -> None:
    header, *rows = HERON.read_text(encoding="utf-8").replace('"No"', '"Maybe"').splitlines()
    path = tmp_path / HERON.name
    path.write_text("\n".join([header, "", *rows]) + "\n", encoding="utf-8")
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert str(info.value) == (
        f"{HERON.name} line 6, 'Operational': 'Maybe' is not yes/no or true/false"
    )


def _rows(source: Path) -> list[list[str]]:
    return list(csv.reader(io.StringIO(source.read_text(encoding="utf-8"), newline="")))


@pytest.mark.parametrize(
    ("source", "cut"),
    [
        pytest.param(source, cut, id=f"{source.stem}:{header.strip()}")
        for source in (EAGLE, FALCON, HERON)
        for cut, header in enumerate(_rows(source)[0])
    ],
)
def test_a_csv_missing_any_column_imports_or_raises_source_data_error(
    tmp_path: Path, source: Path, cut: int
) -> None:
    path = tmp_path / source.name
    with path.open("w", encoding="utf-8", newline="") as out:
        csv.writer(out).writerows(row[:cut] + row[cut + 1 :] for row in _rows(source))
    with contextlib.suppress(nv.SourceDataError):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")


def test_an_unknown_time_unit_names_its_column(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"T1 (us)"', '"T1 (hours)"')
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert str(info.value) == (
        f"{path.name}: 'T1 (hours)' (column 2) has the unknown time unit 'hours'; expected ns,"
        " us, µs, ms or s"
    )


def test_a_qubit_written_as_a_float_is_read_as_that_qubit(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"3","240"', '"3.0","240"')
    profile = nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")
    assert profile.table.qubit(3).disabled == _heron().table.qubit(3).disabled
    assert profile.table.qubit(3).t1_ns == pytest.approx(240_000)


def test_a_negative_partner_names_its_line_and_column(tmp_path: Path) -> None:
    path = _edited(tmp_path, HERON, '"1:0.0013","1:68"', '"-1:0.0013","1:68"')
    with pytest.raises(nv.SourceDataError, match=r"line 2, 'CZ error': '-1:0.0013' is not"):
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2025-01-01")


def test_undefined_reads_as_a_blank_cell(tmp_path: Path) -> None:
    blank = _edited(tmp_path, FALCON, ",undefined,", ",,")
    undefined, missing = (
        nv.from_ibm_csv(path, device="ibm_example", calibrated_at="2023-04-11")
        for path in (FALCON, blank)
    )
    assert undefined.fingerprint == missing.fingerprint
    assert undefined.provenance.notes == missing.provenance.notes
    assert any("Qubits [2] have no T2" in note for note in undefined.provenance.notes)


def test_rows_with_nothing_the_reader_reads_are_skipped(tmp_path: Path) -> None:
    header, *rows = HERON.read_text(encoding="utf-8").splitlines()
    width = header.count(",") + 1
    lines = [f'{header},"Note"', *(f'{row},""' for row in rows), "," * width + '"x"', "," * width]
    path = tmp_path / "annotated.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    profile = nv.from_ibm_csv(path, device="ibm_example", calibrated_at="2026-01-06")
    assert profile.fingerprint == _heron().fingerprint


def test_a_row_with_values_but_no_qubit_names_its_line(tmp_path: Path) -> None:
    header, *rows = HERON.read_text(encoding="utf-8").splitlines()
    averages = '"","250"' + "," * (header.count(",") - 1)
    path = tmp_path / "averages.csv"
    path.write_text("\n".join([header, *rows, averages]) + "\n", encoding="utf-8")
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert info.value.message == (
        "averages.csv line 6, 'Qubit' is blank in a row with calibration values"
    )
    assert info.value.hint == "give the row its qubit number or delete it"


@pytest.mark.parametrize(
    ("column", "item"),
    [
        pytest.param("CZ error", "01:00.0", id="1:0.0013 as mm:ss.0"),
        pytest.param("CZ error", "01:00", id="1:0.0013 as mm:ss"),
        pytest.param("CZ error", "01:00.001", id="1:0.0013 as mm:ss.000"),
        pytest.param("CZ error", "0:01", id="1:0.0013 as h:mm"),
        pytest.param("CZ error", "0:01:00", id="1:0.0013 as h:mm:ss"),
        pytest.param("CZ error", "00:01:00.00", id="1:0.0013 as hh:mm:ss.00"),
        pytest.param("CZ error", "12:01:00 AM", id="1:0.0013 as h:mm:ss AM/PM"),
        pytest.param("CZ error", "12:01 am", id="1:0.0013 as h:mm am/pm"),
        pytest.param("CZ error", "0.000694459", id="1:0.0013 as a fraction of a day"),
        pytest.param("CZ error", "1.50463E-08", id="0:0.0013 as a fraction of a day"),
        pytest.param("CZ error", "54:00.0", id="114:0.0013 as mm:ss.0"),
        pytest.param("CZ error", "114:00.0", id="114:0.0013 as elapsed mm:ss.0"),
        pytest.param("CZ error", "1:54:00", id="114:0.0013 as h:mm:ss"),
        pytest.param("Gate length (ns)", "0.088888889", id="1:68 as a fraction of a day"),
        pytest.param("Gate length (ns)", "2:08", id="1:68 as h:mm"),
        pytest.param("Gate length (ns)", "02:28", id="1:88 as hh:mm"),
    ],
)
def test_a_pair_a_spreadsheet_turned_into_a_time_is_refused(
    tmp_path: Path, column: str, item: str
) -> None:
    cells = {"CZ error": f'"{item}","1:68"', "Gate length (ns)": f'"1:0.0013","{item}"'}
    path = _edited(tmp_path, HERON, '"1:0.0013","1:68"', cells[column])
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert info.value.message == (
        f"{path.name} line 2, {column!r}: {item!r} looks like a time that a spreadsheet made from"
        " a 'partner:value' cell, so the value is lost"
    )
    assert info.value.hint == "import the CSV as you downloaded it, not a copy a spreadsheet saved"


def test_a_gate_length_that_could_be_minutes_still_imports(tmp_path: Path) -> None:
    row_0 = _edited(tmp_path, HERON, '"1:0.0013","1:68"', '"1:0.0013","1:56"')
    path = _edited(tmp_path, row_0, '"2:68;0:68"', '"2:68;0:56"')
    profile = nv.from_ibm_csv(path, device="ibm_example", calibrated_at="2026-01-06")
    assert profile.table.gate("cz", (0, 1)).duration_ns == 56


_SUPPORTED = "this reader imports only the 2023 to 2026 formats"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            [
                "Qubit,T1 (us),T2 (us),Frequency (GHz),Anharmonicity (GHz),Readout assignment"
                " error ,Prob meas0 prep1 ,Prob meas1 prep0 ,Readout length (ns),ID error ,√x (sx)"
                " error ,Single-qubit Pauli-X error ,CNOT error ,Gate time (ns)",
                "Q0,72,140,5.1,-0.3,0.01,0.02,0.01,3022,3e-4,3e-4,3e-4,0_1:0.01,0_1:412",
                "Q1,54,102,5.2,-0.3,0.01,0.02,0.01,3022,3e-4,3e-4,3e-4,1_0:0.01,1_0:377",
            ],
            "old.csv is in IBM's CSV format from before 2023, which has a 'Single-qubit Pauli-X"
            f" error' column; {_SUPPORTED}",
        ),
        (
            [
                "Qubit,Frequency (GHz),T1 (µs),T2 (µs),Readout assignment error,√x (sx) error,CNOT"
                " error",
                ',4.83,136.7,200.5,0.03,0.00027,"cx0_1: 6.7e-3 "',
                '1,4.62,179.5,110.9,0.021,0.00014,"cx1_2: 9.3e-3 , cx1_0: 6.7e-3 "',
            ],
            "old.csv line 2, 'Qubit' is blank; IBM's CSVs from before 2023 left qubit 0 blank,"
            f" and {_SUPPORTED}",
        ),
    ],
    ids=["q-labels", "blank-qubit-0"],
)
def test_a_layout_from_before_2023_is_refused_in_one_line(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    path = tmp_path / "old.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2022-01-01")
    assert str(info.value) == message


def test_a_value_a_profile_cannot_hold_names_the_file_and_the_value(tmp_path: Path) -> None:
    path = _edited(
        tmp_path, HERON, '"0","300","250","0.012","0.016"', '"0","300","250","0.012","1.5"'
    )
    with pytest.raises(nv.SourceDataError) as info:
        nv.from_ibm_csv(path, device="ibm_x", calibrated_at="2026-01-06")
    assert (info.value.message, info.value.hint) == (
        f"{path.name}: readout.p0_given_1 of qubit 0: Input should be less than or equal to 1,"
        " got 1.5",
        f"correct that value in {path.name}",
    )
