from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import require

import noisevault as nv
from noisevault.errors import install_hint

pa = require("pyarrow")
pq = require("pyarrow.parquet")

FIXTURE = Path(__file__).parent / "fixtures" / "hf_archive" / "train-00000-of-00001.parquet"
DATASET_URL = "https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift"
REVISION = "3a9423f67c5f1af40b180ee381370b2588e4efad"


def _fez(at: str | None = None, path: Path = FIXTURE) -> nv.Profile:
    return nv.from_calibration_archive(path, "ibm_fez", at=at)


def _qubit(profile: nv.Profile, index: int):
    [record] = [q for q in profile.qubits if q.index == index]
    return record


def _records(profile: nv.Profile, gate: str) -> dict[tuple[int, ...], object]:
    return {r.qubits: r for r in profile.calibrations if r.gate == gate}


def test_lists_each_device_from_its_first_snapshot_to_its_newest_calibration() -> None:
    spans = nv.calibration_archive_devices(FIXTURE)
    assert spans == {
        "ibm_fez": (
            datetime(2026, 1, 31, 17, 5, 3, tzinfo=UTC),
            datetime(2026, 6, 2, 9, 0, 14, 800000, tzinfo=UTC),
        ),
        "ibm_torino": (
            datetime(2026, 1, 31, 17, 5, 3, 496046, tzinfo=UTC),
            datetime(2026, 4, 1, 12, 56, 1, tzinfo=UTC),
        ),
    }
    assert spans["ibm_torino"].last == datetime(2026, 4, 1, 12, 56, 1, tzinfo=UTC)


def test_default_takes_the_newest_calibration_of_each_property() -> None:
    profile = _fez()
    assert (profile.id, profile.device.num_qubits) == ("ibm_fez", 4)
    assert profile.device.processor == "Heron r2"
    assert profile.device.calibrated_at == datetime(2026, 6, 2, 9, 0, 14, 800000, tzinfo=UTC)
    assert _qubit(profile, 0).t1_us == 120.0
    assert _qubit(profile, 1).t1_us == 101.0
    assert _qubit(profile, 2).t2_us == 90.0
    assert _records(profile, "cz")[(0, 1)].avg_infidelity == 0.002


def test_at_between_two_calibrations_takes_the_earlier_one() -> None:
    profile = _fez(at="2026-06-01T20:00:00Z")
    assert profile.device.calibrated_at == datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    assert _qubit(profile, 0).t1_us == 100.0
    assert _records(profile, "cz")[(0, 1)].avg_infidelity == 0.003


def test_at_equal_to_a_calibration_time_includes_it() -> None:
    assert _qubit(_fez(at="2026-06-02T08:00:00Z"), 0).t1_us == 120.0


def test_the_calibration_time_decides_not_the_time_the_archive_saw_it() -> None:
    profile = _fez(at="2026-06-02T09:00:05Z")
    assert _qubit(profile, 2).t2_us == 82.0
    assert profile.device.calibrated_at == datetime(2026, 6, 2, 8, 0, tzinfo=UTC)


def test_legacy_rows_give_t1_and_t2_in_seconds() -> None:
    profile = _fez(at="2026-03-01")
    assert _qubit(profile, 0).t1_us == pytest.approx(48.54)
    assert _qubit(profile, 1).t2_us == pytest.approx(31.0)
    assert profile.idle.t1_us == pytest.approx(50.04)
    assert profile.device.calibrated_at == datetime(2026, 1, 31, 13, 38, 30, tzinfo=UTC)


def test_a_legacy_profile_mixes_in_backfill_rows_and_names_the_gates_it_lacks() -> None:
    profile = _fez(at="2026-03-01")
    assert profile.gates["cz"].avg_infidelity == 0.0439
    assert profile.gates["cz"].duration_ns == 68.0
    assert profile.gates["x"].avg_infidelity == 4e-04
    assert "rzz" not in profile.gates and "id" not in profile.gates
    assert (
        "This profile has no id or rzz gate. The archive calibrates id and rzz on ibm_fez only"
        " after 2026-03-01T00:00:00Z."
    ) in profile.provenance.notes
    assert not [n for n in profile.provenance.notes if n.startswith("Every row")]


def test_a_profile_from_legacy_rows_only_says_what_they_lack() -> None:
    profile = nv.from_calibration_archive(FIXTURE, "ibm_torino")
    assert sorted(profile.gates) == ["cz", "sx"]
    assert _qubit(profile, 1).t1_us == pytest.approx(49.54)
    assert profile.device.processor == "Heron r1"
    assert (
        "The archive recorded every row behind this profile before 8 May 2026, when the archive"
        " began to record units. Those rows hold only T1, T2, the readout errors and the sx and cz"
        " errors. As a result, this profile has no gate or readout durations and no other gates."
    ) in profile.provenance.notes


def test_a_dead_gate_is_disabled() -> None:
    profile = _fez()
    assert _records(profile, "sx")[(2,)].disabled
    assert profile.gates["sx"].avg_infidelity == pytest.approx(2.05e-04)


def test_a_readout_stuck_at_one_disables_the_qubit_and_says_so() -> None:
    profile = _fez()
    assert _qubit(profile, 3).disabled
    assert not _qubit(_fez(at="2026-06-01T20:00:00Z"), 3).disabled
    assert (
        "Qubit 3 is disabled: its readout calibration gives prob_meas1_prep0 = 1."
        in profile.provenance.notes
    )


def test_pair_directions_collapse_only_when_they_agree() -> None:
    cz = _records(_fez(), "cz")
    assert (1, 0) not in cz
    assert cz[(1, 2)].avg_infidelity == 0.004
    assert cz[(2, 1)].avg_infidelity == 0.006


def _stale(profile: nv.Profile) -> list[str]:
    return [n for n in profile.provenance.notes if n.startswith("IBM calibrated these values")]


def _rewritten(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / FIXTURE.name
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(FIXTURE)), path)
    return path


def test_a_note_names_values_calibrated_more_than_7_days_before_at() -> None:
    exactly = nv.from_calibration_archive(FIXTURE, "ibm_torino", at="2026-04-08T12:56:01Z")
    later = nv.from_calibration_archive(FIXTURE, "ibm_torino", at="2026-04-08T12:56:02Z")
    assert _stale(exactly) == []
    assert _stale(later) == [
        "IBM calibrated these values more than 7 days before 2026-04-08T12:56:02Z, the oldest on"
        " 2026-04-01: T1 on qubits 0 and 1; T2 on qubits 0 and 1; cz on qubits 0-1; also readout"
        " and sx."
    ]
    assert later.fingerprint == exactly.fingerprint


def test_the_note_lists_the_oldest_values_first() -> None:
    assert _stale(_fez(at="2026-03-01")) == [
        "IBM calibrated these values more than 7 days before 2026-03-01T00:00:00Z, the oldest on"
        " 2026-01-29: cz on qubits 0-1, 1-2 and 2-3; readout on qubits 0, 1, 2 and 3; sx on qubits"
        " 0, 1, 2 and 3; also x, T1 and T2."
    ]


def test_without_at_the_note_counts_back_from_the_newest_calibration(tmp_path: Path) -> None:
    rows = pq.read_table(FIXTURE).to_pylist()
    june = datetime(2026, 6, 1, 8, tzinfo=UTC)
    t1 = next(r for r in rows if r["property"] == "T1" and r["calibrated_time"] == june)
    newest = max(rows, key=lambda r: r["calibrated_time"])
    newest["calibrated_time"] = datetime(2026, 6, 8, 8, 0, 1, tzinfo=UTC)
    rows += [{**t1, "qubit_a": q} for q in range(4, 40)]
    assert _stale(_fez(path=_rewritten(tmp_path, rows))) == [
        "IBM calibrated these values more than 7 days before the newest calibration, the oldest"
        " on 2026-06-01: T1 on qubits 1, 2, 3 and 36 more; T2 on qubits 0, 1 and 3; cz on qubits"
        " 0-1, 1-2 and 2-3; also id, prep, readout, rzz, sx and x."
    ]


def test_the_note_leaves_out_rows_the_profile_does_not_read(tmp_path: Path) -> None:
    rows = pq.read_table(FIXTURE).to_pylist()
    for row in rows:
        unread = row["property"].startswith(("measure", "rz_"))
        if row["calibrated_time"] == datetime(2026, 6, 1, 8, tzinfo=UTC) and not unread:
            row["calibrated_time"] = datetime(2026, 6, 1, 12, tzinfo=UTC)
    profile = _fez(at="2026-06-08T08:00:01Z", path=_rewritten(tmp_path, rows))
    assert "Not converted: measure_2." in profile.provenance.notes
    assert _stale(profile) == []


def test_dynamic_circuit_variants_are_not_converted() -> None:
    profile = _fez()
    assert "measure_2" not in profile.gates
    assert "Not converted: measure_2." in profile.provenance.notes


def test_provenance_credits_the_dataset_and_ibm_and_hashes_the_file() -> None:
    prov = _fez().provenance
    assert prov.source_hash == "sha256:" + hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert (prov.data_kind, prov.source_kind) == ("measured", "published_data")
    assert prov.license == "CC-BY-4.0"
    assert prov.attribution == "IBM Quantum, via phanerozoic/qiskit-calibration-drift"
    assert prov.source == (
        "Hugging Face dataset phanerozoic/qiskit-calibration-drift (CC-BY-4.0),"
        " converted by NoiseVault"
    )
    assert prov.source_url == DATASET_URL
    assert prov.redistributable == "unknown"
    assert prov.extra == {}


def test_a_hub_snapshot_path_pins_the_revision(tmp_path: Path) -> None:
    data = tmp_path / "snapshots" / REVISION / "data"
    data.mkdir(parents=True)
    copy = data / FIXTURE.name
    shutil.copy(FIXTURE, copy)
    profile = _fez(path=copy)
    prov = profile.provenance
    assert prov.extra == {"revision": REVISION}
    assert prov.source_url == f"{DATASET_URL}/blob/{REVISION}/data/{FIXTURE.name}"
    assert f"revision {REVISION[:12]}" in prov.source
    assert profile.fingerprint == _fez().fingerprint


def test_never_reads_the_noncommercial_sunspot_column(tmp_path: Path) -> None:
    copy = tmp_path / FIXTURE.name
    shutil.copy(FIXTURE, copy)
    file = pq.ParquetFile(copy)
    chunk = file.metadata.row_group(0).column(file.schema_arrow.names.index("SN"))
    start = chunk.dictionary_page_offset or chunk.data_page_offset
    with copy.open("r+b") as f:
        f.seek(start)
        f.write(b"\xff" * chunk.total_compressed_size)
    with pytest.raises(OSError):
        pq.read_table(copy)
    assert nv.calibration_archive_devices(copy) == nv.calibration_archive_devices(FIXTURE)
    assert _fez(path=copy).fingerprint == _fez().fingerprint


def test_device_names_ignore_case_and_spaces() -> None:
    assert nv.from_calibration_archive(FIXTURE, " IBM_Fez ").fingerprint == _fez().fingerprint


def test_an_unknown_device_names_the_ones_the_file_holds() -> None:
    with pytest.raises(nv.SourceDataError) as caught:
        nv.from_calibration_archive(FIXTURE, "ibm_fes")
    assert (
        caught.value.message
        == f"{FIXTURE.name} has no rows for ibm_fes. The file has rows for ibm_fez, ibm_torino"
    )
    assert caught.value.hint == "did you mean 'ibm_fez'?"


def test_a_time_before_the_first_snapshot_is_refused() -> None:
    with pytest.raises(nv.SourceDataError) as caught:
        _fez(at="2026-01-31")
    assert caught.value.message == (
        f"{FIXTURE.name} has no complete ibm_fez calibration before 2026-01-31T17:05:03Z,"
        " when the archive first recorded ibm_fez"
    )
    assert caught.value.hint == "pass an at= time on or after that time"


LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:93e10633e2fb4b7d3e8bb0f4c0b8d7cdb2d0a0e3c6d5f4a0b1c2d3e4f5a6870d\n"
    "size 65668453\n"
)


@pytest.mark.parametrize("text", [LFS_POINTER, "backend,property\nibm_fez,T1\n"])
def test_a_file_that_is_not_parquet_points_to_the_download(tmp_path: Path, text: str) -> None:
    path = tmp_path / "train-00000-of-00001.parquet"
    path.write_text(text)
    with pytest.raises(nv.SourceDataError) as caught:
        nv.from_calibration_archive(path, "ibm_fez")
    assert caught.value.message.startswith(f"{path.name} is not a readable parquet file: ")
    assert "hf download phanerozoic/qiskit-calibration-drift" in caught.value.hint
    assert "Git LFS" in caught.value.hint


def test_a_parquet_file_without_the_calibration_columns_names_them(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    table = pq.read_table(FIXTURE, columns=["backend", "property", "qubit_a", "value"])
    pq.write_table(table, path)
    with pytest.raises(nv.SourceDataError) as caught:
        nv.calibration_archive_devices(path)
    assert caught.value.message == (
        "other.parquet is not a phanerozoic/qiskit-calibration-drift data file: it has no"
        " qubit_b, unit, calibrated_time or observed_time column"
    )


def test_a_column_of_the_wrong_type_is_named(tmp_path: Path) -> None:
    path = tmp_path / "strings.parquet"
    table = pq.read_table(FIXTURE)
    times = table["calibrated_time"].cast(pa.string())
    column = table.schema.get_field_index("calibrated_time")
    pq.write_table(table.set_column(column, "calibrated_time", times), path)
    with pytest.raises(nv.SourceDataError) as caught:
        nv.from_calibration_archive(path, "ibm_fez")
    assert caught.value.message == (
        "strings.parquet: column calibrated_time holds string, expected a timestamp"
    )


def test_without_pyarrow_the_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(nv.SourceUnavailable) as caught:
        nv.from_calibration_archive(FIXTURE, "ibm_fez")
    assert caught.value.hint == install_hint("hf")


def test_importing_noisevault_does_not_import_pyarrow() -> None:
    code = "import sys, noisevault; print('pyarrow' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


_ON_QUBIT_0 = {
    "a T1 in minutes": (
        "T1",
        {"unit": "min"},
        "T1 of qubit 0 has the unknown time unit 'min', not one of ns, us, µs, ms or s",
    ),
    "a readout error above 1": (
        "prob_meas0_prep1",
        {"value": 1.5},
        "readout.p0_given_1 of qubit 0: Input should be less than or equal to 1, got 1.5",
    ),
}


@pytest.mark.parametrize("case", list(_ON_QUBIT_0))
def test_a_value_it_cannot_read_names_the_rows_and_the_value(tmp_path: Path, case: str) -> None:
    prop, change, problem = _ON_QUBIT_0[case]
    rows = pq.read_table(FIXTURE).to_pylist()
    for row in rows:
        if (row["backend"], row["property"], row["qubit_a"]) == ("ibm_fez", prop, 0):
            row.update(change)
    path = _rewritten(tmp_path, rows)
    with pytest.raises(nv.SourceDataError) as info:
        _fez(path=path)
    assert (info.value.message, info.value.hint) == (
        f"the ibm_fez rows of {path.name}: {problem}",
        "pass an earlier at= to use an older calibration",
    )


def test_a_newer_time_it_cannot_read_names_the_newest_rows(tmp_path: Path) -> None:
    rows = pq.read_table(FIXTURE).to_pylist()
    for row in rows:
        newest_t1 = (row["property"], row["calibrated_time"].day, row["qubit_a"]) == ("T1", 2, 0)
        if row["backend"] == "ibm_fez" and newest_t1:
            row["unit"] = "min"
    path = _rewritten(tmp_path, rows)
    assert _qubit(_fez(at="2026-06-01T20:00:00Z"), 0).t1_us == 100.0
    with pytest.raises(nv.SourceDataError) as info:
        _fez(at="2026-06-01T20:00:00Z", path=path)
    assert (info.value.message, info.value.hint) == (
        f"the newest ibm_fez rows of {path.name}: T1 of qubit 0 has the unknown time unit 'min',"
        " not one of ns, us, µs, ms or s",
        None,
    )


_NOT_A_LOCUS = {
    "a fractional qubit_b": (
        "cz_gate_error",
        "qubit_b",
        2.5,
        "qubit_b of the cz_gate_error row calibrated at 2026-06-02T08:00:00Z is 2.5, not a qubit"
        " index",
    ),
    "a qubit_b that is not a number": (
        "cz_gate_error",
        "qubit_b",
        float("nan"),
        "qubit_b of the cz_gate_error row calibrated at 2026-06-02T08:00:00Z is nan, not a qubit"
        " index",
    ),
    "a null qubit_a": (
        "T1",
        "qubit_a",
        None,
        "qubit_a of the T1 row calibrated at 2026-06-01T08:00:00Z is null, not a qubit index",
    ),
    "a qubit_b on a value of one qubit": (
        "T1",
        "qubit_b",
        2.0,
        "qubit_b of the T1 row calibrated at 2026-06-01T08:00:00Z is 2.0, but T1 is a value of one"
        " qubit",
    ),
}


@pytest.mark.parametrize("case", list(_NOT_A_LOCUS))
def test_a_row_whose_qubits_are_not_a_locus_names_the_row(tmp_path: Path, case: str) -> None:
    prop, column, value, problem = _NOT_A_LOCUS[case]
    rows = pq.read_table(FIXTURE).to_pylist()
    fez = [r for r in rows if (r["backend"], r["property"], r["qubit_a"]) == ("ibm_fez", prop, 1)]
    max(fez, key=lambda r: r["calibrated_time"])[column] = value
    with pytest.raises(nv.SourceDataError) as info:
        _fez(path=_rewritten(tmp_path, rows))
    assert (info.value.message, info.value.hint) == (
        f"the ibm_fez rows of {FIXTURE.name}: {problem}",
        "pass an earlier at= to use an older calibration",
    )


def test_rows_with_no_property_or_calibration_time_are_named(tmp_path: Path) -> None:
    rows = pq.read_table(FIXTURE).to_pylist()
    fez = [r for r in rows if r["backend"] == "ibm_fez"]
    fez[0]["property"] = None
    fez[1]["calibrated_time"] = None
    profile = _fez(path=_rewritten(tmp_path, rows))
    assert (
        "This profile does not use 2 rows of ibm_fez with no property or no calibrated_time."
        in profile.provenance.notes
    )
