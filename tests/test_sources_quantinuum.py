"""Quantinuum profiles from the published hardware-specification data."""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pytest

from noisevault import Profile
from noisevault.errors import SourceDataError, SourceUnavailable
from noisevault.reference import Op, probabilities
from noisevault.sources import quantinuum

FIXTURE = Path(__file__).parent / "fixtures" / "quantinuum"
DATASET = FIXTURE / "H2-2" / "2024_12_06"
CSV = FIXTURE / "spec_sheet_parameters.csv"


def _files() -> dict[str, bytes]:
    return {name: (DATASET / f"{name}.json").read_bytes() for name in quantinuum.FILES}


@pytest.fixture(scope="module")
def h2_2():
    return quantinuum.from_data("H2-2", "2024_12_06", _files())


def test_raw_data_reproduces_the_published_spec_sheet_row(h2_2) -> None:
    # The repository's CSV row for H2-2 on 2024-12-06: 1Q 8(2)E-05, 1Q leakage 1.2(3)E-05,
    # 2Q 1.4(1)E-03, 2Q leakage 4.3(6)E-04, memory 5.3(4)E-04, SPAM 1.33(9)E-03; its emulator
    # CSV gives p_meas_0 0.0009 and p_meas_1 0.0018 for the same SPAM data.
    gates = h2_2.gates
    assert gates["r"].avg_infidelity == pytest.approx(8e-5, abs=0.5e-5)
    assert gates["zz"].avg_infidelity == pytest.approx(1.4e-3, abs=0.05e-3)
    leakage = {e.gate: e.prob for e in h2_2.effects if e.type == "leakage"}
    assert leakage["r"] == pytest.approx(1.2e-5, abs=0.05e-5)
    assert leakage["zz"] == pytest.approx(4.3e-4, abs=0.05e-4)
    assert h2_2.benchmarks["memory_error_depth1"]["value"] == pytest.approx(5.3e-4, abs=0.05e-4)
    readout = h2_2.readout
    assert (readout.p1_given_0 + readout.p0_given_1) / 2 == pytest.approx(1.33e-3, abs=0.005e-3)
    assert readout.p1_given_0 == pytest.approx(0.0009, abs=0.5e-4)
    assert readout.p0_given_1 == pytest.approx(0.0018, abs=0.5e-4)


def test_profile_states_what_each_number_means(h2_2) -> None:
    assert h2_2.id == "quantinuum_h2-2"
    assert h2_2.device.num_qubits == 56
    assert h2_2.device.technology == "trapped_ion"
    assert h2_2.device.calibrated_at.isoformat() == "2024-12-06T00:00:00+00:00"
    assert h2_2.connectivity == "all_to_all"
    assert h2_2.gates["rz"].virtual
    one, two = h2_2.gates["r"], h2_2.gates["zz"]
    assert (one.method, one.statistic, one.includes) == ("rb", "mean", ("leakage",))
    assert (two.method, two.statistic, two.includes) == ("rb", "mean", ("1q_dressing", "leakage"))
    assert "1.5 ZZ gates per Clifford" in two.assumption
    assert h2_2.prep is None  # combined SPAM holds the preparation error
    assert any("combined SPAM" in note for note in h2_2.provenance.notes)
    assert all(effect.allow == "omit" for effect in h2_2.effects)


def test_provenance_pins_the_exact_source(h2_2) -> None:
    prov = h2_2.provenance
    joined = b"".join(_files()[name] for name in quantinuum.FILES)
    assert prov.source_hash == "sha256:" + hashlib.sha256(joined).hexdigest()
    assert prov.source_url.endswith(f"/tree/{quantinuum.COMMIT}/data/H2-2/2024_12_06")
    assert (prov.data_kind, prov.source_kind) == ("measured", "published_data")
    assert (prov.license, prov.attribution, prov.redistributable) == (
        "Apache-2.0",
        "Quantinuum",
        "yes",
    )


def test_spec_csv_import_agrees_with_the_raw_data(h2_2) -> None:
    from_csv = quantinuum.from_spec_csv(CSV, machine="quantinuum_h2-2", date="2024_12_06")
    for gate in ("r", "zz"):
        spec = from_csv.gates[gate]
        assert spec.stderr is not None
        assert abs(spec.avg_infidelity - h2_2.gates[gate].avg_infidelity) <= spec.stderr
    assert from_csv.readout.error == pytest.approx(1.33e-3)
    assert (
        from_csv.provenance.source_hash == "sha256:" + hashlib.sha256(CSV.read_bytes()).hexdigest()
    )


def test_legacy_csv_rows_have_no_leakage_correction() -> None:
    profile = quantinuum.from_spec_csv(CSV, machine="H1-1", date="2022_06_09")
    assert profile.gates["zz"].avg_infidelity == pytest.approx(2.40e-3)
    assert profile.gates["zz"].includes == ("1q_dressing",)
    assert profile.effects == ()


def test_profile_converts_on_its_natives_without_approximation(h2_2) -> None:
    ops = [Op("r", (0,), (np.pi / 2, 0.0)), Op("zz", (0, 1)), Op("rz", (1,), (0.4,))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a typical-noise fallback would warn
        noisy = probabilities(h2_2, ops, 2)
    perfect = Profile.uniform(
        "perfect", technology="trapped_ion", num_qubits=2, one_qubit_error=0, two_qubit_error=0
    )
    tvd = 0.5 * np.abs(noisy - probabilities(perfect, ops, 2)).sum()
    assert 1e-5 < tvd < 1e-2


def test_decay_rate_recovers_a_known_decay() -> None:
    lengths = np.array([2, 32, 128, 512, 2048], dtype=float)
    means = 0.73 * 0.99871**lengths + 0.25
    assert quantinuum.decay_rate(lengths, means, asymptote=0.25) == pytest.approx(0.99871, abs=1e-9)


@pytest.mark.parametrize(
    ("cell", "value", "stderr"),
    [("2.15(8)E-03", 2.15e-3, 8e-5), ("0(2)E-06", 0.0, 2e-6), ("1.33(9)E-03", 1.33e-3, 9e-5)],
)
def test_spec_sheet_cells_parse_value_and_uncertainty(cell: str, value: float, stderr: float):
    parsed = quantinuum.parse_cell(cell)
    assert parsed.value == pytest.approx(value, abs=1e-15)
    assert parsed.stderr == pytest.approx(stderr)


def test_bad_inputs_say_what_is_known() -> None:
    assert quantinuum.parse_cell(" ") is None
    with pytest.raises(SourceDataError, match="2.15"):
        quantinuum.parse_cell("0.002")
    with pytest.raises(ValueError, match="H1-1, H1-2, H2-1, H2-2, REIMEI"):
        quantinuum.from_repository("H3-1")
    with pytest.raises(ValueError, match="known: 2024_12_06, 2025_05_29, 2025_08_28"):
        quantinuum.from_repository("H2-2", "2025_01_01")
    with pytest.raises(
        SourceDataError, match=r"Dates \(YYYY_MM_DD\) it has for H2-1: 2023_03_10, 2024_05_20$"
    ):
        quantinuum.from_spec_csv(CSV, machine="H2-1", date="2025_04_30")


def test_spec_csv_accepts_iso_dates() -> None:
    iso = quantinuum.from_spec_csv(CSV, machine="H2-2", date="2024-12-06")
    underscored = quantinuum.from_spec_csv(CSV, machine="H2-2", date="2024_12_06")
    assert iso.fingerprint == underscored.fingerprint


def test_spec_csv_with_a_repeated_column_names_both(tmp_path: Path) -> None:
    header, *rows = CSV.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "twice.csv"
    lines = [f"{header},2Q error", *(f"{row},9.9(1)E-02" for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")
    assert info.value.message == f"{path}: columns 6 and 13 are both '2Q error'"
    assert info.value.hint == "delete one of them"


def test_spec_csv_of_a_date_with_no_known_qubit_count_asks_for_it(tmp_path: Path) -> None:
    header, *rows = CSV.read_text(encoding="utf-8").splitlines()
    [row] = [r for r in rows if r.startswith("2024_12_06,H2-2,")]
    path = tmp_path / "later.csv"
    path.write_text(f"{header}\n{row.replace('2024_12_06', '2030_01_01')}\n", encoding="utf-8")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2030_01_01")
    assert info.value.message == "the qubit count of H2-2 on 2030_01_01 is unknown"
    assert info.value.hint == "pass num_qubits="
    assert quantinuum.from_spec_csv(path, machine="H2-2", date="2030_01_01", num_qubits=56)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (",1.33(9)E-03", ",", "H2-2 2024_12_06: the CSV row has no SPAM error"),
        (
            ",8(2)E-05,7(2)E-05,",
            ",,,",
            "H2-2 2024_12_06: the CSV row has none of 1Q error, 1Q error (legacy)",
        ),
    ],
    ids=["no-spam", "no-1q"],
)
def test_spec_csv_row_missing_a_value_names_it(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    header, *rows = CSV.read_text(encoding="utf-8").splitlines()
    [row] = [r for r in rows if r.startswith("2024_12_06,H2-2,")]
    assert old in row
    path = tmp_path / "damaged.csv"
    path.write_text(f"{header}\n{row.replace(old, new)}\n", encoding="utf-8")
    with pytest.raises(SourceDataError) as info:
        quantinuum.from_spec_csv(path, machine="H2-2", date="2024_12_06")
    assert str(info.value) == message


def test_a_failed_download_says_what_loads_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = []

    def offline(request: urllib.request.Request, timeout: float) -> None:
        requested.append(request.full_url)
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(quantinuum.urllib.request, "urlopen", offline)
    with pytest.raises(SourceUnavailable) as info:
        quantinuum.from_repository("H2-2")
    assert info.value.message == f"could not download {requested[0]} (timed out)"
    assert info.value.hint == (
        "check the network connection, or run nv list to see every profile you can load offline"
    )


def test_bundle_takes_the_newest_dataset_of_every_machine(monkeypatch) -> None:
    fetched = []
    monkeypatch.setattr(quantinuum, "from_repository", lambda m: fetched.append(m) or m)
    assert quantinuum.bundled_profiles() == ["H1-1", "H1-2", "H2-1", "H2-2", "REIMEI"]
    newest = {m: max(d for mm, d in quantinuum.QUBITS if mm == m) for m in fetched}
    assert newest == {
        "H1-1": "2025_05_02",
        "H1-2": "2023_08_21",
        "H2-1": "2025_04_30",
        "H2-2": "2025_08_28",
        "REIMEI": "2025_06_18",
    }


@pytest.mark.network
def test_download_matches_the_fixture(h2_2) -> None:
    live = quantinuum.from_repository("H2-2", "2024_12_06")
    assert live.fingerprint == h2_2.fingerprint
    assert live.provenance.source_hash == h2_2.provenance.source_hash
