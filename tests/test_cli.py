from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from conftest import MANILA_V01, require, toy
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.errors import SourceUnavailable, install_hint
from noisevault.profile import Profile

runner = CliRunner()


def test_validate_accepts_a_0_1_file_and_reports_the_migration() -> None:
    result = runner.invoke(app, ["validate", str(MANILA_V01)])
    assert result.exit_code == 0, result.output
    assert "ok: ibm_manila" in result.stdout
    assert "warning: upgraded a NoiseVault 0.1 file" in result.stderr


def test_validate_strict_fails_on_warnings() -> None:
    result = runner.invoke(app, ["validate", "--strict", str(MANILA_V01)])
    assert result.exit_code == 1


def test_validate_lists_every_error(tmp_path: Path) -> None:
    data = toy(calibrations=[{"gate": "ecr", "qubits": [0, 1]}, {"gate": "cz", "qubits": [0]}])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    errors = [line for line in result.stderr.splitlines() if line.startswith("error: ")]
    assert len(errors) == 2 and any("not defined in gates" in line for line in errors)
    assert "Traceback" not in result.output


def test_validate_missing_file_is_a_friendly_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.json")])
    assert result.exit_code == 1 and "error: no file" in result.stderr


@pytest.mark.parametrize("damage", ["truncated", "corrupt", "directory"])
def test_validate_unreadable_input_is_a_friendly_error(tmp_path: Path, damage: str) -> None:
    packed = gzip.compress(json.dumps(toy()).encode())
    path = tmp_path / "bad.json.gz"
    if damage == "truncated":
        path.write_bytes(packed[:5])
    elif damage == "corrupt":
        path.write_bytes(packed[:10] + b"\xff" * 40)
    else:
        path.mkdir()
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert result.stderr.startswith(f"error: cannot read {path}")
    assert "Traceback" not in result.output and "Aborted" not in result.output


def test_validate_notes_t2_clamps(tmp_path: Path) -> None:
    path = tmp_path / "t2.json"
    path.write_text(json.dumps(toy(idle={"t1_us": 50, "t2_us": 150})))
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0 and "T2 exceeds 2*T1" in result.stderr


def test_validate_says_what_an_unknown_or_a_missing_key_means(tmp_path: Path) -> None:
    data = toy(unknown_field=1)
    del data["device"]
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(data))
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "error: device: missing; format 1.0 requires it",
        "error: unknown_field: not a format 1.0 key; put your own data under extensions",
    ]


# profile display ------------------------------------------------------------------------------


def test_profile_prints_as_one_line() -> None:
    profile = nv.load("ibm_fez")
    expected = f"<Profile ibm_fez@2025-02-26 superconducting 156q {profile.short_fingerprint}>"
    assert repr(profile) == str(profile) == expected
    undated = Profile.model_validate(toy())
    assert (
        repr(undated)
        == f"<Profile test_toy@undated superconducting 3q {undated.short_fingerprint}>"
    )


# global behavior ------------------------------------------------------------------------------


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.stdout == f"noisevault {nv.__version__}\n"


@pytest.mark.parametrize("columns", [80, 120])
@pytest.mark.parametrize(
    "args",
    [
        ["list"],
        ["show", "ibm_fez", "--qubits", "0,1,2"],
        ["show", "google_rainbow"],
        ["diff", "ibm_kyiv", "ibm_brisbane"],
        ["check", "ibm_manila", "--framework", "cirq,stim"],
    ],
    ids=["list", "show-qubits", "show-notes", "diff", "check"],
)
def test_output_fits_the_terminal(args: list[str], columns: int) -> None:
    if args[0] == "check":
        require("cirq"), require("stim")
    result = runner.invoke(app, args, env={"COLUMNS": str(columns)})
    assert result.exit_code == 0, result.output
    assert max(len(line) for line in result.stdout.splitlines()) <= columns


@pytest.mark.parametrize("columns", [80, 120])
def test_list_keeps_every_cell_whole_with_a_pulled_profile(monkeypatch, columns: int) -> None:
    _serve_ionq(monkeypatch)
    assert runner.invoke(app, ["pull", "ionq_forte-1"]).exit_code == 0
    result = runner.invoke(app, ["list"], env={"COLUMNS": str(columns)})
    lines = result.stdout.splitlines()
    assert max(map(len, lines)) <= columns
    group = lines.index(next(line for line in lines if line.strip() == "trapped_ion"))
    first = lines[group + 1].split()
    assert first == [
        "*",
        "ionq_forte-1",
        "2026-09-27",
        "4",
        "Forte",
        "public",
        "API",
        "IonQ",
        "EULA",
    ]
    assert lines[group + 2].split()[:3] == ["quantinuum_h1-1", "2025-05-02", "20"]
    assert "* in your vault" in result.stdout


def test_diff_lists_disabled_gates_without_splitting_an_item(tmp_path: Path) -> None:
    data = nv.load("ibm_fez").model_dump(mode="json", exclude_none=True)
    data["device"]["calibrated_at"] = "2026-10-01T00:00:00Z"
    for record in data["calibrations"]:
        if {27, 28, 71, 72, 129, 130, 153, 154} & set(record["qubits"]):
            record["disabled"] = True
    path = tmp_path / "fez_now.json"
    after = Profile.model_validate(data)
    after.save(path)
    expected = set(nv.load("ibm_fez").diff(after).newly_disabled)
    out = runner.invoke(app, ["diff", "ibm_fez", str(path)], env={"COLUMNS": "80"}).stdout
    lines = out.splitlines()
    section = lines[lines.index("newly disabled") + 1 :]
    shown, gate = set(), None
    for line in section:
        if not line.startswith("  "):
            break
        tokens = line.replace(",", " ").split()
        if not re.fullmatch(r"\d+(-\d+)?", tokens[0]):
            gate, tokens = tokens[0], tokens[1:]
        shown |= {f"{gate} {locus}" for locus in tokens}
    assert len(expected) > 20 and shown == expected


def test_a_profile_id_that_names_a_folder_here_still_loads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ibm_manila").mkdir()
    result = runner.invoke(app, ["show", "ibm_manila"])
    assert result.exit_code == 0 and result.stdout.startswith("ibm_manila@2024-05-27")


def test_no_color_removes_color_codes() -> None:
    args = ["diff", "ibm_kyiv", "ibm_brisbane"]
    colored = runner.invoke(app, args, env={"FORCE_COLOR": "1"}).stdout
    plain = runner.invoke(app, args, env={"FORCE_COLOR": "1", "NO_COLOR": "1"}).stdout
    color = re.compile(r"\x1b\[[0-9;]*3[0-7]m")
    assert color.search(colored) and not color.search(plain)


@pytest.mark.parametrize(
    ("args", "error", "hint"),
    [
        (["show", "ibm_fezz"], "did you mean ibm_fez?", None),
        (["show", "ibm_fez@2020-01-01"], "you have ibm_fez@2025-02-26T20:16:25Z", None),
        (["show", "missing.json"], "error: no file missing.json", "check the path"),
        (["cite", "ibm fez"], "is not a profile id", None),
        (
            ["check", "ibm_manila", "--framework", "qiskt"],
            "'qiskt': did you mean qiskit? give one or more of qiskit",
            None,
        ),
        (["list", "--tech", "photonics"], "choose from superconducting, trapped_ion", None),
        (["show", "ibm_fez", "--qubits", "0,200"], "has qubits 0 to 155", None),
        (["check", "ibm_manila", "--framework", ""], "--framework '': give one or more", None),
        (["check", "ibm_manila", "--framework", ","], "--framework ',': give one or more", None),
        (["show", "./"], "./ is a folder; give a profile file", None),
    ],
)
def test_expected_failures_print_an_error_and_no_traceback(args, error, hint) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error: ") and error in result.stderr
    if hint:
        assert "hint: " in result.stderr and hint in result.stderr
    else:
        assert "hint: " not in result.stderr
    assert "Traceback" not in result.output and result.stdout == ""


# list and show --------------------------------------------------------------------------------


def test_list_groups_profiles_by_technology() -> None:
    result = runner.invoke(app, ["list"], env={"COLUMNS": "80"})
    lines = [line.rstrip() for line in result.stdout.splitlines()]
    assert lines[0].split() == ["id", "date", "qubits", "processor", "source"]
    row = next(line for line in lines if line.strip().startswith("ibm_fez "))
    assert row.split() == ["ibm_fez", "2025-02-26", "156", "Heron", "r2", "package"]
    assert lines.index("superconducting") < lines.index(row) < lines.index("trapped_ion")
    assert lines[-2:] == [
        f"{len(nv.profiles())} profiles, all Apache-2.0. See one with `nv show <id>`.",
        'Load one in Python with nv.load("<id>").',
    ]


def test_list_filters_ignore_case_and_dashes() -> None:
    for args in (["--tech", "Trapped-Ion"], ["--vendor", "Quantinuum"]):
        result = runner.invoke(app, ["list", *args, "--json"])
        assert result.exit_code == 0 and {r["vendor"] for r in json.loads(result.stdout)} == {
            "quantinuum"
        }


def test_list_filters_and_prints_json() -> None:
    result = runner.invoke(app, ["list", "--tech", "trapped_ion", "--json"])
    rows = json.loads(result.stdout)
    assert rows and {r["technology"] for r in rows} == {"trapped_ion"}
    assert {"id", "date", "num_qubits", "processor", "source_kind", "license"} <= set(rows[0])
    vendor = json.loads(runner.invoke(app, ["list", "--vendor", "google", "--json"]).stdout)
    assert {r["id"] for r in vendor} == {"google_rainbow", "google_weber"}


@pytest.mark.parametrize("columns", [60, 72, 80, 120])
def test_list_keeps_every_row_on_one_line(columns: int) -> None:
    infos = nv.profiles()
    result = runner.invoke(app, ["list"], env={"COLUMNS": str(columns)})
    lines = result.stdout.splitlines()
    assert max(map(len, lines)) <= columns
    assert len(lines) == 1 + len({i.technology for i in infos}) + len(infos) + 2
    rows = {tuple(line.split()[:3]) for line in lines if line.startswith("  ")}
    assert rows == {(i.id, i.calibrated_at.date().isoformat(), str(i.num_qubits)) for i in infos}
    header = ["id", "date", "qubits", "processor"] + (["source"] if columns >= 72 else [])
    assert lines[0].split() == header


def test_list_shortens_processors_before_any_other_cell() -> None:
    lines = runner.invoke(app, ["list"], env={"COLUMNS": "50"}).stdout.splitlines()
    assert max(map(len, lines)) <= 50 and lines[0].split() == ["id", "date", "qubits", "processor"]
    row = next(line for line in lines if "quantinuum_h1-1 " in line)
    assert row.split()[:3] == ["quantinuum_h1-1", "2025-05-02", "20"]
    assert row.rstrip().endswith("System M\u2026")


def test_show_prints_a_card() -> None:
    profile = nv.load("ibm_fez")
    out = runner.invoke(app, ["show", "ibm_fez"]).stdout
    assert out.startswith(f"ibm_fez@2025-02-26T20:16:25Z  {profile.short_fingerprint}\n")
    for text in (
        "ibm, Heron r2, superconducting, 156 qubits",
        "176 edges, 1 to 3 neighbors per qubit",
        "rz (1q)",
        "virtual",
        "176 (7 disabled)",
        "median T1 144.9 us",
        "redistributable yes",
        profile.fingerprint,
    ):
        assert text in out
    cz = next(line for line in out.splitlines() if "cz (2q)" in line)
    assert cz.split()[-8:] == ["cz", "(2q)", "3.82e-03", "84", "ns", "176", "(7", "disabled)"]


def test_show_qubits_and_json() -> None:
    data = json.loads(runner.invoke(app, ["show", "ibm_fez", "--qubits", "0,5", "--json"]).stdout)
    assert [q["qubit"] for q in data["qubits"]] == [0, 5]
    table = nv.load("ibm_fez").table
    assert data["qubits"][1]["t1_us"] == pytest.approx(table.qubit(5).t1_ns / 1000)
    assert data["connectivity"] == {
        "kind": "edges",
        "edges": 176,
        "directed": False,
        "min_degree": 1,
        "max_degree": 3,
    }
    cz = next(n for n in data["natives"] if n["gate"] == "cz")
    assert cz["records"] == 176 and cz["disabled"] == 7
    text = runner.invoke(app, ["show", "ibm_fez", "--qubits", "0,5"]).stdout
    assert re.search(r"^\s+5\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+ \(sx\)", text, re.M)
    assert data["qubits"][1]["gate_1q"] == "sx"


@pytest.mark.parametrize("ref", ["ibm_fez", "ibm_kyiv", "ibm_manila"])
def test_show_qubits_names_a_real_gate_not_the_identity(ref: str) -> None:
    text = runner.invoke(app, ["show", ref, "--qubits", "0,1,2,3,4"]).stdout
    rows = text[text.index("1q avg infidelity") :].splitlines()[1:]
    assert len(rows) == 5 and all(row.endswith("(sx)") for row in map(str.rstrip, rows))


def test_show_all_to_all_and_notes() -> None:
    out = runner.invoke(app, ["show", "quantinuum_h1-1"], env={"COLUMNS": "200"}).stdout
    assert "all-to-all" in out and "device-wide" in out
    assert "T1 and T2 unknown" in out and re.search(r"not modeled\s+leakage on r\s", out)


def test_show_and_diff_name_the_error_metric(tmp_path: Path) -> None:
    lines = runner.invoke(app, ["show", "ibm_fez"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert max(map(len, lines)) <= 80
    header = next(line for line in lines if line.startswith("natives"))
    assert header.split()[1:] == ["gate", "median", "avg", "infidelity", "duration", "records"]
    reset = next(line for line in lines if "reset (1q)" in line)
    assert reset.split() == ["reset", "(1q)", "-", "1.58", "us", "156"]
    gates = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-3}, "cz": {"duration_ns": 70}}
    calibrations = [{"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02}]
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(gates=gates, calibrations=calibrations)))
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "120"}).stdout
    cz = next(line for line in out.splitlines() if "cz (2q)" in line)
    assert cz.split()[2:] == ["2.00e-02", "70", "ns", "1", "(1", "without", "error)"]
    drift = runner.invoke(app, ["diff", "ibm_kyiv", "ibm_brisbane"], env={"COLUMNS": "80"}).stdout
    medians = [line.split("  ")[0] for line in drift.splitlines()[1:7]]
    assert medians == [
        "device median",
        "T1 (us)",
        "T2 (us)",
        "1q avg infidelity",
        "2q avg infidelity",
        "readout error",
    ]
    assert re.search(r"^\d+-\d+ +2q avg infidelity ", drift, re.M)


def test_show_says_where_the_data_came_from_in_plain_words(tmp_path: Path) -> None:
    def provenance(ref: str) -> str:
        out = runner.invoke(app, ["show", ref], env={"COLUMNS": "200"}).stdout
        return next(line for line in out.splitlines() if line.startswith("provenance")).rstrip()

    assert provenance("ibm_fez") == (
        "provenance    measured, package snapshot, qiskit-ibm-runtime 0.49.0 FakeFez"
    )
    assert provenance("quantinuum_h1-1") == (
        "provenance    measured, published data, Quantinuum hardware specifications,"
        " data/H1-1/2025_05_02 at 59e68bb55bd6"
    )
    uniform = Profile.uniform(
        "u", technology="trapped_ion", num_qubits=2, one_qubit_error=1e-4, two_qubit_error=1e-3
    ).save(tmp_path / "u.json")
    assert provenance(str(uniform)) == "provenance    hypothetical, written by hand"


def test_show_lists_what_the_model_leaves_out_before_provenance_and_notes() -> None:
    out = runner.invoke(app, ["show", "google_weber"], env={"COLUMNS": "120"}).stdout
    labels = [line.split("  ")[0] for line in out.splitlines()[1:] if line[:1].isalpha()]
    assert labels == [
        "device",
        "connectivity",
        "natives",
        "coherence",
        "readout",
        "not modeled",
        "provenance",
        "license",
        "attribution",
        "fingerprint",
        "assumptions",
        "notes",
    ]


def _manila_in_the_vault(calibrated_at: str) -> Profile:
    data = nv.load("ibm_manila").model_dump(mode="json", exclude_none=True)
    data["device"]["calibrated_at"] = calibrated_at
    data["qubits"][0]["t1_us"] *= 1.1
    profile = Profile.model_validate(data)
    path = nv.catalog.vault_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.save(path)
    return profile


def test_show_says_which_calibration_a_bare_id_picked(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    assert "calibrations" not in runner.invoke(app, ["show", "ibm_manila"]).stdout
    newer = _manila_in_the_vault("2024-06-03T10:00:00Z")
    lines = runner.invoke(app, ["show", "ibm_manila"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert lines[0] == f"ibm_manila@2024-06-03T10:00:00Z  {newer.short_fingerprint}"
    assert [line.rstrip() for line in lines[2:4]] == [
        "calibrations  newest of the 2 you have",
        "              ibm_manila@2024-05-27T18:27:23Z is the bundled one",
    ]
    assert nv.load("ibm_manila@2024-05-27T18:27:23Z").fingerprint == bundled.fingerprint
    for ref in ("ibm_manila@2024-05-27", "ibm_manila@2024-06-03T10:00:00Z"):
        assert "calibrations" not in runner.invoke(app, ["show", ref]).stdout


def test_show_counts_the_calibrations_when_the_bundled_one_is_the_newest(vault: Path) -> None:
    _manila_in_the_vault("2024-01-02T10:00:00Z")
    lines = runner.invoke(app, ["show", "ibm_manila"], env={"COLUMNS": "80"}).stdout.splitlines()
    assert lines[0].startswith("ibm_manila@2024-05-27T18:27:23Z")
    assert lines[2].rstrip() == "calibrations  newest of the 2 you have"
    assert lines[3].startswith("connectivity")


def test_check_says_which_calibration_a_bare_id_picked(vault: Path) -> None:
    require("cirq")
    newer = _manila_in_the_vault("2024-06-03T10:00:00Z")
    out = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq"]).stdout
    assert out.startswith(f"ibm_manila (newest of 2) {newer.short_fingerprint} on qubits ")
    dated = runner.invoke(app, ["check", "ibm_manila@2024-06-03", "--framework", "cirq"]).stdout
    assert dated.startswith(f"ibm_manila {newer.short_fingerprint} on qubits ")


# pull ---------------------------------------------------------------------------------------


def test_pull_saves_to_the_vault_and_prints_the_card(monkeypatch, vault: Path) -> None:
    from noisevault.sources import ibm_public

    pulled = nv.load("ibm_fez")
    monkeypatch.setattr(ibm_public, "pull", lambda device, at=None: pulled)
    result = runner.invoke(app, ["pull", "ibm_fez"])
    assert result.exit_code == 0, result.output
    (saved,) = vault.glob("*.json.gz")
    assert f"saved: {saved}" in result.stdout and "ibm_fez@2025-02-26" in result.stdout
    assert nv.load(str(saved)).fingerprint == pulled.fingerprint
    assert result.stdout.splitlines()[-1] == f"saved: {saved}"


def test_pull_of_a_calibration_already_in_the_vault_says_nothing_was_written(
    monkeypatch, vault: Path
) -> None:
    from noisevault.sources import ibm_public

    monkeypatch.setattr(ibm_public, "pull", lambda device, at=None: nv.load("ibm_fez"))
    runner.invoke(app, ["pull", "ibm_fez"])
    (saved,) = vault.glob("*.json.gz")
    written = saved.stat().st_mtime_ns
    again = runner.invoke(app, ["pull", "ibm_fez"])
    assert again.exit_code == 0, again.output
    assert again.stdout.splitlines()[-1] == f"already saved: {saved}"
    assert saved.stat().st_mtime_ns == written


def _serve_ionq(monkeypatch: pytest.MonkeyPatch) -> None:
    """IonQ's public endpoint, answered from the recorded responses in the test fixture."""
    from noisevault.sources import ionq

    path = Path(__file__).parent / "fixtures" / "ionq" / "responses.json"
    bodies = {url: json.dumps(body).encode() for url, body in json.loads(path.read_text()).items()}
    monkeypatch.setattr(ionq, "_get", bodies.__getitem__)


def test_pull_at_works_for_every_source_its_help_names(monkeypatch) -> None:
    from typer.main import get_command

    (at,) = [p for p in get_command(app).commands["pull"].params if p.name == "at"]
    assert all(source in at.help for source in ("IBM public", "IBM account", "IonQ"))
    _serve_ionq(monkeypatch)
    result = runner.invoke(app, ["pull", "ionq_forte-1", "--at", "2026-09-01"])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("ionq_forte-1@2026-09-01")  # without --at: 2026-09-27


def test_pull_to_a_file(monkeypatch, tmp_path: Path) -> None:
    from noisevault.sources import ionq

    monkeypatch.setattr(ionq, "pull", lambda device, at=None: nv.load("quantinuum_h1-1"))
    target = tmp_path / "forte.json"
    result = runner.invoke(app, ["pull", "ionq_forte-1", "-o", str(target)])
    assert result.exit_code == 0 and target.exists() and f"saved: {target}" in result.stdout


def test_pull_network_failure_gives_one_piece_of_advice(monkeypatch) -> None:
    import urllib.error
    import urllib.request

    def offline(*args, **kwargs):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    result = runner.invoke(app, ["pull", "ibm_fez", "--at", "2025-01-01"])
    assert result.exit_code == 1
    assert result.stderr.startswith("error: could not reach IBM's public endpoint (timed out)")
    assert result.stderr.count("check the network") == 1 and "hint:" not in result.stderr
    assert "Traceback" not in result.output


def test_pull_errors_name_flags_not_python_arguments(monkeypatch) -> None:
    from noisevault.sources import ibm_account, ibm_public

    def no_account(device, at=None):
        raise SourceUnavailable(
            f"could not open your IBM Quantum account (no token); {ibm_account.SETUP}"
        )

    def not_listed(url):
        raise ibm_public._NotFound(url)

    monkeypatch.setattr(ibm_account, "pull", no_account)
    result = runner.invoke(app, ["pull", "ibm_fez", "--source", "ibm-account"])
    assert result.exit_code == 1
    assert "to pull without an account use --source ibm" in result.stderr
    assert "source=" not in result.stderr and "hint:" not in result.stderr
    monkeypatch.setattr(ibm_public, "fetch", not_listed)
    result = runner.invoke(app, ["pull", "ibm_fez"])
    assert "try `nv show ibm_fez` for the bundled snapshot or --source ibm-account" in result.stderr


def test_pull_checks_the_output_folder_before_fetching(monkeypatch, tmp_path: Path) -> None:
    from noisevault.sources import ibm_public

    calls = []
    monkeypatch.setattr(ibm_public, "pull", lambda device, at=None: calls.append(device))
    result = runner.invoke(app, ["pull", "ibm_fez", "-o", str(tmp_path / "no" / "x.json")])
    assert result.exit_code == 1 and calls == []
    assert f"the folder {tmp_path / 'no'} does not exist" in result.stderr


def test_pull_of_an_unsupported_vendor_says_which_sources_exist() -> None:
    result = runner.invoke(app, ["pull", "rigetti_ankaa-3"])
    assert result.exit_code == 1 and "IonQ devices" in result.stderr


# diff, check, cite, schema, doctor --------------------------------------------------------------


def test_diff_prints_tables_and_json() -> None:
    result = runner.invoke(app, ["diff", "ibm_kyiv", "ibm_brisbane", "--top", "2"])
    assert result.exit_code == 0
    out = result.stdout
    assert out.startswith("ibm_kyiv@2025-02-26 -> ibm_brisbane@2025-02-26")
    assert "largest changes by qubit" in out and "largest changes by pair" in out
    assert "warning: these are different devices" in result.stderr
    data = json.loads(runner.invoke(app, ["diff", "ibm_kyiv", "ibm_brisbane", "--json"]).stdout)
    assert data == nv.load("ibm_kyiv").diff(nv.load("ibm_brisbane")).to_dict()


def test_every_device_median_leaves_out_a_disabled_qubit(tmp_path: Path) -> None:
    def device(disabled_t1_us: float) -> Profile:
        qubits = [
            {"index": 0, "t1_us": 300, "readout": {"error": 0.012}},
            {"index": 1, "t1_us": 280, "readout": {"error": 0.01}},
            {"index": 2, "t1_us": 260, "readout": {"error": 0.014}},
            {"index": 3, "t1_us": disabled_t1_us, "readout": {"error": 0.2}, "disabled": True},
        ]
        data = toy(qubits=qubits)
        data["device"]["num_qubits"] = 4
        return Profile.from_dict(data)

    profile = device(240)
    before, after = profile.save(tmp_path / "a.json"), device(1000).save(tmp_path / "b.json")
    shown = json.loads(runner.invoke(app, ["show", str(before), "--json"]).stdout)
    assert (shown["median_t1_us"], shown["median_readout_error"]) == (280, 0.012)
    summary = profile.summary().splitlines()
    assert summary[-2:] == ["  median T1 280 us", "  median readout error 0.012"]
    drift = json.loads(runner.invoke(app, ["diff", str(before), str(after), "--json"]).stdout)
    medians = {m["metric"]: (m["before"], m["after"]) for m in drift["medians"]}
    assert medians["t1_us"] == (280, 280) and medians["readout_error"] == (0.012, 0.012)


def test_diff_of_identical_profiles() -> None:
    result = runner.invoke(app, ["diff", "ibm_fez", "ibm_fez@2025-02-26"])
    assert result.exit_code == 0 and "No change" in result.stdout


def test_check_prints_a_table_per_framework() -> None:
    require("cirq"), require("pennylane")
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq,pennylane"])
    assert result.exit_code == 0, result.output
    rows = {line.split()[0]: line.split() for line in result.stdout.splitlines() if line}
    assert rows["cirq"][1] == "pass" and rows["pennylane"][1] == "pass"
    assert "qiskit" not in rows
    assert "not a measure of how well the model matches the hardware" in " ".join(
        result.stdout.split()
    )


def test_check_failure_exits_1(monkeypatch) -> None:
    require("cirq")
    from noisevault.frameworks import cirq as nv_cirq

    monkeypatch.setattr(nv_cirq.NoiseVaultNoiseModel, "_noise", lambda self, *a: [])
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq"])
    assert result.exit_code == 1 and "FAIL" in result.stdout
    assert "A pass means" not in result.stdout


def test_check_json_and_skips() -> None:
    require("cirq")
    result = runner.invoke(app, ["check", "quantinuum_h1-1", "--framework", "stim,cirq", "--json"])
    data = json.loads(result.stdout)
    assert [f["framework"] for f in data["frameworks"]] == ["cirq"]
    assert data["skipped"][0]["framework"] == "stim"
    assert result.exit_code == 0


def test_check_counts_a_reduced_circuit_apart_and_names_its_missing_gates(tmp_path) -> None:
    require("stim")
    errors = {"h": 1e-3, "ms": 0.02, "rxx": 0.01, "ryy": 0.01, "zz": 0.01, "rzz": 0.01}
    natives = {name: {"avg_infidelity": error} for name, error in errors.items()}
    path = tmp_path / "ions.json"
    path.write_text(json.dumps(toy(gates=natives, readout={"error": 0.01})))
    result = runner.invoke(app, ["check", str(path), "--framework", "stim"])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.stdout.splitlines() if line.startswith("stim "))
    assert "4 of 5, 1 reduced" in row
    assert (
        "stim: two_qubit_natives ran without rxx, ryy, rzz: Stim simulates only Clifford gates,"
        " and this profile's rxx gate is not Clifford" in " ".join(result.stdout.split())
    )
    data = json.loads(
        runner.invoke(app, ["check", str(path), "--framework", "stim", "--json"]).stdout
    )
    (entry,) = data["frameworks"][0]["not_run"]
    assert entry["ran_without"] == ["rxx", "ryy", "rzz"]


def test_check_names_one_whole_install_command_for_the_missing_frameworks(monkeypatch) -> None:
    require("cirq")
    monkeypatch.setitem(sys.modules, "pennylane", None)
    monkeypatch.setitem(sys.modules, "stim", None)
    result = runner.invoke(
        app, ["check", "ibm_manila", "--framework", "cirq,pennylane,stim"], env={"COLUMNS": "80"}
    )
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert f"To add the missing frameworks: {install_hint('pennylane,stim')}" in lines
    assert result.stdout.count("pip install") == 1
    rows = {line.split()[0]: line for line in lines if line}
    assert rows["stim"].split()[1:] == ["not", "installed"]


def test_cite() -> None:
    profile = nv.load("ibm_fez")
    text = runner.invoke(app, ["cite", "ibm_fez"]).stdout
    assert text == (
        "IBM Quantum, via qiskit-ibm-runtime. Calibration of ibm_fez, 2025-02-26T20:16:25Z."
        f" qiskit-ibm-runtime 0.49.0 FakeFez. NoiseVault {nv.__version__} profile"
        f" ibm_fez@2025-02-26T20:16:25Z, fingerprint sha256:{profile.fingerprint}.\n"
    )
    bibtex = runner.invoke(app, ["cite", "ibm_fez", "--bibtex"]).stdout
    assert bibtex == (
        "@misc{nv_ibm_fez_2025_02_26,\n"
        "  title = {{Calibrated noise of ibm\\_fez at 2025-02-26T20:16:25Z}},\n"
        "  author = {{IBM Quantum}},\n"
        "  year = {2025},\n"
        f"  howpublished = {{NoiseVault {nv.__version__} profile"
        f" ibm\\_fez@2025-02-26T20:16:25Z, sha256:{profile.fingerprint}}},\n"
        "  note = {Retrieved via qiskit-ibm-runtime."
        " Source: qiskit-ibm-runtime 0.49.0 FakeFez; license Apache-2.0}\n"
        "}\n"
    )


def test_cite_names_a_ref_that_loads_the_cited_calibration(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    _manila_in_the_vault("2024-06-03T10:00:00Z")
    for style in ([], ["--bibtex"]):
        text = runner.invoke(app, ["cite", "ibm_manila@2024-05-27", *style]).stdout
        ref = re.search(rf"NoiseVault {re.escape(nv.__version__)} profile (\S+),", text)
        assert ref, text
        assert nv.load(ref[1].replace("\\_", "_")).fingerprint == bundled.fingerprint


def _special_characters_profile(tmp_path: Path) -> Path:
    data = toy(
        provenance={
            "attribution": "R&D Lab (via my_tool #2)",
            "source": "50% of run_7 {raw}",
            "license": "CC0-1.0",
        }
    )
    data["device"]["calibrated_at"] = "2025-02-26T09:12:00Z"
    path = tmp_path / "special.json"
    path.write_text(json.dumps(data))
    return path


def test_cite_bibtex_escapes_latex_specials(tmp_path: Path) -> None:
    path = _special_characters_profile(tmp_path)
    fingerprint = nv.load(str(path)).fingerprint
    bibtex = runner.invoke(app, ["cite", str(path), "--bibtex"]).stdout
    assert bibtex == (
        "@misc{nv_test_toy_2025_02_26,\n"
        "  title = {{Calibrated noise of toy at 2025-02-26T09:12:00Z}},\n"
        "  author = {{R\\&D Lab}},\n"
        "  year = {2025},\n"
        f"  howpublished = {{NoiseVault {nv.__version__} profile"
        f" test\\_toy@2025-02-26T09:12:00Z, sha256:{fingerprint}}},\n"
        "  note = {Retrieved via my\\_tool \\#2."
        " Source: 50\\% of run\\_7 \\{raw\\}; license CC0-1.0}\n"
        "}\n"
    )


def _latex_command() -> list[list[str]] | None:
    if shutil.which("pdflatex") and shutil.which("bibtex"):
        return [["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"]] + [
            ["pdflatex", "-interaction=nonstopmode", "main.tex"]
        ] * 2
    if shutil.which("tectonic"):
        return [["tectonic", "-X", "compile", "--keep-intermediates", "main.tex"]]
    return None


def test_cite_bibtex_compiles_with_latex(tmp_path: Path) -> None:
    commands = _latex_command()
    if commands is None:
        pytest.skip("neither pdflatex with bibtex nor tectonic is installed")
    refs = [
        runner.invoke(app, ["cite", ref, "--bibtex"]).stdout
        for ref in ("ibm_fez", "google_rainbow", str(_special_characters_profile(tmp_path)))
    ]
    (tmp_path / "refs.bib").write_text("\n".join(refs))
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\nocite{*}\n"
        "\\bibliographystyle{plain}\n\\bibliography{refs}\n\\end{document}\n"
    )
    for command in commands:
        done = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
    bbl = (tmp_path / "main.bbl").read_text()
    # unbraced, "IBM Quantum, via qiskit-ibm-runtime" prints as "via qiskit-ibm-runtime IBM Quantum"
    for author in ("IBM Quantum", "R\\&D Lab"):
        assert f"\n{{{author}}}.\n" in bbl
    assert "2025-02-26T20:16:25Z" in bbl  # the braced title keeps its capitals


def test_schema_is_the_format_json_schema() -> None:
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0 and json.loads(result.stdout) == nv.json_schema()


def test_doctor_lists_frameworks_and_the_vault(vault: Path) -> None:
    for module in ("qiskit", "cirq", "pennylane", "stim"):
        require(module)
    out = runner.invoke(app, ["doctor"]).stdout
    for package in ("qiskit", "cirq-core", "pennylane", "stim"):
        line = next(line for line in out.splitlines() if line.split()[:1] == [package])
        assert line.split()[1] == version(package)
    assert f"vault: {vault} (0 profiles)" in out
    assert f"bundled profiles: {len(nv.catalog.bundled_profiles())}" in out


def test_doctor_gives_a_whole_install_command_for_missing_frameworks(monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    import noisevault.cli as cli

    def without_stim(package: str) -> str:
        if package == "stim":
            raise PackageNotFoundError(package)
        return "1.0"

    monkeypatch.setattr(cli, "version", without_stim)
    out = runner.invoke(app, ["doctor"], env={"COLUMNS": "80"}).stdout
    assert f"To add the missing frameworks: {install_hint('stim')}" in out.splitlines()


def test_check_with_no_framework_installed_is_one_error_and_one_install_command(
    monkeypatch,
) -> None:
    for module in ("qiskit", "cirq", "pennylane", "stim"):
        monkeypatch.setitem(sys.modules, module, None)
    result = runner.invoke(app, ["check", "ibm_manila"], env={"COLUMNS": "80"})
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr.splitlines() == [
        "error: nv check needs a framework to check, and none is installed",
        f"hint: {install_hint('qiskit')}",
        "      (or cirq, pennylane or stim, or several, as in noisevault[qiskit,stim])",
    ]


def test_check_of_a_named_framework_that_is_not_installed_installs_that_one(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cirq", None)
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq"])
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr.splitlines() == [
        "error: nv check needs a framework to check, and cirq is not installed",
        f"hint: {install_hint('cirq')}",
    ]


def test_check_states_what_a_pass_means_only_after_a_pass() -> None:
    require("stim")
    result = runner.invoke(app, ["check", "quantinuum_h1-1", "--framework", "stim"])
    assert result.exit_code == 1 and "A pass means" not in result.stdout
    assert "stim skipped: Stim simulates only Clifford gates" in result.stdout
    assert result.stderr == "error: no framework could run the check\n"


# bad input never prints a traceback -----------------------------------------------------------

_DAMAGED = {
    "not-json": ("{not json", "is not JSON (Expecting property name"),
    "empty": ("", "is not JSON (Expecting value"),
    "legacy-empty": ('{"schema_version": "0.1"}', "not a valid NoiseVault 0.1 file: provider"),
    "legacy-null-gates": (
        json.dumps({**json.loads(MANILA_V01.read_text()), "gates": None}),
        "gates should be a list, not null; fix that field",
    ),
    "invalid": (json.dumps(toy(gates=None)), "is not a valid profile (1 problem); run `nv valid"),
}
_COMMANDS = {
    "show": lambda f: ["show", f],
    "cite": lambda f: ["cite", f],
    "check": lambda f: ["check", f, "--framework", "stim"],
    "diff-before": lambda f: ["diff", f, "ibm_manila"],
    "diff-after": lambda f: ["diff", "ibm_manila", f],
    "validate": lambda f: ["validate", f],
}


@pytest.mark.parametrize("command", list(_COMMANDS))
@pytest.mark.parametrize("damage", list(_DAMAGED))
def test_every_command_names_a_damaged_file_and_what_is_wrong(
    tmp_path: Path, command: str, damage: str
) -> None:
    text, expected = _DAMAGED[damage]
    path = tmp_path / f"{damage}.json"
    path.write_text(text)
    result = runner.invoke(app, _COMMANDS[command](str(path)), env={"COLUMNS": "80"})
    assert result.exit_code == 1 and result.stdout == ""
    assert "Traceback" not in result.output
    if command == "validate" and damage == "invalid":
        assert result.stderr == "error: gates: Input should be a valid dictionary\n"
        return
    assert result.stderr.startswith(f"error: {path}") and expected in result.stderr
    assert result.stderr.count("\n") == 1


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["show", "ibm_fez@"], "after @ give a date (2025-02-26)"),
        (["show", "ibm_fez@2025-13-40"], "2025-13-40 is not a calendar date; give one such as"),
        (["diff", "ibm_fez", "ibm_fez@"], "after @ give a date (2025-02-26)"),
        (["list", "--tech", "superconductin"], "did you mean superconducting?"),
        (["list", "--vendor", "ibmm"], "did you mean ibm?"),
        (["check", "ibm_manila", "--framework", "qiskt"], "did you mean qiskit?"),
        (["validate", "missing.json"], "error: no file missing.json; check the path\n"),
        (["show", "nosuch"], "error: no profile with id 'nosuch'; run `nv list`\n"),
    ],
)
def test_a_typo_or_bad_date_gets_one_error_with_the_way_out(args: list[str], error: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and result.stdout == ""
    assert error in result.stderr and "Traceback" not in result.output
    assert len(result.stderr.splitlines()) == 1


def test_pull_names_the_at_flag_for_a_bad_date() -> None:
    result = runner.invoke(app, ["pull", "ibm_fez", "--at", "2025-13-40"])
    assert result.exit_code == 1
    assert result.stderr.startswith("error: --at '2025-13-40' is not an ISO 8601 date")


# show assumptions -----------------------------------------------------------------------------


def test_show_states_the_assumption_behind_each_overriding_record(tmp_path: Path) -> None:
    gates = {
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 1e-3},
        "cz": {"avg_infidelity": 0.01, "assumption": "Default measured with isolated RB"},
    }
    calibrations = [
        {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 0.02, "assumption": "Inferred from XEB"},
        {"gate": "cz", "qubits": [1, 2], "avg_infidelity": 0.03},
    ]
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(gates=gates, calibrations=calibrations)))
    data = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)
    assert data["assumptions"] == [
        "cz: Default measured with isolated RB",
        "cz 0-1: Inferred from XEB",
    ]
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "120"}).stdout
    assert re.search(r"^assumptions\s+cz: Default measured with isolated RB\s*$", out, re.M)
    assert re.search(r"^\s+cz 0-1: Inferred from XEB\s*$", out, re.M)


def test_show_natives_as_resolved_not_as_defined(tmp_path: Path) -> None:
    gates = {
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 1e-3},
        "cz": {"avg_infidelity": 1e-2, "disabled": True},
    }
    calibrations = [{"gate": "rz", "qubits": [0], "virtual": False, "avg_infidelity": 2e-3}]
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(gates=gates, calibrations=calibrations)))
    natives = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)["natives"]
    rz, _, cz = natives
    assert rz["virtual"] is False and rz["median_avg_infidelity"] == 2e-3
    assert rz["loci"] == {"calibrated": 1, "ideal": 2, "uncalibrated": 0, "disabled": 0}
    assert cz["median_avg_infidelity"] is None and cz["disabled"] == 2
    out = runner.invoke(app, ["show", str(path)], env={"COLUMNS": "120"}).stdout
    rows = {line.split()[0]: line.split()[1:] for line in out.splitlines() if "(1q)" in line}
    rows |= {line.split()[0]: line.split()[1:] for line in out.splitlines() if "(2q)" in line}
    assert rows["rz"] == ["(1q)", "2.00e-03", "-", "1", "(2", "virtual)"]
    assert rows["cz"] == ["(2q)", "-", "-", "device-wide", "(2", "disabled)"]


def test_show_counts_a_disabled_reverse_order_of_a_symmetric_gate(tmp_path: Path) -> None:
    calibrations = [
        {"gate": "cz", "qubits": [0, 1], "avg_infidelity": 1e-2},
        {"gate": "cz", "qubits": [1, 0], "disabled": True},
    ]
    data = toy(connectivity="all_to_all", calibrations=calibrations)
    data["device"]["num_qubits"] = 2
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(data))
    natives = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)["natives"]
    cz = next(native for native in natives if native["gate"] == "cz")
    assert cz["loci"] == {"calibrated": 1, "ideal": 0, "uncalibrated": 0, "disabled": 1}


def test_show_leaves_a_disabled_locus_out_of_the_median_duration(tmp_path: Path) -> None:
    calibrations = [{"gate": "cz", "qubits": [0, 1], "disabled": True, "duration_ns": 500}]
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy(calibrations=calibrations)))
    natives = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)["natives"]
    cz = next(native for native in natives if native["gate"] == "cz")
    assert (cz["median_duration_ns"], cz["disabled"]) == (70, 1)


def test_show_groups_a_record_assumption_shared_by_many_loci(tmp_path: Path) -> None:
    calibrations = [
        {"gate": "cz", "qubits": list(pair), "avg_infidelity": 0.02, "assumption": "From XEB"}
        for pair in ([0, 1], [1, 2], [2, 3], [3, 4])
    ]
    data = toy(calibrations=calibrations)
    data["device"]["num_qubits"] = 5
    data["connectivity"] = {"edges": [[0, 1], [1, 2], [2, 3], [3, 4]]}
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(data))
    shown = json.loads(runner.invoke(app, ["show", str(path), "--json"]).stdout)
    assert shown["assumptions"] == ["cz (4 records): From XEB"]


# doctor ---------------------------------------------------------------------------------------


def _doctor_without(monkeypatch: pytest.MonkeyPatch, *absent: str) -> str:
    from importlib.metadata import PackageNotFoundError

    import noisevault.cli as cli

    def installed(package: str) -> str:
        if package in absent:
            raise PackageNotFoundError(package)
        return "1.0"

    monkeypatch.setattr(cli, "version", installed)
    return runner.invoke(app, ["doctor"], env={"COLUMNS": "200"}).stdout


def test_doctor_installs_pymatching_by_name_not_through_an_extra(monkeypatch) -> None:
    out = _doctor_without(monkeypatch, "pymatching")
    assert "pip install pymatching" in out and "noisevault[" not in out


def test_doctor_names_only_the_extras_that_install_what_is_missing(monkeypatch) -> None:
    out = _doctor_without(monkeypatch, "cirq-google", "stim", "pymatching")
    hint = install_hint("google,stim")
    assert hint in out and "pip install pymatching" in out


def test_each_extra_doctor_names_installs_its_package() -> None:
    import tomllib

    from noisevault.cli import _PACKAGES

    extras = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]

    def requirements(extra: str) -> set[str]:
        found = set()
        for item in extras[extra]:
            nested = re.fullmatch(r"noisevault\[(.+)\]", item)
            if nested:
                found |= {r for name in nested[1].split(",") for r in requirements(name)}
            else:
                found.add(re.split(r"[<>=!~ ;\[]", item)[0])
        return found

    for package, extra in _PACKAGES.items():
        if extra is not None:
            assert package in requirements(extra), (package, extra)
        assert package in requirements("dev")


# diff and list with two calibrations on one day -----------------------------------------------


def _same_day(vault: Path) -> tuple[str, str]:
    """Two H1-1 calibrations on 2025-05-02 in the vault, beside the bundled one of that day."""
    base = nv.load("quantinuum_h1-1").model_dump(mode="json", exclude_none=True)

    def save(when: str, zz: float, qubits: list, calibrations: list) -> str:
        data = {**base, "device": {**base["device"], "calibrated_at": when}}
        data["gates"] = {**base["gates"], "zz": {**base["gates"]["zz"], "avg_infidelity": zz}}
        data["qubits"], data["calibrations"] = qubits, calibrations
        profile = Profile.model_validate(data)
        path = nv.catalog.vault_path(profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        profile.save(path)
        return f"quantinuum_h1-1@{when}"

    before = save(
        "2025-05-02T09:15:00Z",
        9.8e-4,
        [{"index": 4, "t1_us": 1.0e7}],
        [{"gate": "r", "qubits": [5], "avg_infidelity": 0.0}],
    )
    after = save(
        "2025-05-02T14:30:00Z",
        1.2e-3,
        [{"index": 3, "t1_us": 5.0e6}],
        [
            {"gate": "r", "qubits": [5], "avg_infidelity": 4e-5},
            {"gate": "zz", "qubits": [0, 1], "avg_infidelity": 3e-3},
        ],
    )
    return before, after


def test_diff_says_new_or_gone_and_colors_every_change(vault: Path) -> None:
    before, after = _same_day(vault)
    env = {"COLUMNS": "80", "FORCE_COLOR": "1"}
    out = runner.invoke(app, ["diff", before, after], env=env).stdout
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    lines = plain.splitlines()
    assert (
        lines[0] == "quantinuum_h1-1 2025-05-02T09:15:00Z -> 2025-05-02T14:30:00Z  (5 hours later)"
    )
    rows = {tuple(line.split()[:3]): line.split()[-1] for line in lines if line[:1].isalnum()}
    assert rows[("3", "T1", "(us)")] == "new"
    assert rows[("4", "T1", "(us)")] == "gone"
    assert rows[("default", "2q", "avg")] == "+22.4%"
    assert re.search(
        r"^5 +1q avg infidelity +0\.00e\+00 +4\.00e-05 +\x1b\[31m-\x1b\[0m$", out, re.M
    )
    assert max(map(len, lines)) <= 80


def test_list_shows_the_time_only_where_a_date_is_shared(vault: Path) -> None:
    _same_day(vault)
    out = runner.invoke(app, ["list"], env={"COLUMNS": "80"}).stdout
    lines = out.splitlines()
    at = [i for i, line in enumerate(lines) if "quantinuum_h1-1 " in line]
    rows = [(lines[i].lstrip("* ").split()[1:3], lines[i + 1].split()) for i in at]
    assert rows == [
        (["2025-05-02", "20"], ["09:15", "UTC"]),
        (["2025-05-02", "20"], ["14:30", "UTC"]),
        (["2025-05-02", "20"], ["00:00", "UTC"]),
    ]
    h1_2 = next(i for i, line in enumerate(lines) if "quantinuum_h1-2 " in line)
    assert lines[h1_2].split()[1] == "2023-08-21" and "quantinuum_h2-1" in lines[h1_2 + 1]
    fez = next(line for line in lines if "ibm_fez" in line)
    assert fez.split()[:3] == ["ibm_fez", "2025-02-26", "156"]
    assert max(map(len, lines)) <= 80


def test_diff_lists_added_qubits_as_ranges() -> None:
    out = runner.invoke(app, ["diff", "ibm_manila", "ibm_fez"], env={"COLUMNS": "80"}).stdout
    assert "qubits added: 5 to 155" in out.splitlines()


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["diff", "ibm_fez"], "error: missing argument 'after'; see `nv diff --help`\n"),
        (["shwo", "ibm_fez"], "error: no such command 'shwo'; did you mean 'show'?\n"),
        (["show", "ibm_fez", "--qubit", "1"], "error: no such option: --qubit (Possible options"),
    ],
)
def test_a_usage_mistake_is_one_line(args: list[str], error: str) -> None:
    result = runner.invoke(app, args, env={"COLUMNS": "80"}, prog_name="nv")
    assert result.exit_code == 2 and result.stdout == ""
    assert result.stderr.startswith(error) and result.stderr.count("\n") == 1


def test_nv_alone_prints_the_help_and_no_error() -> None:
    result = runner.invoke(app, [], env={"COLUMNS": "80"}, prog_name="nv")
    assert result.exit_code == 0, result.output
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.output)  # typer forces color on CI runners
    assert "Usage: nv" in text and "list" in text
    assert "error" not in text


def test_help_ends_with_the_commands_to_start_with() -> None:
    start = [
        " Start with:",
        "   nv list             the bundled devices, offline",
        "   nv show ibm_fez     one device's calibration",
        "   nv check ibm_fez    test each installed framework export",
    ]
    for args in ([], ["--help"]):
        result = runner.invoke(app, args, env={"COLUMNS": "60"}, prog_name="nv")
        assert result.exit_code == 0, result.output
        lines = re.sub(r"\x1b\[[0-9;]*m", "", result.output).rstrip().splitlines()
        assert lines[-4:] == start and lines[-5] == ""
    sub = runner.invoke(app, ["show", "--help"], env={"COLUMNS": "80"}, prog_name="nv")
    assert "Start with" not in sub.output


def test_an_unexpected_failure_is_one_line_unless_debugging(monkeypatch) -> None:
    def broken(ref: str) -> None:
        raise KeyError("provider")

    monkeypatch.setattr(nv.catalog, "load", broken)
    result = runner.invoke(app, ["show", "ibm_fez"])
    assert result.exit_code == 1 and result.stderr.count("\n") == 1
    assert result.stderr.startswith("error: unexpected KeyError: 'provider'; please report")
    debug = runner.invoke(app, ["show", "ibm_fez"], env={"NOISEVAULT_DEBUG": "1"})
    assert isinstance(debug.exception, KeyError)
