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
from conftest import MANILA_V01, toy
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.errors import SourceUnavailable
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
        (["show", "ibm_fez@2020-01-01"], "available: ibm_fez@2025-02-26T20:16:25Z", "nv list"),
        (["show", "missing.json"], "error: no file missing.json", "check the path"),
        (["cite", "ibm fez"], "is not a profile id", None),
        (
            ["check", "ibm_manila", "--framework", "qiskt"],
            "'qiskt': give one or more of qiskit",
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
    assert "Traceback" not in result.output and result.stdout == ""


# list and show --------------------------------------------------------------------------------


def test_list_groups_profiles_by_technology() -> None:
    result = runner.invoke(app, ["list"])
    lines = [line.rstrip() for line in result.stdout.splitlines()]
    assert lines[0].split() == ["id", "date", "qubits", "processor", "source", "license"]
    row = next(line for line in lines if line.strip().startswith("ibm_fez "))
    assert row.split() == ["ibm_fez", "2025-02-26", "156", "Heron", "r2", "package", "Apache-2.0"]
    assert lines.index("superconducting") < lines.index(row) < lines.index("trapped_ion")
    assert lines[-1].startswith(f"{len(nv.profiles())} profiles.")


def test_list_filters_and_prints_json() -> None:
    result = runner.invoke(app, ["list", "--tech", "trapped_ion", "--json"])
    rows = json.loads(result.stdout)
    assert rows and {r["technology"] for r in rows} == {"trapped_ion"}
    assert {"id", "date", "num_qubits", "processor", "source_kind", "license"} <= set(rows[0])
    vendor = json.loads(runner.invoke(app, ["list", "--vendor", "google", "--json"]).stdout)
    assert {r["id"] for r in vendor} == {"google_rainbow", "google_weber"}


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
    rows = text[text.index("1q error") :].splitlines()[1:]
    assert len(rows) == 5 and all(row.endswith("(sx)") for row in map(str.rstrip, rows))


def test_show_all_to_all_and_notes() -> None:
    out = runner.invoke(app, ["show", "quantinuum_h1-1"], env={"COLUMNS": "200"}).stdout
    assert "all-to-all" in out and "device-wide" in out
    assert "T1 and T2 unknown" in out and re.search(r"not modeled\s+leakage on r\s", out)


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


def test_diff_of_identical_profiles() -> None:
    result = runner.invoke(app, ["diff", "ibm_fez", "ibm_fez@2025-02-26"])
    assert result.exit_code == 0 and "No change" in result.stdout


def test_check_prints_a_table_per_framework() -> None:
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq,pennylane"])
    assert result.exit_code == 0, result.output
    rows = {line.split()[0]: line.split() for line in result.stdout.splitlines() if line}
    assert rows["cirq"][1] == "pass" and rows["pennylane"][1] == "pass"
    assert "qiskit" not in rows
    assert "not a measure of how well the model matches the hardware" in " ".join(
        result.stdout.split()
    )


def test_check_failure_exits_1(monkeypatch) -> None:
    from noisevault.frameworks import cirq as nv_cirq

    monkeypatch.setattr(nv_cirq.NoiseVaultNoiseModel, "_noise", lambda self, *a: [])
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq"])
    assert result.exit_code == 1 and "FAIL" in result.stdout


def test_check_json_and_skips() -> None:
    result = runner.invoke(
        app, ["check", "quantinuum_h1-1", "--framework", "stim,cirq", "--json"]
    )
    data = json.loads(result.stdout)
    assert [f["framework"] for f in data["frameworks"]] == ["cirq"]
    assert data["skipped"][0]["framework"] == "stim"
    assert result.exit_code == 0


def test_cite() -> None:
    profile = nv.load("ibm_fez")
    text = runner.invoke(app, ["cite", "ibm_fez"]).stdout
    assert text == profile.citation() + "\n"
    bibtex = runner.invoke(app, ["cite", "ibm_fez", "--bibtex"]).stdout
    assert bibtex == (
        "@misc{nv_ibm_fez_2025_02_26,\n"
        "  title = {{Calibrated noise of ibm\\_fez at 2025-02-26T20:16:25Z}},\n"
        "  author = {{IBM Quantum}},\n"
        "  year = {2025},\n"
        f"  howpublished = {{NoiseVault profile ibm\\_fez, sha256:{profile.fingerprint}}},\n"
        "  note = {Retrieved via qiskit-ibm-runtime."
        " Source: qiskit-ibm-runtime 0.49.0 FakeFez; license Apache-2.0}\n"
        "}\n"
    )


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
        f"  howpublished = {{NoiseVault profile test\\_toy, sha256:{fingerprint}}},\n"
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
    out = runner.invoke(app, ["doctor"]).stdout
    for package in ("qiskit", "cirq-core", "pennylane", "stim"):
        line = next(line for line in out.splitlines() if line.split()[:1] == [package])
        assert line.split()[1] == version(package)
    assert f"vault: {vault} (0 profiles)" in out
    assert f"bundled profiles: {len(nv.catalog.bundled_profiles())}" in out


_INSTALL_ALL = (
    'pip install "noisevault[all] @ git+https://github.com/Kyoshiki-Murasaki/noisevault@main"'
)


def test_doctor_gives_a_whole_install_command_for_missing_frameworks(monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    import noisevault.cli as cli

    def without_stim(package: str) -> str:
        if package == "stim":
            raise PackageNotFoundError(package)
        return version(package)

    monkeypatch.setattr(cli, "version", without_stim)
    out = runner.invoke(app, ["doctor"], env={"COLUMNS": "80"}).stdout
    assert f"To add the missing frameworks: {_INSTALL_ALL}" in out.splitlines()


def test_check_with_no_framework_installed_gives_the_install_command(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cirq", None)
    result = runner.invoke(app, ["check", "ibm_manila", "--framework", "cirq"])
    assert result.exit_code == 1
    assert f"no framework could run the check; install one: {_INSTALL_ALL}" in result.stderr
