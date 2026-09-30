from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from conftest import MANILA_V01, toy
from typer.testing import CliRunner

from noisevault.cli import app

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


def test_doctor_runs() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0 and "noisevault" in result.stdout and "0.2.0" in result.stdout
