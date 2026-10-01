from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import require

import noisevault as nv
from noisevault.sources import ibm_account


class _Backend:
    """The parts of an IBMBackend the account source uses, backed by FakeManilaV2's data."""

    name = "ibm_manila"
    processor_type = {"family": "Falcon", "revision": "5.11"}

    def __init__(self, calls: list[Any]) -> None:
        fake_provider = require("qiskit_ibm_runtime.fake_provider")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fake = fake_provider.FakeManilaV2()
        self._calls = calls

    def properties(self, refresh: bool = False, datetime: datetime | None = None) -> Any:
        self._calls.append(datetime)
        return self._fake.properties()


class _Service:
    def __init__(self, calls: list[Any], **options: Any) -> None:
        calls.append(options)
        self._calls = calls

    def backend(self, name: str) -> _Backend:
        if name != "ibm_manila":
            raise RuntimeError(f"No backend matches the criteria: {name}")
        return _Backend(self._calls)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    runtime = require("qiskit_ibm_runtime")
    seen: list[Any] = []
    monkeypatch.setattr(runtime, "QiskitRuntimeService", lambda **o: _Service(seen, **o))
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    return seen


def test_pull_reads_the_account_snapshot(calls: list[Any]) -> None:
    profile = ibm_account.pull("ibm_manila", at="2025-06-01")
    assert calls[0] == {"channel": "ibm_quantum_platform"}
    assert calls[1] == datetime(2025, 6, 1, tzinfo=UTC)
    assert profile.fingerprint == nv.load("ibm_manila").fingerprint
    prov = profile.provenance
    assert (prov.source_kind, prov.redistributable) == ("account_api", "unknown")
    assert prov.source_hash.startswith("sha256:") and prov.retrieved_at is not None


def test_token_from_the_environment(calls: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-key")
    monkeypatch.setenv("IBM_QUANTUM_INSTANCE", "test-instance")
    ibm_account.pull("ibm_manila")
    assert calls[0] == {
        "channel": "ibm_quantum_platform",
        "token": "test-key",
        "instance": "test-instance",
    }
    assert calls[1] is None


def test_missing_account_explains_the_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = require("qiskit_ibm_runtime")

    def no_account(**options: Any) -> None:
        raise runtime.accounts.AccountNotFoundError("Unable to find account.")

    monkeypatch.setattr(runtime, "QiskitRuntimeService", no_account)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    with pytest.raises(nv.SourceUnavailable) as info:
        nv.pull("ibm_manila", source="ibm-account")
    message = str(info.value)
    assert "save_account" in message and "IBM_QUANTUM_TOKEN" in message
    assert "source='ibm'" in message


def test_device_the_account_cannot_see(calls: list[Any]) -> None:
    with pytest.raises(nv.SourceUnavailable, match="cannot open ibm_nowhere"):
        ibm_account.pull("ibm_nowhere")


def test_missing_runtime_gives_an_install_command_that_works(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)
    command = (
        'pip install "noisevault[ibm] @ git+https://github.com/Kyoshiki-Murasaki/noisevault@main"'
    )
    with pytest.raises(nv.SourceUnavailable) as info:
        nv.pull("ibm_manila", source="ibm-account")
    assert command in str(info.value)


def test_a_calibration_request_ibm_rejects_is_a_one_line_cli_error(
    calls: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from noisevault.cli import app

    exceptions = require("qiskit_ibm_runtime.exceptions")

    def rejected(self: _Backend, refresh: bool = False, datetime: datetime | None = None) -> Any:
        raise exceptions.IBMBackendApiProtocolError("Unexpected return value from the server.")

    monkeypatch.setattr(_Backend, "properties", rejected)
    with pytest.raises(nv.SourceUnavailable, match="Unexpected return value"):
        ibm_account.pull("ibm_manila")
    result = CliRunner().invoke(app, ["pull", "ibm_manila", "--source", "ibm-account"])
    assert result.exit_code == 1 and result.exception.__class__ is SystemExit
    [line] = result.stderr.splitlines()
    assert line.startswith("error: IBM did not return the calibration of ibm_manila")
    assert "try again later" in line and "--source ibm" in line
