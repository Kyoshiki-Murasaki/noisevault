from __future__ import annotations

import contextlib
import copy
import functools
import importlib.util
import io
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import require
from typer.testing import CliRunner

import noisevault as nv
from noisevault.cli import app
from noisevault.counts import _SAMPLER_V2_OPTIONS, PlannedCircuit, load_counts, plan
from noisevault.errors import NoiseVaultError, install_hint
from noisevault.profile import Profile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_on_ibm.py"
SHOTS = 4000
FEZ = "nv:06404cefa54f"
FEZ_PLAN = """\
ibm_fez@2025-02-26 nv:06404cefa54f on qubits 136-143-142-141

circuit       qubits           gates  duration
ghz_chain     136-143-142-141     11    300 ns
mirror        136-143-142         37    600 ns
single_qubit  136-143             12    144 ns
readout       136-143-142-141      0      0 ns

shots           4000 per circuit, 4 circuits in one job
usage           about 6.1 s of QPU time (IBM's estimate)
counts file     fez.counts.json
"""
WAITING = (
    "waiting for it to run\nCtrl-C stops waiting; run the same command again to collect the job\n"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_on_ibm", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _fez() -> Any:
    fake_provider = require("qiskit_ibm_runtime.fake_provider")
    aer = require("qiskit_aer")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fake = fake_provider.FakeFez()
    # Without a sim, each local run deep-copies the fake and rebuilds its 156-qubit noise model.
    fake.sim = aer.AerSimulator()
    return fake


def _fractional_fez() -> Any:
    converter = require("qiskit_ibm_runtime.utils.backend_converter")
    fake = _fez()
    configuration = copy.deepcopy(fake.configuration())
    configuration.basis_gates = [*configuration.basis_gates, "rx", "rzz"]
    fake._target = converter.convert_to_target(
        configuration, fake.properties(), include_fractional_gates=True
    )
    return fake


@functools.cache
def _fez_profile() -> Profile:
    return nv.from_qiskit_backend(_fez())


def _pinned_snapshot() -> Profile:
    profile = _fez_profile()
    if profile.short_fingerprint != FEZ:
        pytest.skip(f"FakeFez here is {profile.short_fingerprint}; the pinned numbers are {FEZ}'s")
    return profile


@contextlib.contextmanager
def _five_hours_behind_utc() -> Iterator[None]:
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("TZ", "NVT+5")
            time.tzset()
            yield
    finally:
        time.tzset()


@dataclass
class Submission:
    sampler: Any
    pubs: list[Any]
    job: Any


def _recording_sampler(
    monkeypatch: pytest.MonkeyPatch, *, lose_the_first_wait: bool = False
) -> list[Submission]:
    runtime = require("qiskit_ibm_runtime")
    submissions: list[Submission] = []

    class Recording(runtime.SamplerV2):
        def run(self, pubs: Any, **options: Any) -> Any:
            job = super().run(pubs, **options)
            if lose_the_first_wait and not submissions:

                def lost(*args: Any, **kwargs: Any) -> Any:
                    raise ConnectionError("network is unreachable")

                job.result = lost
            submissions.append(Submission(self, list(pubs), job))
            return job

    monkeypatch.setattr(runtime, "SamplerV2", Recording)
    return submissions


def _no_job(job_id: str) -> Any:
    raise AssertionError(f"the script opened job {job_id} instead of submitting one")


@dataclass
class LocalRun:
    output: Path
    printed: str
    warned: list[str]
    submissions: list[Submission]
    calibrated_at: list[datetime | None]
    fractional: list[bool]
    before: datetime
    after: datetime
    fake: Any


@pytest.fixture(scope="module")
def local_run(tmp_path_factory: pytest.TempPathFactory) -> LocalRun:
    folder = tmp_path_factory.mktemp("run_on_ibm")
    with pytest.MonkeyPatch.context() as monkeypatch, _five_hours_behind_utc():
        monkeypatch.setenv("NOISEVAULT_HOME", str(folder / "nv_home"))
        submissions = _recording_sampler(monkeypatch)
        fake = _fez()
        calibrated_at: list[datetime | None] = []
        fractional: list[bool] = []

        def calibration(at: datetime | None) -> Profile:
            calibrated_at.append(at)
            return nv.from_qiskit_backend(fake)

        def open_backend(flag: bool) -> Any:
            fractional.append(flag)
            return fake

        output = folder / "fez.counts.json"
        printed = io.StringIO()
        before = datetime.now(UTC)
        with contextlib.redirect_stdout(printed), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            script.run(
                shots=SHOTS,
                output=output,
                calibration=calibration,
                open_backend=open_backend,
                open_job=_no_job,
                confirm=None,
            )
        after = datetime.now(UTC)
    return LocalRun(
        output,
        printed.getvalue(),
        [str(w.message) for w in caught],
        submissions,
        calibrated_at,
        fractional,
        before,
        after,
        fake,
    )


def test_a_missing_ibm_extra_gives_the_install_command(tmp_path: Path) -> None:
    hide = "import sys; sys.modules['qiskit_ibm_runtime'] = None; import runpy;"
    run = f"sys.argv = sys.argv[1:]; runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    out = tmp_path / "x.counts.json"
    result = subprocess.run(
        [sys.executable, "-c", hide + run, "run_on_ibm.py", "ibm_fez", "-o", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr == (
        f"error: scripts/run_on_ibm.py needs qiskit-ibm-runtime\nhint: {install_hint('ibm')}\n"
    )


def test_the_counts_file_binds_to_the_calibration_and_lists_the_planned_circuits(
    local_run: LocalRun,
) -> None:
    counts = load_counts(local_run.output)
    profile = _fez_profile()
    assert (counts.source, counts.backend) == ("simulated", "ibm_fez")
    assert (counts.profile.id, counts.profile.fingerprint) == ("ibm_fez", profile.fingerprint)
    (submission,) = local_run.submissions
    assert counts.execution.job_ids == (submission.job.job_id(),)
    assert [(c.name, c.qubits, c.ops) for c in counts.circuits] == [
        (c.name, c.qubits, c.ops) for c in plan(profile)
    ]
    assert all(c.shots == SHOTS for c in counts.circuits)
    assert local_run.calibrated_at == [None, counts.run_at]
    assert local_run.fractional == [False]
    assert not (local_run.output.parent / "fez.job.json").exists()


def test_run_at_is_when_the_job_started_in_utc(local_run: LocalRun) -> None:
    run_at = load_counts(local_run.output).run_at
    assert local_run.before <= run_at <= local_run.after


def _instructions(qc: Any) -> list[tuple[Any, ...]]:
    out = []
    for instruction in qc.data:
        operation = instruction.operation
        params = tuple(float(p) for p in operation.params)
        if operation.name == "delay":
            params = (operation.params[0], operation.unit)
        out.append(
            (
                operation.name,
                tuple(qc.find_bit(q).index for q in instruction.qubits),
                params,
                tuple(qc.find_bit(c).index for c in instruction.clbits),
            )
        )
    return out


def test_the_submitted_circuits_are_the_planned_ops_then_the_measurements(
    local_run: LocalRun,
) -> None:
    (submission,) = local_run.submissions
    target = local_run.fake.target
    dt_ns = target.dt * 1e9
    planned = plan(_fez_profile())
    assert len(submission.pubs) == len(planned)
    for circuit, qc in zip(planned, submission.pubs, strict=True):
        expected = []
        for op in circuit.ops:
            qubits = tuple(circuit.qubits[q] for q in op.qubits)
            if op.name == "delay":
                expected.append(("delay", qubits, (round(op.params[0] / dt_ns), "dt"), ()))
            else:
                expected.append((op.name, qubits, op.params, ()))
        expected.append(("barrier", circuit.qubits, (), ()))
        expected += [("measure", (q,), (), (i,)) for i, q in enumerate(circuit.qubits)]
        assert qc.num_qubits == target.num_qubits
        assert (circuit.name, _instructions(qc)) == (circuit.name, expected)


def test_one_job_carries_every_option_the_counts_format_checks(local_run: LocalRun) -> None:
    (submission,) = local_run.submissions
    recorded = load_counts(local_run.output).execution.options
    paths = {
        **_SAMPLER_V2_OPTIONS,
        "default_shots": SHOTS,
        "execution.rep_delay": local_run.fake.default_rep_delay,
    }
    for path, value in paths.items():
        submitted, saved = submission.sampler.options, recorded
        for part in path.split("."):
            submitted, saved = getattr(submitted, part), saved[part]
        assert (path, submitted, saved) == (path, value, value)
    timing = {"execution": {"scheduler_timing": True}}
    assert submission.sampler.options.experimental == recorded["experimental"] == timing
    assert any("have no effect in local testing mode" in w for w in local_run.warned)


def test_the_run_ends_with_its_files_and_a_command_that_scores_them(local_run: LocalRun) -> None:
    counts = load_counts(local_run.output)
    profile = _fez_profile()
    (job_id,) = counts.execution.job_ids
    ref = f"ibm_fez@{profile.device.calibrated_at:%Y-%m-%d}"
    assert local_run.printed.split("\n\n")[-2:] == [
        f"submitted job {job_id} to ibm_fez\n{WAITING.rstrip()}",
        f"run             {counts.run_at:%Y-%m-%d %H:%MZ}, job {job_id}\n"
        f"counts file     {local_run.output}\n"
        f"calibration     {ref} {profile.short_fingerprint}\n"
        f"next            nv compare {ref} {local_run.output}\n",
    ]
    result = CliRunner().invoke(app, ["compare", ref, str(local_run.output)], env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    header = result.output.splitlines()[:2]
    assert header[0].split()[1] == profile.short_fingerprint
    assert header[1].startswith(f"counts {local_run.output}, simulated, sha256:")


def test_the_summary_shows_the_circuits_and_the_usage_and_a_no_submits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch)
    profile = _pinned_snapshot()
    fake = _fez()
    monkeypatch.chdir(tmp_path)
    prompts: list[str] = []

    def decline(prompt: str) -> bool:
        prompts.append(prompt)
        return False

    measured = script.run(
        shots=SHOTS,
        output=Path("fez.counts.json"),
        calibration=lambda at: profile,
        open_backend=lambda fractional: fake,
        open_job=_no_job,
        confirm=decline,
    )
    assert measured is None
    assert capsys.readouterr().out == FEZ_PLAN + "\nnothing submitted\n"
    assert prompts == ["Submit the job to ibm_fez? [y/N] "]
    assert submissions == []
    assert list(tmp_path.iterdir()) == []


def test_the_counts_keep_circuit_qubit_0_first() -> None:
    fake = _fez()
    profile = _fez_profile()
    flip = PlannedCircuit(name="flip", qubits=(136, 143), ops=(("x", [0], []),))
    qc = script.isa_circuit(flip, profile, fake.target).circuit
    batch = script.Batch(
        profile=profile,
        planned=(flip,),
        circuits=(qc,),
        durations_ns=(0.0,),
        shots=100,
        options=script.sampler_options(100, fake.default_rep_delay),
        usage_s=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        job = script.submit(fake, batch)
        submitted = script.Submitted(job.job_id(), profile, (flip,), batch.options)
        ran = script.collect(job, submitted)
    counts = script.counts_file(profile, submitted, ran)
    assert dict(counts.circuits[0].counts) == {"10": 100}


@pytest.mark.parametrize(
    ("circuit", "message"),
    [
        (
            PlannedCircuit(name="far", qubits=(136, 0), ops=(("cz", [0, 1], []),)),
            "circuit far: the backend does not support cz on qubits 136-0",
        ),
        (
            PlannedCircuit(name="odd", qubits=(136,), ops=(("delay", [0], [10.0]),)),
            "circuit odd: the 10 ns delay on qubit 136 is 2.5 dt, and the backend times delays in"
            " whole dt of 4 ns",
        ),
        (
            PlannedCircuit(name="blip", qubits=(136,), ops=(("delay", [0], [4.0]),)),
            "circuit blip: the 4 ns delay on qubit 136 is 1 dt, shorter than the backend's minimum"
            " of 2 dt",
        ),
    ],
)
def test_ops_the_backend_would_not_run_as_planned_are_refused(
    circuit: PlannedCircuit, message: str
) -> None:
    fake = _fez()
    with pytest.raises(NoiseVaultError) as info:
        script.isa_circuit(circuit, _fez_profile(), fake.target)
    assert info.value.message == message


def test_a_gate_whose_backend_duration_changed_is_refused() -> None:
    target_module = require("qiskit.transpiler")
    profile = _pinned_snapshot()
    fake = _fez()
    error = fake.target["sx"][(136,)].error
    fake.target.update_instruction_properties(
        "sx", (136,), target_module.InstructionProperties(duration=28e-9, error=error)
    )
    with pytest.raises(NoiseVaultError) as info:
        script.isa_circuit(plan(profile)[0], profile, fake.target)
    assert info.value.message == (
        "circuit ghz_chain: sx on qubit 136 lasts 6 dt (24 ns) in the calibration and 7 dt"
        " (28 ns) on the backend, so the planned delays would not fill the gaps"
    )
    assert info.value.hint == "run the script again to plan from the current calibration"


@pytest.mark.parametrize(
    ("grid", "circuit", "message"),
    [
        (
            {"pulse_alignment": 8},
            "ghz_chain",
            "circuit ghz_chain: cz on qubits 136-143 would start at 6 dt, off the backend's grid"
            " of 8 dt",
        ),
        (
            {"acquire_alignment": 16},
            "single_qubit",
            "circuit single_qubit: the measurements would start at 36 dt, off the backend's grid"
            " of 16 dt",
        ),
    ],
)
def test_an_op_or_measurement_off_the_backend_grid_is_refused(
    grid: dict[str, int], circuit: str, message: str
) -> None:
    profile = _pinned_snapshot()
    fake = _fez()
    for name, value in grid.items():
        setattr(fake.target, name, value)
    (planned,) = (c for c in plan(profile) if c.name == circuit)
    with pytest.raises(NoiseVaultError) as info:
        script.isa_circuit(planned, profile, fake.target)
    assert info.value.message == message


def _edited(profile: Profile, edit: Callable[[dict[str, Any]], None]) -> Profile:
    data = profile.to_dict()
    edit(data)
    return Profile.model_validate(data)


def _lower_sx_error(data: dict[str, Any]) -> None:
    record = next(r for r in data["calibrations"] if (r["gate"], r["qubits"]) == ("sx", [136]))
    record["avg_infidelity"] = 1e-4


def _submitted_and_ran(profile: Profile) -> tuple[Any, Any]:
    planned = plan(profile)
    submitted = script.Submitted("job-1", profile, planned, script.sampler_options(SHOTS, 2.5e-4))
    ran = script.Ran(
        job_id="job-1",
        source="simulated",
        run_at=datetime(2026, 4, 16, 9, 30, tzinfo=UTC),
        counts=tuple({"0" * len(c.qubits): SHOTS} for c in planned),
        timing={},
    )
    return submitted, ran


def test_the_counts_bind_to_the_calibration_in_effect_when_the_job_ran() -> None:
    planned = _fez_profile()
    latest = _edited(planned, _lower_sx_error)
    submitted, ran = _submitted_and_ran(planned)
    assert script.bind(submitted, ran, lambda at: latest) == script.Binding(latest)


def test_a_changed_duration_binds_the_counts_to_the_plan_and_warns() -> None:
    planned = _pinned_snapshot()

    def slower_sx(data: dict[str, Any]) -> None:
        data["gates"]["sx"]["duration_ns"] = 28.0

    latest = _edited(planned, slower_sx)
    submitted, ran = _submitted_and_ran(planned)
    assert script.bind(submitted, ran, lambda at: latest) == script.Binding(
        planned,
        f"IBM recalibrated ibm_fez before the job ran ({latest.short_fingerprint}), and sx on"
        " qubit 136 now takes 28 ns, not 24 ns, so the submitted delays may not match the"
        f" timeline that ran; the counts bind to the planned calibration {FEZ}",
        "run the script again for counts that match one calibration",
    )


def test_a_gate_disabled_when_the_job_ran_binds_the_counts_to_the_plan() -> None:
    planned = _fez_profile()

    def disable_cz(data: dict[str, Any]) -> None:
        data["calibrations"] = [
            {"gate": "cz", "qubits": [136, 143], "disabled": True}
            if (r["gate"], r["qubits"]) == ("cz", [136, 143])
            else r
            for r in data["calibrations"]
        ]

    submitted, ran = _submitted_and_ran(planned)
    binding = script.bind(submitted, ran, lambda at: _edited(planned, disable_cz))
    assert binding.profile is planned
    assert binding.warning is not None
    assert "nv compare would refuse the counts against it (circuit ghz_chain: " in binding.warning


def test_a_calibration_older_than_the_plan_binds_the_counts_to_the_plan() -> None:
    planned = _fez_profile()

    def a_day_earlier(data: dict[str, Any]) -> None:
        _lower_sx_error(data)
        data["device"]["calibrated_at"] = "2025-02-25T20:16:25Z"

    latest = _edited(planned, a_day_earlier)
    submitted, ran = _submitted_and_ran(planned)
    assert script.bind(submitted, ran, lambda at: latest) == script.Binding(
        planned,
        f"the calibration IBM returned for the time the job ran, {latest.short_fingerprint}, is"
        f" older than the planned {planned.short_fingerprint}; the counts bind to the planned"
        " calibration",
    )


def test_a_failed_pull_at_run_time_binds_the_counts_to_the_plan() -> None:
    planned = _fez_profile()
    submitted, ran = _submitted_and_ran(planned)

    def calibration(at: datetime | None) -> Profile:
        raise nv.SourceUnavailable("IBM did not return the calibration of ibm_fez (timeout)")

    assert script.bind(submitted, ran, calibration) == script.Binding(
        planned,
        "could not pull the calibration in effect when the job ran (IBM did not return the"
        " calibration of ibm_fez (timeout)); the counts bind to the planned calibration"
        f" {planned.short_fingerprint}",
    )


def _collecting_job_1(folder: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[], Any]:
    profile = _fez_profile()
    submitted, ran = _submitted_and_ran(profile)
    submitted.save(folder / "fez.job.json")
    monkeypatch.setattr(script, "collect", lambda *args: ran)

    def plan_again(fractional: bool) -> Any:
        raise AssertionError("the script planned a new job instead of collecting job-1")

    return lambda: script.run(
        shots=SHOTS,
        output=folder / "fez.counts.json",
        calibration=lambda at: profile,
        open_backend=plan_again,
        open_job=lambda job_id: job_id,
        confirm=None,
    )


def test_a_disk_that_refuses_the_counts_leaves_the_job_to_collect_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch)
    (tmp_path / "fez.counts.json").mkdir()
    pending = tmp_path / "fez.job.json"
    with pytest.raises(NoiseVaultError) as info:
        run()
    assert info.value.message.startswith("could not collect the counts of job job-1 (")
    assert info.value.hint == (
        f"run the same command again to collect them, or delete {pending} to submit a new job"
    )
    assert pending.exists()


def test_a_defect_after_the_job_ran_raises_instead_of_asking_to_collect_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _collecting_job_1(tmp_path, monkeypatch)

    def defect(*args: Any) -> Any:
        raise TypeError("a defect in the script")

    monkeypatch.setattr(script, "counts_file", defect)
    with pytest.raises(TypeError, match="a defect in the script"):
        run()


@dataclass
class Account:
    pulled_at: list[datetime | None]
    opened: list[bool]


def _account(
    monkeypatch: pytest.MonkeyPatch, fake: Any, submissions: list[Submission] | None = None
) -> Account:
    runtime = require("qiskit_ibm_runtime")
    snapshot = fake.properties()
    seen = Account([], [])

    class Calibrated:
        name = "ibm_fez"
        processor_type = fake.processor_type

        def properties(self, refresh: bool = False, datetime: datetime | None = None) -> Any:
            seen.pulled_at.append(datetime)
            return snapshot

    class Service:
        def __init__(self, **options: Any) -> None:
            pass

        def backend(self, name: str, **options: Any) -> Any:
            if "use_fractional_gates" not in options:
                return Calibrated()
            seen.opened.append(options["use_fractional_gates"])
            return fake

        def job(self, job_id: str) -> Any:
            return next(s.job for s in submissions or () if s.job.job_id() == job_id)

    monkeypatch.setattr(runtime, "QiskitRuntimeService", Service)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    return seen


def test_the_script_runs_end_to_end_through_the_account_with_fractional_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _account(monkeypatch, _fractional_fez())
    output = tmp_path / "fez.counts.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(["ibm_fez", "--yes", "-o", str(output)]) == 0
    counts = load_counts(output)
    assert seen.pulled_at == [None, counts.run_at]
    assert seen.opened == [True]
    assert "rx" in {op.name for circuit in counts.circuits for op in circuit.ops}
    assert counts.profile.fingerprint != _fez_profile().fingerprint
    label, *command, ref, path = capsys.readouterr().out.rstrip("\n").splitlines()[-1].split()
    assert (label, command, path) == ("next", ["nv", "compare"], str(output))
    assert nv.load(ref).fingerprint == counts.profile.fingerprint


def test_running_the_same_command_again_collects_a_job_whose_wait_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch, lose_the_first_wait=True)
    _account(monkeypatch, _fractional_fez(), submissions)
    output, pending = tmp_path / "fez.counts.json", tmp_path / "fez.job.json"
    command = ["ibm_fez", "--yes", "-o", str(output)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert script.main(command) == 1
        (submission,) = submissions
        job_id = submission.job.job_id()
        assert capsys.readouterr().err == (
            f"error: could not collect the counts of job {job_id} (network is unreachable)\n"
            "hint: run the same command again to collect them, or delete"
            f" {pending} to submit a new job\n"
        )
        assert (pending.exists(), output.exists()) == (True, False)
        del submission.job.result
        assert script.main(command) == 0
    assert len(submissions) == 1
    assert load_counts(output).execution.job_ids == (job_id,)
    assert not pending.exists()
    printed = capsys.readouterr().out
    assert printed.startswith(f"collecting job {job_id}, submitted earlier for {output}\n{WAITING}")


def test_a_backend_without_the_planned_fractional_gate_refuses_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _account(monkeypatch, _fez())
    assert script.main(["ibm_fez", "--yes", "-o", str(tmp_path / "fez.counts.json")]) == 1
    assert seen.opened == [True]
    assert capsys.readouterr().err == (
        "error: circuit ghz_chain: the backend does not support rx on qubit 136\n"
    )


def test_without_a_terminal_the_script_asks_for_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submissions = _recording_sampler(monkeypatch)
    _account(monkeypatch, _fractional_fez())
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    folder = tmp_path / "counts"
    folder.mkdir()
    assert script.main(["ibm_fez", "-o", str(folder / "fez.counts.json")]) == 1
    assert capsys.readouterr().err == (
        "error: cannot ask before submitting, because standard input is not a terminal\n"
        "hint: pass --yes to submit without asking\n"
    )
    assert submissions == []
    assert list(folder.iterdir()) == []


def test_started_at_reads_ibm_timestamps_as_utc() -> None:
    with _five_hours_behind_utc():
        for stamp in ("2026-04-16T09:30:02.5Z", "2026-04-16T09:30:02.5"):
            assert script.started_at({"timestamps": {"running": stamp}}) == datetime(
                2026, 4, 16, 9, 30, 2, 500000, tzinfo=UTC
            )


def test_a_missing_account_gives_the_setup_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = require("qiskit_ibm_runtime")
    missing = runtime.accounts.AccountNotFoundError("Unable to find account.")

    def no_account(**options: Any) -> None:
        raise missing

    monkeypatch.setattr(runtime, "QiskitRuntimeService", no_account)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    assert script.main(["ibm_fez", "-o", str(tmp_path / "fez.counts.json")]) == 1
    assert capsys.readouterr() == (
        "",
        f"error: could not open your IBM Quantum account ({missing})\n"
        f"hint: {script.ACCOUNT_SETUP}\n",
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("name", "message", "hint"),
    [
        (
            "fez.counts.json",
            "{output} exists",
            "give -o a new file name; the script never replaces counts",
        ),
        ("nowhere/fez.counts.json", "no folder {folder}", "create it, or give -o another path"),
    ],
)
def test_an_output_the_run_could_not_write_is_refused_before_the_account_opens(
    name: str,
    message: str,
    hint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = require("qiskit_ibm_runtime")

    def account(**options: Any) -> None:
        raise AssertionError("the script opened the IBM account")

    monkeypatch.setattr(runtime, "QiskitRuntimeService", account)
    (tmp_path / "fez.counts.json").write_text("paid for\n")
    output = tmp_path / name
    assert script.main(["ibm_fez", "-o", str(output)]) == 1
    error = message.format(output=output, folder=output.parent)
    assert capsys.readouterr() == ("", f"error: {error}\nhint: {hint}\n")
    assert (tmp_path / "fez.counts.json").read_text() == "paid for\n"
