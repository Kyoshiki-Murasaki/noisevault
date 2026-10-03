# /// script
# requires-python = ">=3.11"
# dependencies = ["noisevault[ibm] @ git+https://github.com/Kyoshiki-Murasaki/noisevault"]
# ///
r"""Run the nv compare circuits on an IBM device and save the counts for nv compare.

    uv run \
      https://raw.githubusercontent.com/Kyoshiki-Murasaki/noisevault/main/scripts/run_on_ibm.py \
      ibm_kingston --shots 4000 -o kingston-0416.counts.json

uv installs the dependencies listed at the top of this file. In a clone with the ``ibm`` extra,
run ``python scripts/run_on_ibm.py`` with the same arguments.

The script pulls the device's calibration through your IBM Quantum account and plans the
circuits with ``noisevault.counts.plan``. It checks every op and delay against the backend, shows
IBM's usage estimate, and asks before it submits one SamplerV2 job. When the job is done, the
script writes a counts file and binds the counts to the calibration in effect at that time. Then
the script prints the ``nv compare`` command to run next. If the wait for the job stops, run
the same command again. The script then collects the submitted job and does not submit another.

The script needs an IBM Quantum account. Save the account with
``QiskitRuntimeService.save_account``, or put an API key in ``IBM_QUANTUM_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import noisevault as nv  # noqa: E402
from noisevault import catalog, gates  # noqa: E402
from noisevault.compare import _bind  # noqa: E402
from noisevault.counts import (  # noqa: E402
    _RUN_RULES,
    COUNTS_FORMAT,
    SAMPLER_V2_OPTIONS,
    MeasuredCounts,
    PlannedCircuit,
    _duration,
    plan,
)
from noisevault.errors import CountsError, NoiseVaultError, install_hint, qubit_loci  # noqa: E402
from noisevault.profile import Profile, exact_ref, ref_on_day, write_atomically  # noqa: E402
from noisevault.sources import ibm_account  # noqa: E402

DEFAULT_SHOTS = 4000
FRACTIONAL_GATES = frozenset({"rx", "rzz"})
CLBITS = "meas"
ACCOUNT_SETUP = (
    "set IBM_QUANTUM_TOKEN to your IBM Quantum API key, or save the key once with"
    " QiskitRuntimeService.save_account(token=...)"
)
LABEL = 16
SUB_JOB_OVERHEAD_S = 2.0

Calibration = Callable[[datetime | None], Profile]


@dataclass(frozen=True)
class Batch:
    profile: Profile
    planned: tuple[PlannedCircuit, ...]
    circuits: tuple[Any, ...]
    durations_ns: tuple[float, ...]
    shots: int
    options: dict[str, Any]
    usage_s: float


@dataclass(frozen=True)
class IsaCircuit:
    circuit: Any
    measure_at_ns: float


@dataclass(frozen=True)
class Submitted:
    job_id: str
    profile: Profile
    planned: tuple[PlannedCircuit, ...]
    options: dict[str, Any]

    def save(self, path: Path) -> None:
        data = {
            "job_id": self.job_id,
            "profile": self.profile.to_dict(),
            "planned": [circuit.model_dump(mode="json") for circuit in self.planned],
            "options": self.options,
        }
        write_atomically(path, (json.dumps(data) + "\n").encode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> Submitted:
        data = json.loads(path.read_bytes())
        return cls(
            job_id=data["job_id"],
            profile=Profile.model_validate(data["profile"]),
            planned=tuple(PlannedCircuit.model_validate(c) for c in data["planned"]),
            options=data["options"],
        )


@dataclass(frozen=True)
class Binding:
    profile: Profile
    warning: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class Ran:
    job_id: str
    source: str
    run_at: datetime
    counts: tuple[dict[str, int], ...]
    timing: dict[str, Any]


def prepare(
    calibration: Calibration, open_backend: Callable[[bool], Any], shots: int
) -> tuple[Any, Batch]:
    profile = calibration(None)
    planned = plan(profile)
    fractional = any(op.name in FRACTIONAL_GATES for c in planned for op in c.ops)
    try:
        backend = open_backend(fractional)
        target = backend.target
        rep_delay = backend.configuration().default_rep_delay
    except Exception as exc:
        raise NoiseVaultError(
            f"could not open {profile.device.name} through your IBM Quantum account"
            f" ({_reason(exc)})",
            hint="run the same command again. The script did not submit a job",
        ) from None
    built = [isa_circuit(circuit, profile, target) for circuit in planned]
    circuits = tuple(b.circuit for b in built)
    return backend, Batch(
        profile=profile,
        planned=planned,
        circuits=circuits,
        durations_ns=tuple(b.measure_at_ns for b in built),
        shots=shots,
        options=sampler_options(shots, rep_delay),
        usage_s=usage_seconds(planned, circuits, target, rep_delay, shots),
    )


def isa_circuit(circuit: PlannedCircuit, profile: Profile, target: Any) -> IsaCircuit:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Delay

    def refuse(message: str, hint: str | None = None) -> NoiseVaultError:
        return NoiseVaultError(f"circuit {circuit.name}: {message}", hint=hint)

    dt_ns = target.dt * 1e9
    qc = QuantumCircuit(
        QuantumRegister(target.num_qubits, "q"),
        ClassicalRegister(len(circuit.qubits), CLBITS),
        name=circuit.name,
    )
    free = [0.0] * len(circuit.qubits)
    for op in circuit.ops:
        qubits = tuple(circuit.qubits[q] for q in op.qubits)
        name = "delay" if op.name == "delay" else gates.GATES[op.name].qiskit
        if name is None or not target.instruction_supported(name, qubits):
            raise refuse(f"the backend does not support {op.name} on {qubit_loci(qubits)}")
        if op.name == "delay":
            length = op.params[0] / dt_ns
            if _whole(length) is None:
                raise refuse(
                    f"the {op.params[0]:g} ns delay on {qubit_loci(qubits)} is {length:g} dt, and"
                    f" the backend times delays in whole dt of {dt_ns:g} ns"
                )
            if round(length) < target.min_length:
                raise refuse(
                    f"the {op.params[0]:g} ns delay on {qubit_loci(qubits)} is {round(length)} dt,"
                    f" shorter than the backend's minimum of {target.min_length} dt"
                )
            instruction = Delay(round(length), unit="dt")
        else:
            calibrated = float(_duration(profile, op, circuit.qubits)) / dt_ns
            length = _seconds(target, name, qubits) / target.dt
            if abs(length - calibrated) > 1e-6:
                raise refuse(
                    f"{op.name} on {qubit_loci(qubits)} lasts {_dt(calibrated, dt_ns)} in the"
                    f" calibration and {_dt(length, dt_ns)} on the backend, so the planned delays"
                    " would not fill the gaps",
                    hint="run the script again to plan from the current calibration",
                )
            operation = target.operation_from_name(name)
            instruction = operation.base_class(*op.params) if op.params else operation
        start = max(free[q] for q in op.qubits)
        if _whole(start / target.pulse_alignment) is None:
            raise refuse(
                f"{op.name} on {qubit_loci(qubits)} would start at {start:g} dt, off the backend's"
                f" grid of {target.pulse_alignment} dt"
            )
        qc.append(instruction, list(qubits))
        for q in op.qubits:
            free[q] = start + length
    end = max(free)
    if _whole(end / target.acquire_alignment) is None:
        raise refuse(
            f"the measurements would start at {end:g} dt, off the backend's grid of"
            f" {target.acquire_alignment} dt"
        )
    qc.barrier(list(circuit.qubits))
    for i, q in enumerate(circuit.qubits):
        qc.measure(q, i)
    return IsaCircuit(qc, end * dt_ns)


def sampler_options(shots: int, rep_delay: float) -> dict[str, Any]:
    """Every option the job sets.

    The client does not send an unset option, so the server uses its own default for that option.
    """
    options: dict[str, Any] = {"default_shots": shots}
    for path, value in {**SAMPLER_V2_OPTIONS, "execution.rep_delay": rep_delay}.items():
        *parents, leaf = path.split(".")
        node = options
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    options["experimental"] = {"execution": {"scheduler_timing": True}}
    return options


def usage_seconds(
    planned: Sequence[PlannedCircuit],
    circuits: Sequence[Any],
    target: Any,
    rep_delay: float,
    shots: int,
) -> float:
    """IBM's estimate, from https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time"""
    total = SUB_JOB_OVERHEAD_S
    for circuit, qc in zip(planned, circuits, strict=True):
        reset = max(_seconds(target, "reset", (q,)) for q in circuit.qubits)
        total += (qc.estimate_duration(target, unit="s") + reset + rep_delay) * shots
    return total


def submit(backend: Any, batch: Batch) -> Any:
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend, options=batch.options)
    try:
        return sampler.run(list(batch.circuits))
    except Exception as exc:
        raise NoiseVaultError(
            f"submitting the job failed ({_reason(exc)})",
            hint="check your IBM Quantum account for a new job before you run the script again",
        ) from None


def collect(job: Any, submitted: Submitted) -> Ran:
    from qiskit_ibm_runtime import RuntimeJobV2

    result = job.result()
    timing = {}
    for circuit, pub in zip(submitted.planned, result, strict=True):
        compiled = pub.metadata.get("compilation", {}).get("scheduler_timing", {})
        if "timing" in compiled:
            timing[circuit.name] = compiled["timing"]
    return Ran(
        job_id=submitted.job_id,
        source="hardware" if isinstance(job, RuntimeJobV2) else "simulated",
        run_at=started_at(job.metrics()),
        counts=tuple(getattr(pub.data, CLBITS).get_counts() for pub in result),
        timing=timing,
    )


def started_at(metrics: dict[str, Any]) -> datetime:
    """When the job started running, in UTC. IBM sends an ISO 8601 string, which the runtime
    reads as UTC when it has no offset. Local testing mode gives a naive datetime in local time,
    which ``astimezone`` reads as local time."""
    stamp = metrics["timestamps"]["running"]
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def bind(submitted: Submitted, ran: Ran, calibration: Calibration) -> Binding:
    planned = submitted.profile
    try:
        latest = calibration(ran.run_at)
    except Exception as exc:
        return Binding(
            planned,
            f"could not pull the calibration in effect when the job ran ({_reason(exc)}). The"
            f" counts bind to the planned calibration {planned.short_fingerprint}",
        )
    before, after = planned.device.calibrated_at, latest.device.calibrated_at
    if before is not None and after is not None and after < before:
        return Binding(
            planned,
            f"the calibration IBM returned for the time the job ran, {latest.short_fingerprint},"
            f" is older than the planned {planned.short_fingerprint}. The counts bind to the"
            " planned calibration",
        )
    problem = _unlike_the_plan(latest, submitted, ran)
    if problem is None:
        return Binding(latest)
    return Binding(
        planned,
        f"IBM recalibrated {planned.device.name} before the job ran"
        f" ({latest.short_fingerprint}), and {problem}. The counts bind to the planned"
        f" calibration {planned.short_fingerprint}",
        "run the script again for counts that match one calibration",
    )


def _unlike_the_plan(latest: Profile, submitted: Submitted, ran: Ran) -> str | None:
    try:
        _bind(latest, counts_file(latest, submitted, ran))
    except CountsError as exc:
        return f"nv compare would refuse the counts against it ({exc.message})"
    for circuit in submitted.planned:
        for op in circuit.ops:
            if op.name == "delay":
                continue
            before = _duration(submitted.profile, op, circuit.qubits)
            after = _duration(latest, op, circuit.qubits)
            if before != after:
                qubits = tuple(circuit.qubits[q] for q in op.qubits)
                return (
                    f"{op.name} on {qubit_loci(qubits)} now takes {float(after):g} ns, not"
                    f" {float(before):g} ns, so the submitted delays may not match the timeline"
                    " that ran"
                )
    return None


def counts_file(profile: Profile, submitted: Submitted, ran: Ran) -> MeasuredCounts:
    import qiskit
    import qiskit_ibm_runtime

    runtime, framework = qiskit_ibm_runtime.__version__, qiskit.__version__
    flags = {rule.flag: rule.required for rule in _RUN_RULES if rule.flag}
    return MeasuredCounts.model_validate(
        {
            "nv_counts": COUNTS_FORMAT,
            "source": ran.source,
            "profile": {"id": profile.id, "fingerprint": profile.calibration_fingerprint},
            "backend": profile.device.name,
            "run_at": ran.run_at,
            "bit_order": "qiskit",
            "execution": {
                "client": f"qiskit-ibm-runtime {runtime} SamplerV2, qiskit {framework}",
                **flags,
                "job_ids": [ran.job_id],
                "options": submitted.options,
            },
            "circuits": [
                {
                    "name": circuit.name,
                    "qubits": circuit.qubits,
                    "ops": circuit.ops,
                    "shots": sum(counts.values()),
                    "counts": counts,
                }
                for circuit, counts in zip(submitted.planned, ran.counts, strict=True)
            ],
        }
    )


def catalog_ref(profile: Profile) -> str | None:
    when = profile.device.calibrated_at
    if when is not None:
        for ref in (ref_on_day(profile.id, when), exact_ref(profile.id, when)):
            try:
                if catalog.resolve(ref).fingerprint == profile.fingerprint:
                    return ref
            except NoiseVaultError:
                continue
    return None


def run(
    *,
    shots: int,
    output: Path,
    pending: Path,
    calibration: Calibration,
    open_backend: Callable[[bool], Any],
    open_job: Callable[[str], Any],
    confirm: Callable[[str], bool] | None,
) -> MeasuredCounts | None:
    if pending.exists():
        submitted, job = _resume(pending, open_job)
        print(f"collecting job {submitted.job_id}, submitted earlier for {output}")
    else:
        backend, batch = prepare(calibration, open_backend, shots)
        print(summary(batch, output))
        print()
        device = batch.profile.device.name
        if confirm is not None and not confirm(f"Submit the job to {device}? [y/N] "):
            print("nothing submitted")
            return None
        try:
            _claim(pending)
        except FileExistsError:
            raise NoiseVaultError(
                f"{pending} exists, so the script did not submit a job",
                hint=_new_name_hint(pending),
            ) from None
        try:
            job = submit(backend, batch)
            submitted = Submitted(job.job_id(), batch.profile, batch.planned, batch.options)
            submitted.save(pending)
        except BaseException:
            pending.unlink()
            raise
        print(f"submitted job {submitted.job_id} to {device}")
    print("waiting for it to run")
    print("Ctrl-C stops waiting. Run the same command again to collect the job")
    again = f"run the same command again to collect them, or delete {pending} to submit a new job"
    uncollected = f"could not collect the counts of job {submitted.job_id}"
    try:
        try:
            ran = collect(job, submitted)
        except Exception as exc:
            raise NoiseVaultError(f"{uncollected} ({_reason(exc)})", hint=again) from None
        binding = bind(submitted, ran, calibration)
        bound = binding.profile
        measured = counts_file(bound, submitted, ran)
        _, timing, profile_file = _files(output)
        files: dict[Path, Callable[[Path], object]] = {output: measured.save}
        if ran.timing:
            files[timing] = lambda path: path.write_text(
                json.dumps(ran.timing, indent=1) + "\n", encoding="utf-8"
            )
        ref = catalog_ref(bound)
        if ref is None:
            ref = str(profile_file)
            files[profile_file] = bound.save
        try:
            _create(files)
        except FileExistsError as exc:
            raise NoiseVaultError(
                f"could not save the counts of job {submitted.job_id}, because {exc.filename}"
                " exists",
                hint=_new_name_hint(pending),
            ) from None
        except OSError as exc:
            raise NoiseVaultError(f"{uncollected} ({_reason(exc)})", hint=again) from None
    except KeyboardInterrupt:
        raise NoiseVaultError(
            f"stopped before the script saved the counts of job {submitted.job_id}", hint=again
        ) from None
    pending.unlink(missing_ok=True)
    if binding.warning:
        print(f"warning: {binding.warning}", file=sys.stderr)
        if binding.hint:
            print(f"hint: {binding.hint}", file=sys.stderr)
    elif bound.fingerprint != submitted.profile.fingerprint:
        print(
            f"IBM recalibrated {bound.device.name} before the job ran. The counts bind to the new"
            " calibration, which keeps every planned gate duration"
        )
    lines = [
        ("run", f"{ran.run_at:%Y-%m-%d %H:%MZ}, job {ran.job_id}"),
        ("counts file", str(output)),
        (
            "calibration",
            f"{ref_on_day(bound.id, bound.device.calibrated_at)} {bound.short_fingerprint}",
        ),
    ]
    if ran.timing:
        lines.append(("timing", f"{timing}, how IBM scheduled each circuit"))
    lines.append(("next", shlex.join(["nv", "compare", ref, str(output)])))
    print()
    print("\n".join(f"{label:<{LABEL}}{value}" for label, value in lines))
    return measured


def _resume(pending: Path, open_job: Callable[[str], Any]) -> tuple[Submitted, Any]:
    try:
        submitted = Submitted.load(pending)
    except Exception as exc:
        raise NoiseVaultError(
            f"{pending} is damaged ({_reason(exc)})", hint="delete it to submit a new job"
        ) from None
    try:
        return submitted, open_job(submitted.job_id)
    except Exception as exc:
        raise NoiseVaultError(
            f"could not open job {submitted.job_id} ({_reason(exc)})",
            hint=f"delete {pending} to submit a new job",
        ) from None


def summary(batch: Batch, output: Path) -> str:
    profile = batch.profile
    chain = tuple(dict.fromkeys(q for c in batch.planned for q in c.qubits))
    rows = [("circuit", "qubits", "gates", "duration")] + [
        (
            c.name,
            "-".join(map(str, c.qubits)),
            str(sum(op.name != "delay" for op in c.ops)),
            f"{ns:.0f} ns",
        )
        for c, ns in zip(batch.planned, batch.durations_ns, strict=True)
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    table = [
        f"{a:<{widths[0]}}  {b:<{widths[1]}}  {c:>{widths[2]}}  {d:>{widths[3]}}"
        for a, b, c, d in rows
    ]
    labels = [
        ("shots", f"{batch.shots} per circuit, {len(batch.planned)} circuits in one job"),
        ("usage", f"about {batch.usage_s:.1f} s of QPU time (IBM's estimate)"),
        ("counts file", str(output)),
    ]
    return "\n".join(
        [
            f"{ref_on_day(profile.id, profile.device.calibrated_at)}"
            f" {profile.short_fingerprint} on {qubit_loci(chain)}",
            "",
            *table,
            "",
            *(f"{label:<{LABEL}}{value}" for label, value in labels),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError:
        return _fail("run_on_ibm.py needs qiskit-ibm-runtime", install_hint("ibm"))
    try:
        pending = _pending(args.output, args.collect)
        _writable(args.output, pending)
        service = _service()
        measured = run(
            shots=args.shots,
            output=args.output,
            pending=pending,
            calibration=lambda at: nv.pull(args.device, source="ibm-account", at=at),
            open_backend=lambda fractional: service.backend(
                args.device, use_fractional_gates=fractional
            ),
            open_job=service.job,
            confirm=None if args.yes else _ask,
        )
    except NoiseVaultError as exc:
        return _fail(exc.message, exc.hint)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    return 0 if measured is not None else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_on_ibm.py",
        description="Run the nv compare circuits on an IBM device and save the counts.",
    )
    parser.add_argument("device", metavar="DEVICE", help="the IBM device, such as ibm_kingston")
    parser.add_argument(
        "--shots",
        metavar="N",
        type=_positive,
        default=DEFAULT_SHOTS,
        help=f"shots per circuit (default {DEFAULT_SHOTS})",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        type=Path,
        required=True,
        help="the counts file to write, which must not exist yet",
    )
    parser.add_argument(
        "--collect",
        metavar="JOB_FILE",
        type=Path,
        help="collect the job that JOB_FILE records instead of submitting a new job",
    )
    parser.add_argument("--yes", action="store_true", help="submit without asking")
    return parser


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive whole number")
    return value


def _pending(output: Path, collect: Path | None) -> Path:
    if collect is None:
        return _beside(output, ".job.json")
    if not collect.exists():
        raise NoiseVaultError(
            f"no job file {collect}",
            hint="give --collect the .job.json file that the script saved beside the counts file",
        )
    return collect


def _writable(output: Path, pending: Path) -> None:
    for path in _files(output):
        if path.exists():
            raise NoiseVaultError(f"{path} exists", hint=_new_name_hint(pending))
    folder = output.parent
    if not folder.is_dir():
        raise NoiseVaultError(f"no folder {folder}", hint="create it, or give -o another path")
    if not os.access(folder, os.W_OK):
        raise NoiseVaultError(f"cannot write to {folder}", hint="give -o a folder you can write to")


def _service() -> Any:
    try:
        return ibm_account._service()
    except NoiseVaultError as exc:
        raise NoiseVaultError(exc.message, hint=ACCOUNT_SETUP) from None


def _ask(prompt: str) -> bool:
    if not sys.stdin.isatty():
        raise NoiseVaultError(
            "cannot ask before submitting, because standard input is not a terminal",
            hint="give --yes to submit without asking",
        )
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _fail(message: str, hint: str | None = None) -> int:
    print(f"error: {message}", file=sys.stderr)
    if hint:
        print(f"hint: {hint}", file=sys.stderr)
    return 1


def _reason(exc: BaseException) -> str:
    return exc.message if isinstance(exc, NoiseVaultError) else str(exc) or type(exc).__name__


def _warning_line(message: Warning | str, *_: Any, **__: Any) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _seconds(target: Any, name: str, qubits: tuple[int, ...]) -> float:
    props = target[name].get(qubits) if name in target.operation_names else None
    return 0.0 if props is None or props.duration is None else props.duration


def _whole(value: float) -> int | None:
    n = round(value)
    return n if abs(value - n) <= 1e-6 else None


def _dt(length: float, dt_ns: float) -> str:
    return f"{length:g} dt ({length * dt_ns:g} ns)"


def _new_name_hint(pending: Path) -> str:
    if pending.exists():
        return (
            f"run the same command with --collect {pending} and a new -o file name. The script"
            " never replaces a file"
        )
    return "give -o a new file name. The script never replaces a file"


def _files(output: Path) -> tuple[Path, Path, Path]:
    return output, _beside(output, ".timing.json"), _beside(output, ".profile.json")


def _claim(path: Path) -> None:
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))


def _create(files: dict[Path, Callable[[Path], object]]) -> None:
    """Save every file or none. For a path that exists, keep that file and raise FileExistsError.

    ``os.open`` with ``O_EXCL`` creates each file empty, and the save replaces only that empty file.
    ``os.link`` needs no empty file, but exFAT does not support hard links.
    """
    created: list[Path] = []
    try:
        for path, save in files.items():
            _claim(path)
            created.append(path)
            save(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _beside(output: Path, suffix: str) -> Path:
    stem = output.name.removesuffix(".gz").removesuffix(".json").removesuffix(".counts")
    return output.with_name(stem + suffix)


if __name__ == "__main__":
    warnings.showwarning = _warning_line
    sys.exit(main())
