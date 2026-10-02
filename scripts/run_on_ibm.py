"""Run the nv compare circuits on an IBM device and save the counts for nv compare.

    python scripts/run_on_ibm.py ibm_kingston --shots 4000 -o kingston-0416.counts.json

The script pulls the device's calibration through your IBM Quantum account and plans the
circuits with ``noisevault.counts.plan``. It checks every op and delay against the backend, shows
IBM's usage estimate, and asks before it submits one SamplerV2 job. When the job has run, it
writes a counts file bound to the calibration in effect at that time and prints the
``nv compare`` command to run next. If the wait for the job is interrupted, run the same command
again. It then collects the submitted job instead of submitting another.

It needs the ``ibm`` extra and an IBM Quantum account, either saved with
``QiskitRuntimeService.save_account`` or given as an API key in ``IBM_QUANTUM_TOKEN``.
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
    _SAMPLER_V2_OPTIONS,
    COUNTS_FORMAT,
    MeasuredCounts,
    PlannedCircuit,
    _duration,
    plan,
)
from noisevault.errors import CountsError, NoiseVaultError, install_hint  # noqa: E402
from noisevault.profile import Profile, _calibration_fingerprint, write_atomically  # noqa: E402
from noisevault.sources import ibm_account  # noqa: E402

DEFAULT_SHOTS = 4000
FRACTIONAL_GATES = frozenset({"rx", "rzz"})
CLBITS = "meas"
ACCOUNT_SETUP = (
    "set IBM_QUANTUM_TOKEN to your IBM Quantum API key, or save the key once with"
    " QiskitRuntimeService.save_account(token=...)"
)
LABEL = 16

Calibration = Callable[[datetime | None], Profile]


@dataclass(frozen=True)
class Batch:
    """One SamplerV2 job: the planned circuits, the same circuits built for the backend, and the
    options it is submitted with."""

    profile: Profile
    planned: tuple[PlannedCircuit, ...]
    circuits: tuple[Any, ...]
    durations_ns: tuple[float, ...]
    shots: int
    options: dict[str, Any]
    usage_s: float


@dataclass(frozen=True)
class Submitted:
    """A submitted job and what collecting it needs. Until its counts are saved, a file beside
    the counts file holds it, so running the same command again collects this job instead of
    submitting another."""

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
    """The calibration the counts bind to. ``warning`` says why when it is not the one in effect
    when the job ran."""

    profile: Profile
    warning: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class Ran:
    """What the finished job returned: the counts of each circuit with classical bit 0 on the
    right, when the job started running, and IBM's schedule of each circuit if it sent one."""

    job_id: str
    source: str
    run_at: datetime
    counts: tuple[dict[str, int], ...]
    timing: dict[str, Any]


def prepare(
    calibration: Calibration, open_backend: Callable[[bool], Any], shots: int
) -> tuple[Any, Batch]:
    """Plan from the calibration in effect now and build each circuit for the backend."""
    profile = calibration(None)
    planned = plan(profile)
    backend = open_backend(any(op.name in FRACTIONAL_GATES for c in planned for op in c.ops))
    target = backend.target
    built = [isa_circuit(circuit, profile, target) for circuit in planned]
    circuits = tuple(qc for qc, _ in built)
    rep_delay = backend.configuration().default_rep_delay
    return backend, Batch(
        profile=profile,
        planned=planned,
        circuits=circuits,
        durations_ns=tuple(ns for _, ns in built),
        shots=shots,
        options=sampler_options(shots, rep_delay),
        usage_s=usage_seconds(planned, circuits, target, rep_delay, shots),
    )


def isa_circuit(circuit: PlannedCircuit, profile: Profile, target: Any) -> tuple[Any, float]:
    """``circuit`` on the backend's physical qubits with every delay in whole dt, then a barrier
    and one measurement per circuit qubit, and its duration before the measurements in ns.

    Refuses an op the backend does not support on its qubits. Refuses a gate whose backend
    duration differs from the calibration's, because the plan timed the delays from the
    calibration. Refuses a delay that is not a whole number of dt or is shorter than the backend
    allows, and an op or measurement that would start off the backend's timing grid.
    """
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
            raise refuse(f"the backend does not support {op.name} on {_on(qubits)}")
        if op.name == "delay":
            length = op.params[0] / dt_ns
            if _whole(length) is None:
                raise refuse(
                    f"the {op.params[0]:g} ns delay on {_on(qubits)} is {length:g} dt, and the"
                    f" backend times delays in whole dt of {dt_ns:g} ns"
                )
            if round(length) < target.min_length:
                raise refuse(
                    f"the {op.params[0]:g} ns delay on {_on(qubits)} is {round(length)} dt,"
                    f" shorter than the backend's minimum of {target.min_length} dt"
                )
            instruction = Delay(round(length), unit="dt")
        else:
            calibrated = float(_duration(profile, op, circuit.qubits)) / dt_ns
            length = _seconds(target, name, qubits) / target.dt
            if abs(length - calibrated) > 1e-6:
                raise refuse(
                    f"{op.name} on {_on(qubits)} lasts {_dt(calibrated, dt_ns)} in the calibration"
                    f" and {_dt(length, dt_ns)} on the backend, so the planned delays would not"
                    " fill the gaps",
                    hint="run the script again to plan from the current calibration",
                )
            operation = target.operation_from_name(name)
            instruction = operation.base_class(*op.params) if op.params else operation
        start = max(free[q] for q in op.qubits)
        if _whole(start / target.pulse_alignment) is None:
            raise refuse(
                f"{op.name} on {_on(qubits)} would start at {start:g} dt, off the backend's"
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
    return qc, end * dt_ns


def sampler_options(shots: int, rep_delay: float) -> dict[str, Any]:
    """Every SamplerV2 option the counts format checks, set explicitly, plus the shots and the
    backend's default repetition delay. The client sends no unset option, and the server would
    choose its own default for it. ``scheduler_timing`` asks IBM to return how it scheduled each
    circuit."""
    options: dict[str, Any] = {"default_shots": shots}
    for path, value in {**_SAMPLER_V2_OPTIONS, "execution.rep_delay": rep_delay}.items():
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
    """IBM's estimate of the job's QPU usage: 2 s, plus each circuit's length, a reset and the
    repetition delay for every shot."""
    total = 2.0
    for circuit, qc in zip(planned, circuits, strict=True):
        reset = max(_seconds(target, "reset", (q,)) for q in circuit.qubits)
        total += (qc.estimate_duration(target, unit="s") + reset + rep_delay) * shots
    return total


def submit(backend: Any, batch: Batch) -> Any:
    """Submit the circuits as one SamplerV2 job."""
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
    """Wait for the job and read its counts, its start time and IBM's schedule."""
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
    """The calibration the counts bind to: the one in effect when the job ran, if nv compare
    accepts the counts against it and every gate keeps its planned duration. Otherwise the
    planned calibration, with a warning that says why."""
    planned = submitted.profile
    try:
        latest = calibration(ran.run_at)
    except Exception as exc:
        return Binding(
            planned,
            f"could not pull the calibration in effect when the job ran ({_reason(exc)}); the"
            f" counts bind to the planned calibration {planned.short_fingerprint}",
        )
    before, after = planned.device.calibrated_at, latest.device.calibrated_at
    if before is not None and after is not None and after < before:
        return Binding(
            planned,
            f"the calibration IBM returned for the time the job ran, {latest.short_fingerprint},"
            f" is older than the planned {planned.short_fingerprint}; the counts bind to the"
            " planned calibration",
        )
    problem = _unlike_the_plan(latest, submitted, ran)
    if problem is None:
        return Binding(latest)
    return Binding(
        planned,
        f"IBM recalibrated {planned.device.name} before the job ran"
        f" ({latest.short_fingerprint}), and {problem}; the counts bind to the planned"
        f" calibration {planned.short_fingerprint}",
        "run the script again for counts that match one calibration",
    )


def _unlike_the_plan(latest: Profile, submitted: Submitted, ran: Ran) -> str | None:
    """Why the counts cannot bind to ``latest``, or None when they can."""
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
                    f"{op.name} on {_on(qubits)} now takes {float(after):g} ns, not"
                    f" {float(before):g} ns, so the submitted delays may not match the timeline"
                    " that ran"
                )
    return None


def counts_file(profile: Profile, submitted: Submitted, ran: Ran) -> MeasuredCounts:
    """The run as a counts file bound to ``profile``. Each circuit's shots are the sum of its
    counts, so the file records the shots that ran."""
    import qiskit
    import qiskit_ibm_runtime

    runtime, framework = qiskit_ibm_runtime.__version__, qiskit.__version__
    flags = {rule.flag: rule.required for rule in _RUN_RULES if rule.flag}
    return MeasuredCounts.model_validate(
        {
            "nv_counts": COUNTS_FORMAT,
            "source": ran.source,
            "profile": {"id": profile.id, "fingerprint": _calibration_fingerprint(profile)},
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


def ref_for(profile: Profile, output: Path) -> str:
    """The shortest ref that loads ``profile``. When no catalog ref loads it, ``ref_for`` saves
    the profile beside the counts and returns that file's path."""
    when = profile.device.calibrated_at
    if when is not None:
        stamp = when.isoformat().replace("+00:00", "Z")
        for ref in (_ref(profile), f"{profile.id}@{stamp}"):
            try:
                if catalog.resolve(ref).fingerprint == profile.fingerprint:
                    return ref
            except NoiseVaultError:
                continue
    path = _beside(output, ".profile.json")
    profile.save(path)
    return str(path)


def run(
    *,
    shots: int,
    output: Path,
    calibration: Calibration,
    open_backend: Callable[[bool], Any],
    open_job: Callable[[str], Any],
    confirm: Callable[[str], bool] | None,
) -> MeasuredCounts | None:
    """Plan, check, ask, submit, wait, bind and save. When a job submitted earlier for ``output``
    has no saved counts yet, ``run`` collects that job instead. Returns None when the user
    declines."""
    pending = _beside(output, ".job.json")
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
        job = submit(backend, batch)
        submitted = Submitted(job.job_id(), batch.profile, batch.planned, batch.options)
        submitted.save(pending)
        print(f"submitted job {submitted.job_id} to {device}")
    print("waiting for it to run")
    print("Ctrl-C stops waiting; run the same command again to collect the job")
    again = f"run the same command again to collect them, or delete {pending} to submit a new job"
    try:
        ran = collect(job, submitted)
        binding = bind(submitted, ran, calibration)
        measured = counts_file(binding.profile, submitted, ran)
        measured.save(output)
    except KeyboardInterrupt:
        raise NoiseVaultError(
            f"stopped before the counts of job {submitted.job_id} were saved", hint=again
        ) from None
    except Exception as exc:
        raise NoiseVaultError(
            f"could not collect the counts of job {submitted.job_id} ({_reason(exc)})", hint=again
        ) from None
    pending.unlink()
    bound = binding.profile
    if binding.warning:
        print(f"warning: {binding.warning}", file=sys.stderr)
        if binding.hint:
            print(f"hint: {binding.hint}", file=sys.stderr)
    elif bound.fingerprint != submitted.profile.fingerprint:
        print(
            f"IBM recalibrated {bound.device.name} before the job ran; the counts bind to the new"
            " calibration, which keeps every planned gate duration"
        )
    lines = [
        ("run", f"{ran.run_at:%Y-%m-%d %H:%MZ}, job {ran.job_id}"),
        ("counts file", str(output)),
        ("calibration", f"{_ref(bound)} {bound.short_fingerprint}"),
    ]
    if ran.timing:
        timing = _beside(output, ".timing.json")
        timing.write_text(json.dumps(ran.timing, indent=1) + "\n", encoding="utf-8")
        lines.append(("timing", f"{timing}, how IBM scheduled each circuit"))
    lines.append(("next", shlex.join(["nv", "compare", ref_for(bound, output), str(output)])))
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
    """What the job runs and costs, for the user to read before submitting."""
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
            f"{_ref(profile)} {profile.short_fingerprint} on {_on(chain)}",
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
        return _fail("scripts/run_on_ibm.py needs qiskit-ibm-runtime", install_hint("ibm"))
    try:
        _writable(args.output)
        service = _service()
        measured = run(
            shots=args.shots,
            output=args.output,
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
        prog="python scripts/run_on_ibm.py",
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
        help="the counts file to write; it must not exist yet",
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


def _writable(output: Path) -> None:
    """Refuse an output the run could not write, before the job spends any usage."""
    if output.exists():
        raise NoiseVaultError(
            f"{output} exists", hint="give -o a new file name; the script never replaces counts"
        )
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
            hint="pass --yes to submit without asking",
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
    """The backend's duration of ``name`` on ``qubits`` in seconds, or 0 when it states none."""
    props = target[name].get(qubits) if name in target.operation_names else None
    return 0.0 if props is None or props.duration is None else props.duration


def _whole(value: float) -> int | None:
    n = round(value)
    return n if abs(value - n) <= 1e-6 else None


def _dt(length: float, dt_ns: float) -> str:
    return f"{length:g} dt ({length * dt_ns:g} ns)"


def _on(qubits: Sequence[int]) -> str:
    return f"qubit {qubits[0]}" if len(qubits) == 1 else "qubits " + "-".join(map(str, qubits))


def _ref(profile: Profile) -> str:
    when = profile.device.calibrated_at
    return profile.id if when is None else f"{profile.id}@{when.date().isoformat()}"


def _beside(output: Path, suffix: str) -> Path:
    stem = output.name.removesuffix(".gz").removesuffix(".json").removesuffix(".counts")
    return output.with_name(stem + suffix)


if __name__ == "__main__":
    warnings.showwarning = _warning_line
    sys.exit(main())
