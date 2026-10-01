"""Command line: ``noisevault`` (alias ``nv``).

Every command prints plain results on stdout and ``error:``, ``hint:`` and ``warning:`` lines
on stderr. Expected failures exit with status 1 and no traceback. ``--json`` output is for
machines and never contains color. NO_COLOR and COLUMNS are read at each invocation.
"""

from __future__ import annotations

import errno
import gzip
import json
import os
import platform
import re
import statistics
import sys
import warnings
import zlib
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, NoReturn, get_args

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from . import __version__, catalog
from .diff import METRICS, describe_delta, fmt_error, fmt_metric, fmt_relative, fmt_time
from .errors import (
    REPOSITORY,
    FingerprintMismatch,
    NoiseVaultError,
    ProfileNotFound,
    did_you_mean,
    install_hint,
)
from .profile import Profile, Ref, Technology, gate_loci, json_schema, load_file, parse_ref
from .table import GateNoise

# Click's UsageError; typer exports only this subclass of it.
_USAGE_ERROR = typer.BadParameter.__mro__[1]
# Raised by a bare `nv` once the help is printed; older click has no such class.
_NO_ARGS = getattr(sys.modules[_USAGE_ERROR.__module__], "NoArgsIsHelpError", ())


class _Commands(TyperGroup):
    """A usage mistake or an unexpected failure ends in one ``error:`` line, never a box or a
    traceback. NOISEVAULT_DEBUG=1 lets an unexpected failure raise."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except _NO_ARGS as exc:
            if help_text := exc.format_message():  # empty when rich has printed it already
                typer.echo(help_text)
            raise typer.Exit() from None
        except _USAGE_ERROR as exc:
            _usage_error(exc)

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except _USAGE_ERROR as exc:
            _usage_error(exc)
        except Exception as exc:
            # Exit and Abort, from typer or click, carry an exit code: they are control flow.
            if hasattr(exc, "exit_code") or os.environ.get("NOISEVAULT_DEBUG"):
                raise
            err.print(
                f"error: unexpected {type(exc).__name__}: {exc}; please report this bug at"
                f" {REPOSITORY}/issues (NOISEVAULT_DEBUG=1 shows the traceback)",
                markup=False,
            )
            raise typer.Exit(1) from None


def _usage_error(exc: Any) -> NoReturn:
    message = exc.format_message().rstrip(".").replace(". Did you mean", "; did you mean")
    message = message[:1].lower() + message[1:]
    if not re.search(r"did you mean|possible options", message, re.I):
        message += f"; see `{exc.ctx.command_path if exc.ctx else 'nv'} --help`"
    err.print(f"error: {message}", markup=False)
    raise typer.Exit(2) from None


app = typer.Typer(
    cls=_Commands,
    no_args_is_help=True,
    add_completion=False,
    help="NoiseVault: real device noise, pinned and portable.",
)
out = Console(highlight=False)
err = Console(stderr=True, highlight=False, soft_wrap=True)

# Package doctor reports -> the extra that installs it. pymatching is in no extra: only the QEC
# recipe uses it.
_PACKAGES: dict[str, str | None] = {
    "qiskit": "qiskit",
    "qiskit-aer": "qiskit",
    "qiskit-ibm-runtime": "ibm",
    "cirq-core": "cirq",
    "cirq-google": "google",
    "pennylane": "pennylane",
    "stim": "stim",
    "pymatching": None,
}
_INSTALL_ALL = install_hint("all")
_SOURCE_LABELS = {  # short forms of provenance.source_kind for the list table
    "package_snapshot": "package",
    "public_api": "public API",
    "account_api": "account",
    "user_file": "your file",
    "published_data": "published",
    "vendor_sample": "sample",
    "hand_written": "by hand",
}
# What to do next, by failure type; the first match wins.
_HINTS: tuple[tuple[type[BaseException], str], ...] = (
    (ProfileNotFound, "run `nv list` to see every profile you can load offline"),
    (FingerprintMismatch, "load the ref without a pin to see what it holds now"),
    (ImportError, f"install the frameworks: {_INSTALL_ALL}"),
    (ValidationError, "run `nv validate FILE` to list every problem in the file"),
    (FileNotFoundError, "check the path, or give a profile id such as ibm_fez"),
)
_EXPECTED = (
    NoiseVaultError,
    ValidationError,
    OSError,
    ValueError,
    ImportError,
    EOFError,
    zlib.error,
)


def _version(value: bool) -> None:
    if value:
        out.print(f"noisevault {__version__}", markup=False)
        raise typer.Exit()


@app.callback()
def main(
    _: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Print the version and exit."
        ),
    ] = False,
) -> None:
    """NoiseVault: real device noise, pinned and portable."""
    global out, err
    out = Console(highlight=False)
    err = Console(stderr=True, highlight=False, soft_wrap=True)


# list ---------------------------------------------------------------------------------------


@app.command("list")
def list_profiles(
    tech: Annotated[
        str | None, typer.Option("--tech", help="Only this technology, e.g. trapped_ion.")
    ] = None,
    vendor: Annotated[str | None, typer.Option("--vendor", help="Only this vendor.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """List the profiles you can load offline: the bundled set and your vault."""
    with _friendly():
        technologies = get_args(Technology)
        if tech is not None:
            tech = tech.lower().replace("-", "_")
        if tech is not None and tech not in technologies:
            guess = did_you_mean(tech, technologies)
            raise ValueError(f"--tech {tech!r}: {guess}choose from {', '.join(technologies)}")
        vendors = sorted({i.vendor for i in catalog.profiles() if i.vendor})
        if vendor is not None:
            vendor = next((v for v in vendors if v.lower() == vendor.lower()), vendor)
        if vendor is not None and vendor not in vendors:
            guess = did_you_mean(vendor, vendors)
            raise ValueError(f"--vendor {vendor!r}: {guess}choose from {', '.join(vendors)}")
        infos = catalog.profiles(technology=tech, vendor=vendor)
        rows = [_list_row(info) for info in infos]
        if as_json:
            _echo_json(rows)
            return
        if not rows:
            wanted = " ".join(x for x in (vendor, tech) if x)
            raise ProfileNotFound(f"no {wanted} profile yet; run `nv list` to see them all")
        same_day: dict[tuple[str, Any], list[datetime]] = {}
        for info in infos:
            if info.calibrated_at:
                same_day.setdefault((info.id, info.calibrated_at.date()), []).append(
                    info.calibrated_at
                )
        for row, info in zip(rows, infos, strict=True):
            stamp = info.calibrated_at
            row["when"] = _when(stamp, same_day[info.id, stamp.date()]) if stamp else "undated"
        # Grouped under technology headings so the table fits 80 columns. Only processor and
        # license may wrap; the other cells are short and must stay whole.
        table = Table(box=None, pad_edge=False, header_style="bold")
        for column in ("id", "date", "qubits", "processor", "source", "license"):
            table.add_column(
                column,
                justify="right" if column == "qubits" else "left",
                no_wrap=column not in ("processor", "license"),
                overflow="fold",
            )
        for technology in sorted({row["technology"] for row in rows}):
            table.add_row(f"[bold]{technology}[/bold]")
            group = [r for r in rows if r["technology"] == technology]
            for row in sorted(group, key=lambda r: r["location"] != "vault"):
                mark = "* " if row["location"] == "vault" else "  "
                table.add_row(
                    f"{mark}{row['id']}",
                    row["when"],
                    str(row["num_qubits"]),
                    row["processor"] or "-",
                    _SOURCE_LABELS.get(row["source_kind"], row["source_kind"] or "-"),
                    (row["license"] or "-").split(" (")[0],
                )
        out.print(table)
        in_vault = sum(r["location"] == "vault" for r in rows)
        if in_vault:
            out.print("* in your vault (`nv doctor` shows its folder)", markup=False)
        out.print(
            f"{len(rows)} profiles. Load one with nv.load('<id>'); see one with `nv show <id>`."
        )


def _when(stamp: datetime, same_day: list[datetime]) -> str:
    """The date, and under it as much of the time as tells apart one device's calibrations that
    day. Under, not beside: a wider date column would wrap every processor at 80 columns."""
    day = stamp.date().isoformat()
    if len(same_day) < 2:
        return day
    clock = "%H:%M"
    if len({s.strftime(clock) for s in same_day}) < len(same_day):
        clock = "%H:%M:%S"
    return f"{day}\n{stamp.astimezone(UTC).strftime(clock)} UTC"


def _list_row(info: catalog.ProfileInfo) -> dict[str, Any]:
    return {
        "ref": info.ref,
        "id": info.id,
        "date": info.calibrated_at.date().isoformat() if info.calibrated_at else None,
        "calibrated_at": _iso(info.calibrated_at),
        "technology": info.technology,
        "vendor": info.vendor,
        "num_qubits": info.num_qubits,
        "processor": info.processor,
        "source_kind": info.source_kind,
        "data_kind": info.data_kind,
        "license": info.license,
        "redistributable": info.redistributable,
        "location": info.location,
        "fingerprint": info.fingerprint,
    }


# show ---------------------------------------------------------------------------------------


@app.command()
def show(
    ref: Annotated[str, typer.Argument(help="Profile id (ibm_fez), id@date, or a file path.")],
    qubits: Annotated[
        str | None, typer.Option("--qubits", help="Also list these qubits, e.g. 0,1,2.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Show a profile: device, natives, coherence, readout, provenance and fingerprint."""
    with _friendly():
        profile = _load(ref)
        data = card(profile)
        indices = _parse_qubits(qubits, profile) if qubits is not None else None
        if indices is not None:
            data["qubits"] = [_qubit_row(profile, q) for q in indices]
        if as_json:
            _echo_json(data)
            return
        _print_card(data)
        if indices is not None:
            _print_qubits(data["qubits"])


def card(profile: Profile) -> dict[str, Any]:
    """The facts ``nv show`` prints, as JSON-ready data."""
    dev, prov, table = profile.device, profile.provenance, profile.table
    enabled = [table.qubit(i) for i in range(dev.num_qubits) if not table.qubit(i).disabled]
    readout = [q.readout for q in enabled if q.readout is not None]
    return {
        "ref": f"{profile.id}@{_iso(dev.calibrated_at)}" if dev.calibrated_at else profile.id,
        "id": profile.id,
        "calibrated_at": _iso(dev.calibrated_at),
        "vendor": dev.vendor,
        "processor": dev.processor,
        "technology": dev.technology,
        "num_qubits": dev.num_qubits,
        "connectivity": _connectivity(profile),
        "natives": [_native(profile, name) for name in profile.gates],
        "median_t1_us": _median([q.t1_ns / 1000 for q in enabled if q.t1_ns is not None]),
        "median_t2_us": _median([q.t2_ns / 1000 for q in enabled if q.t2_ns is not None]),
        "median_readout_error": _median([(a + b) / 2 for a, b in readout]),
        "median_p1_given_0": _median([a for a, _ in readout]),
        "median_p0_given_1": _median([b for _, b in readout]),
        "disabled_qubits": [i for i in range(dev.num_qubits) if table.qubit(i).disabled],
        "provenance": prov.model_dump(mode="json", exclude_none=True, exclude={"notes", "extra"}),
        "fingerprint": profile.fingerprint,
        "short_fingerprint": profile.short_fingerprint,
        "assumptions": _assumptions(profile),
        "notes": list(prov.notes),
        "effects": _effects(profile),
    }


def _assumptions(profile: Profile) -> list[str]:
    """Each gate's default assumption, then the records that state a different one."""
    overrides: dict[tuple[str, str], list[tuple[int, ...]]] = {}
    for record in profile.calibrations:
        if record.assumption and record.assumption != profile.gates[record.gate].assumption:
            overrides.setdefault((record.gate, record.assumption), []).append(record.qubits)
    lines = []
    for name, spec in profile.gates.items():
        if spec.assumption:
            lines.append(f"{name}: {spec.assumption}")
        for (gate, text), loci in overrides.items():
            if gate == name:
                where = (
                    ", ".join("-".join(map(str, q)) for q in loci)
                    if len(loci) <= 3
                    else f"({len(loci)} records)"
                )
                lines.append(f"{name} {where}: {text}")
    return lines


def _connectivity(profile: Profile) -> dict[str, Any]:
    table = profile.table
    if table.all_to_all:
        return {"kind": "all_to_all"}
    edges = table.edges()
    degree = [0] * table.num_qubits
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return {
        "kind": "edges",
        "edges": len(edges),
        "directed": profile.connectivity.directed,  # type: ignore[union-attr]
        "min_degree": min(degree),
        "max_degree": max(degree),
    }


def _native(profile: Profile, name: str) -> dict[str, Any]:
    """Median error and duration over the loci where the gate runs, and its loci by state."""
    found = gate_loci(profile, name)
    states = Counter(g.state for g in found)
    usable = [g for g in found if g.state != "disabled"]
    return {
        "gate": name,
        "qubits": profile.table.arity(name),
        "virtual": bool(found) and states["ideal"] == len(found),
        "median_avg_infidelity": _median(
            [g.avg_infidelity for g in usable if g.avg_infidelity is not None]
        ),
        "median_duration_ns": _median([g.duration_ns for g in usable if g.duration_ns is not None]),
        "records": sum(r.gate == name for r in profile.calibrations),
        "disabled": states["disabled"],
        "loci": {
            state: states[state] for state in ("calibrated", "ideal", "uncalibrated", "disabled")
        },
    }


def _effects(profile: Profile) -> list[str]:
    counts: dict[str, int] = {}
    for effect in profile.effects:
        key = f"{effect.type} on {effect.gate or effect.on}"
        counts[key] = counts.get(key, 0) + 1
    return [f"{key} ({n} records)" if n > 1 else key for key, n in counts.items()]


def _print_card(data: dict[str, Any], *, brief: bool = False) -> None:
    out.print(f"[bold]{data['ref']}[/bold]  {data['short_fingerprint']}", soft_wrap=True)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    device = ", ".join(
        x
        for x in (
            data["vendor"],
            data["processor"],
            data["technology"],
            f"{data['num_qubits']} qubits",
        )
        if x
    )
    grid.add_row("device", device)
    grid.add_row("connectivity", _describe_connectivity(data["connectivity"]))
    grid.add_row("natives", _natives_table(data["natives"]))
    grid.add_row("coherence", _coherence(data))
    grid.add_row("readout", _readout(data))
    if data["disabled_qubits"]:
        grid.add_row("disabled", "qubits " + ", ".join(map(str, data["disabled_qubits"])))
    prov = data["provenance"]
    kind = prov.get("source_kind") or "source kind unknown"
    grid.add_row(
        "provenance",
        f"{prov.get('data_kind', 'unknown')} data, {kind}: {prov.get('source') or 'unknown'}",
    )
    grid.add_row(
        "license",
        f"{prov.get('license') or 'unknown'}, redistributable {prov.get('redistributable')}",
    )
    if prov.get("attribution"):
        grid.add_row("attribution", prov["attribution"])
    grid.add_row("fingerprint", data["fingerprint"])
    if not brief:
        for label, items in (
            ("assumptions", data["assumptions"]),
            ("notes", data["notes"]),
            ("not modeled", data["effects"]),
        ):
            for i, item in enumerate(items):
                grid.add_row(label if i == 0 else "", item)
    out.print(grid)


def _describe_connectivity(conn: dict[str, Any]) -> str:
    if conn["kind"] == "all_to_all":
        return "all-to-all"
    kind = "directed edges" if conn["directed"] else "edges"
    return (
        f"{conn['edges']} {kind}, {conn['min_degree']} to {conn['max_degree']} neighbors per qubit"
    )


def _natives_table(natives: list[dict[str, Any]]) -> Table:
    table = Table(box=None, pad_edge=False, show_edge=False, header_style="italic")
    for column in ("gate", "median error", "duration", "records"):
        table.add_column(column, justify="left" if column == "gate" else "right")
    for n in natives:
        arity = f" ({n['qubits']}q)" if n["qubits"] else ""
        if n["virtual"]:
            table.add_row(f"{n['gate']}{arity}", "virtual", "-", "-")
            continue
        records = str(n["records"]) if n["records"] else "device-wide"
        total = sum(n["loci"].values())
        # a state on every locus already shows as the error column's "-", except disabled
        notes = [
            f"{count} {word}"
            for state, word in (
                ("ideal", "virtual"),
                ("uncalibrated", "no metric"),
                ("disabled", "disabled"),
            )
            if (count := n["loci"][state]) and (count < total or state == "disabled")
        ]
        if notes:
            records += f" ({', '.join(notes)})"
        table.add_row(
            f"{n['gate']}{arity}",
            fmt_error(n["median_avg_infidelity"]),
            _duration(n["median_duration_ns"]),
            records,
        )
    return table


def _coherence(data: dict[str, Any]) -> str:
    t1, t2 = data["median_t1_us"], data["median_t2_us"]
    if t1 is None and t2 is None:
        return "T1 and T2 unknown (gates get no relaxation)"
    return f"median T1 {fmt_time(t1)} us, median T2 {fmt_time(t2)} us"


def _readout(data: dict[str, Any]) -> str:
    if data["median_readout_error"] is None:
        return "unknown (no readout error applied)"
    return (
        f"median error {fmt_error(data['median_readout_error'])}"
        f" (P(1|0) {fmt_error(data['median_p1_given_0'])},"
        f" P(0|1) {fmt_error(data['median_p0_given_1'])})"
    )


def _parse_qubits(text: str, profile: Profile) -> list[int]:
    try:
        indices = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError(
            f"--qubits {text!r}: give qubit indices separated by commas, e.g. 0,1,2"
        ) from None
    n = profile.device.num_qubits
    bad = [q for q in indices if not 0 <= q < n]
    if bad or not indices:
        raise ValueError(f"--qubits {text!r}: {profile.id} has qubits 0 to {n - 1}")
    return indices


def _qubit_row(profile: Profile, index: int) -> dict[str, Any]:
    table = profile.table
    q = table.qubit(index)
    one = table.typical(1, (index,))
    record = next((r for r in profile.qubits if r.index == index), None)
    return {
        "qubit": index,
        "t1_us": None if q.t1_ns is None else q.t1_ns / 1000,
        "t2_us": None if q.t2_ns is None else q.t2_ns / 1000,
        "p1_given_0": None if q.readout is None else q.readout[0],
        "p0_given_1": None if q.readout is None else q.readout[1],
        "error_1q": one.avg_infidelity if isinstance(one, GateNoise) else None,
        "gate_1q": one.gate if isinstance(one, GateNoise) else None,
        "disabled": q.disabled,
        "label": record.label if record else None,
    }


def _print_qubits(rows: list[dict[str, Any]]) -> None:
    table = Table(box=None, pad_edge=False, header_style="bold")
    for column in ("qubit", "T1 (us)", "T2 (us)", "P(1|0)", "P(0|1)", "1q error", "state"):
        table.add_column(column, justify="right" if column != "state" else "left")
    for r in rows:
        one = f"{fmt_error(r['error_1q'])} ({r['gate_1q']})" if r["gate_1q"] else "-"
        state = "disabled" if r["disabled"] else (r["label"] or "")
        table.add_row(
            str(r["qubit"]),
            fmt_time(r["t1_us"]),
            fmt_time(r["t2_us"]),
            fmt_error(r["p1_given_0"]),
            fmt_error(r["p0_given_1"]),
            one,
            state,
        )
    out.print()
    out.print(table)


# pull ---------------------------------------------------------------------------------------


@app.command()
def pull(
    device: Annotated[str, typer.Argument(help="Device to pull, e.g. ibm_fez or ionq_forte-1.")],
    at: Annotated[
        str | None,
        typer.Option(
            "--at",
            help="Calibration in effect at this date or time (IBM public, IBM account, IonQ).",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option("--source", help="ibm, ibm-account or ionq; default from the name."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Save here instead of the vault.")
    ] = None,
) -> None:
    """Fetch a live calibration and save it as a profile (network)."""
    with _friendly():
        if output is not None:
            _check_writable(output)
        with err.status(f"Pulling {device}{f' at {at}' if at else ''}..."):
            pulled = catalog.pull_and_save(device, at=at, source=source, output=output)
        _print_card(card(pulled.profile), brief=True)
        saved = "saved" if pulled.written else "already saved"
        out.print(f"{saved}: {pulled.path}", markup=False, soft_wrap=True)


def _check_writable(output: Path) -> None:
    folder = output.parent
    if output.is_dir():
        raise ValueError(
            f"-o {output} is a directory; give a file name such as {output / 'x.json'}"
        )
    if not folder.is_dir():
        raise ValueError(f"-o {output}: the folder {folder} does not exist; create it first")
    if not os.access(folder, os.W_OK):
        raise ValueError(f"-o {output}: you cannot write to {folder}; choose another folder")


# diff ---------------------------------------------------------------------------------------


@app.command()
def diff(
    before: Annotated[str, typer.Argument(help="First profile (ref or file).")],
    after: Annotated[str, typer.Argument(help="Second profile (ref or file).")],
    top: Annotated[int, typer.Option("--top", min=0, help="Qubits and pairs to list.")] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Show how calibration changed between two profiles."""
    with _friendly():
        first, second = _load(before), _load(after)
        result = first.diff(second, top=top)
        if as_json:
            _echo_json(result.to_dict())
            return
        out.print(f"{_diff_title(first, second)}  ({describe_delta(result.time_delta)})")
        for warning in result.warnings:
            err.print(f"warning: {warning}", markup=False)
        if result.identical:
            out.print("No change: the two profiles have the same fingerprint.")
            return
        medians = Table(box=None, pad_edge=False, header_style="bold")
        for column in ("device median", "before", "after", "change"):
            medians.add_column(column, justify="left" if column == "device median" else "right")
        for c in result.medians:
            medians.add_row(
                METRICS[c.metric][0],
                fmt_metric(c.metric, c.before),
                fmt_metric(c.metric, c.after),
                _change(c),
            )
        out.print(medians)
        for title, changes in (("qubit", result.qubits), ("pair", result.pairs)):
            if not changes:
                continue
            table = Table(box=None, pad_edge=False, header_style="bold", title_justify="left")
            for column in (title, "metric", "before", "after", "change"):
                table.add_column(
                    column, justify="right" if column in ("before", "after", "change") else "left"
                )
            for c in changes:
                table.add_row(
                    c.where,
                    METRICS[c.metric][0],
                    fmt_metric(c.metric, c.before),
                    fmt_metric(c.metric, c.after),
                    _change(c),
                )
            out.print()
            out.print(f"largest changes by {title}")
            out.print(table)
        for label, items in (
            ("newly disabled", result.newly_disabled),
            ("re-enabled", result.reenabled),
        ):
            if items:
                out.print()
                out.print(label)
                out.print(_loci_table(items))
        for label, qubits in (
            ("qubits added", result.qubits_added),
            ("qubits removed", result.qubits_removed),
        ):
            if qubits:
                out.print(f"{label}: {_runs(qubits)}", markup=False)


def _runs(indices: Sequence[int]) -> str:
    """``0, 3 to 9, 12``: three or more consecutive indices as one run."""
    runs: list[list[int]] = []
    for i in sorted(indices):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return ", ".join(
        f"{run[0]} to {run[-1]}" if len(run) > 2 else ", ".join(map(str, run)) for run in runs
    )


def _diff_title(first: Profile, second: Profile) -> str:
    """``a@date -> b@date``; one device is named once, with times when the dates match."""
    if first.id != second.id:
        return f"[bold]{_ref_on_day(first)}[/bold] -> [bold]{_ref_on_day(second)}[/bold]"
    a, b = first.device.calibrated_at, second.device.calibrated_at
    if a and b and a.date() == b.date() and a != b:
        when = (_iso(a), _iso(b))
    else:
        when = tuple(t.date().isoformat() if t else "undated" for t in (a, b))
    return f"[bold]{first.id}[/bold] {when[0]} -> {when[1]}"


def _ref_on_day(profile: Profile) -> str:
    when = profile.device.calibrated_at
    return f"{profile.id}@{when.date().isoformat()}" if when else profile.id


def _loci_table(labels: tuple[str, ...]) -> Table:
    """One row per gate (or "qubit"), its loci folded at item boundaries."""
    by_gate: dict[str, list[str]] = {}
    for label in labels:
        gate, locus = label.split(" ", 1)
        by_gate.setdefault(gate, []).append(locus)
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column(no_wrap=True)
    table.add_column()
    for gate, loci in by_gate.items():
        table.add_row(f"  {gate}", ", ".join(loci))
    return table


def _change(change: Any) -> str:
    if change.before is None or change.after is None:
        # "worse" is undefined when one side has no value, so these stay uncolored
        return "-" if change.before == change.after else "new" if change.before is None else "gone"
    text = fmt_relative(change.relative)
    if change.relative == 0:
        return text
    return f"[red]{text}[/red]" if change.worse else f"[green]{text}[/green]"


# check --------------------------------------------------------------------------------------


@app.command()
def check(
    ref: Annotated[str, typer.Argument(help="Profile id, id@date, or a file path.")],
    framework: Annotated[
        str | None,
        typer.Option(
            "--framework", help="Comma-separated: qiskit,cirq,pennylane,stim (default all)."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Check that each framework export reproduces the reference model on small circuits."""
    from .check import FRAMEWORKS, NOTE

    with _friendly():
        names = list(FRAMEWORKS)
        if framework is not None:
            names = [n.strip().lower() for n in framework.split(",") if n.strip()]
            unknown = [n for n in names if n not in FRAMEWORKS]
            if not names or unknown:
                given = unknown[0] if unknown else framework
                raise ValueError(
                    f"--framework {given!r}: {did_you_mean(given, FRAMEWORKS)}give one or more"
                    f" of {','.join(FRAMEWORKS)}"
                )
        profile = _load(ref)
        parts = []
        for name in names:  # one at a time, so the spinner says which one is running
            with err.status(f"Checking the {name} export..."):
                parts.append(profile.check(frameworks=[name]))
        result = replace(
            parts[0],
            frameworks=tuple(f for part in parts for f in part.frameworks),
            skipped=tuple(s for part in parts for s in part.skipped),
        )
        if as_json:
            _echo_json(result.to_dict())
        else:
            chain = "-".join(str(result.layout[i]) for i in range(len(result.layout)))
            out.print(f"[bold]{profile.id}[/bold] {profile.short_fingerprint} on qubits {chain}")
            out.print(
                f"{len(result.circuits)} circuits: {', '.join(c.name for c in result.circuits)}",
                markup=False,
            )
            table = Table(box=None, pad_edge=False, header_style="bold")
            for column in ("framework", "result", "max TVD", "tolerance", "circuits", "method"):
                table.add_column(
                    column, justify="right" if column in ("max TVD", "tolerance") else "left"
                )
            for f in result.frameworks:
                verdict = "[green]pass[/green]" if f.passed else "[red]FAIL[/red]"
                reduced = {n.circuit for n in f.not_run if n.ran_without}
                ran = len({c.circuit for c in f.circuits} - reduced)
                counted = f"{ran} of {len(result.circuits)}"
                kinds = {c.sampled for c in f.circuits}
                method = {
                    frozenset({False}): "exact",
                    frozenset({True}): f"{result.shots} shots, 5 sigma",
                }.get(frozenset(kinds), f"exact + {result.shots} shots")
                table.add_row(
                    f.framework,
                    verdict,
                    f"{f.max_tvd:.1e}",
                    f"{f.worst.tolerance:.1e}",
                    f"{counted}, {len(reduced)} reduced" if reduced else counted,
                    method,
                )
            missing = [n for n, why in result.skipped if why.startswith("not installed")]
            for name, _ in result.skipped:
                table.add_row(name, "not installed" if name in missing else "skipped")
            out.print(table)
            for f in result.frameworks:
                for part in f.not_run:
                    out.print(f"{f.framework}: {part.describe()}", markup=False)
            for name, reason in result.skipped:
                if name not in missing:
                    out.print(f"{name} skipped: {reason}", markup=False, soft_wrap=True)
            if missing:
                command = install_hint(",".join(missing))
                out.print(f"To add the missing frameworks: {command}", markup=False, soft_wrap=True)
            out.print(NOTE, markup=False)
        if not result.frameworks:
            raise NoiseVaultError(f"no framework could run the check; install one: {_INSTALL_ALL}")
        if not result.passed:
            raise typer.Exit(1)


# cite, validate, doctor, schema ---------------------------------------------------------------


@app.command()
def cite(
    ref: Annotated[str, typer.Argument(help="Profile id, id@date, or a file path.")],
    bibtex: Annotated[bool, typer.Option("--bibtex", help="Print a BibTeX entry.")] = False,
) -> None:
    """Print a citation for a profile, with its fingerprint."""
    with _friendly():
        typer.echo(_load(ref).citation("bibtex" if bibtex else "text"))


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="Profile: .json or .json.gz, format 1.0 or 0.1.")],
    strict: Annotated[bool, typer.Option("--strict", help="Treat warnings as errors.")] = False,
) -> None:
    """Check a profile file against the format rules."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            profile = load_file(file)
        except FileNotFoundError:
            _fail(f"no file {file}; check the path")
        except ValidationError as exc:
            for line in _validation_lines(exc):
                err.print(f"error: {line}", markup=False)
            raise typer.Exit(1) from None
        except (*_UNREADABLE, NoiseVaultError) as exc:
            _fail(_unreadable(file, exc))
    notes = [str(w.message) for w in caught] + _soft_issues(profile)
    when = profile.device.calibrated_at.isoformat() if profile.device.calibrated_at else "undated"
    out.print(
        f"ok: {profile.id} calibrated {when}, {profile.device.num_qubits} qubits,"
        f" {len(profile.gates)} gates, {len(profile.calibrations)} calibration records,"
        f" {profile.short_fingerprint}",
        markup=False,
    )
    for note in notes:
        err.print(f"warning: {note}", markup=False)
    if strict and notes:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Show NoiseVault, Python and framework versions, and where profiles live."""
    table = Table("component", "version", box=None, pad_edge=False, header_style="bold")
    table.add_row("noisevault", __version__)
    table.add_row("python", platform.python_version())
    missing = []
    for package in _PACKAGES:
        try:
            table.add_row(package, version(package))
        except PackageNotFoundError:
            table.add_row(package, "not installed")
            missing.append(package)
    out.print(table)
    vault = catalog.vault_dir()
    with _friendly():
        count = len(catalog.vault_profiles()) if vault.is_dir() else 0
    out.print(f"vault: {vault} ({count} profiles)", markup=False, soft_wrap=True)
    out.print(f"bundled profiles: {len(catalog.bundled_profiles())}")
    extras = sorted({extra for p in missing if (extra := _PACKAGES[p])})
    if extras:
        command = install_hint(",".join(extras))
        out.print(f"To add the missing frameworks: {command}", markup=False, soft_wrap=True)
    loose = [p for p in missing if _PACKAGES[p] is None]
    if loose:
        out.print(f"To add {', '.join(loose)}: pip install {' '.join(loose)}", markup=False)


@app.command()
def schema() -> None:
    """Print the JSON Schema of profile format 1.0."""
    _echo_json(json_schema())


# shared helpers -----------------------------------------------------------------------------


def _load(ref: str) -> Profile:
    target = parse_ref(ref)
    if isinstance(target, Path):
        if not target.exists():
            raise FileNotFoundError(errno.ENOENT, "no such file", str(target))
        if target.is_dir():
            text = ref.strip()
            if text == target.name and "@" not in text:  # a bare id that names a folder here
                return catalog.resolve(Ref(text.lower())).load()
            raise ValueError(
                f"{ref} is a folder; give a profile file (.json or .json.gz) or a profile id"
                " such as ibm_fez"
            )
        try:
            return load_file(target)
        except _UNREADABLE as exc:
            raise ValueError(_unreadable(target, exc)) from None
    return catalog.load(ref)


# What reading a profile file can raise besides FileNotFoundError; ValidationError is a ValueError.
_UNREADABLE = (ValueError, OSError, EOFError, zlib.error)


def _unreadable(path: Path, exc: BaseException) -> str:
    """One line naming the file, what is wrong with it, and what to do."""
    if isinstance(exc, ValidationError):
        problems = _count(exc.error_count(), "problem")
        return f"{path} is not a valid profile ({problems}); run `nv validate {path}` to list them"
    if isinstance(exc, json.JSONDecodeError):
        where = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
        return f"{path} is not JSON ({where}); give a profile file (.json or .json.gz)"
    if isinstance(exc, UnicodeDecodeError):
        return f"{path} is not JSON (not UTF-8 text); give a profile file (.json or .json.gz)"
    if isinstance(exc, EOFError | zlib.error | gzip.BadGzipFile):
        return f"cannot read {path}: it is a damaged gzip file ({exc}); copy or pull it again"
    if isinstance(exc, IsADirectoryError):
        return f"cannot read {path}: it is a folder; give a profile file (.json or .json.gz)"
    if isinstance(exc, OSError):
        return f"cannot read {path}: {exc.strerror or exc}; check the file and its permissions"
    return f"{path}: {exc}"


@contextmanager
def _friendly() -> Iterator[None]:
    """Turn expected failures into ``error:`` and ``hint:`` lines, and warnings into lines."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        except typer.Exit:
            raise
        except _EXPECTED as exc:
            _report_warnings(caught)
            _error(exc)
        _report_warnings(caught)


def _report_warnings(caught: list[warnings.WarningMessage]) -> None:
    for message in dict.fromkeys(str(w.message) for w in caught):
        err.print(f"warning: {message}", markup=False)
    caught.clear()


def _error(exc: BaseException) -> None:
    if isinstance(exc, ValidationError):
        message = f"not a valid profile ({_count(exc.error_count(), 'problem')})"
    elif isinstance(exc, FileNotFoundError) and exc.filename:
        message = f"no file {exc.filename}"
    else:
        message = _cli_terms(str(exc) or type(exc).__name__)
    err.print(f"error: {message}", markup=False)
    hint = next((h for kind, h in _HINTS if isinstance(exc, kind)), None)
    command = re.search(r"`[^`]+`", hint or "")
    if hint and "did you mean" not in message and not (command and command[0] in message):
        err.print(f"hint: {hint}", markup=False)
    raise typer.Exit(1) from None


# Library messages name Python arguments; at the command line the same choice is a flag.
_CLI_TERMS = (
    (re.compile(r"""source=(['"])([\w-]+)\1 or (['"])([\w-]+)\3"""), r"--source \2 or \4"),
    (re.compile(r"""source=(['"])([\w-]+)\1"""), r"--source \2"),
    (re.compile(r"""nv\.load\((['"])([\w.@:-]+)\1\)"""), r"`nv show \2`"),
    (re.compile(r"\bat=(?= )"), "--at"),
    (re.compile(r"\bat=(?=['\"])"), "--at "),
    (re.compile(r"pass expect='nv:\.\.\.' or load a file:"), "give one of their files instead:"),
)


def _cli_terms(message: str) -> str:
    for pattern, replacement in _CLI_TERMS:
        message = pattern.sub(replacement, message)
    return message


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _fail(message: str) -> None:
    err.print(f"error: {message}", markup=False)
    raise typer.Exit(1)


def _echo_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


def _iso(when: datetime | None) -> str | None:
    return when.isoformat().replace("+00:00", "Z") if when else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _duration(ns: float | None) -> str:
    if ns is None:
        return "-"
    if ns >= 1e6:
        return f"{ns / 1e6:.3g} ms"
    return f"{ns / 1e3:.3g} us" if ns >= 1e3 else f"{ns:.3g} ns"


def _validation_lines(exc: ValidationError) -> list[str]:
    lines = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        for part in message.split("\n"):
            lines.append(f"{where}: {part}" if where else part)
    return lines


def _soft_issues(profile: Profile) -> list[str]:
    """Legal but noteworthy values: they change what a conversion produces."""
    notes = []
    for i in range(profile.device.num_qubits):
        q = profile.table.qubit(i)
        if q.t2_clamped:
            notes.append(f"qubit {i}: T2 exceeds 2*T1; conversions clamp it to 2*T1")
    for record in profile.calibrations:
        if record.scope == "cycle":
            notes.append(f"{record.gate} {list(record.qubits)}: error is per cycle, not per gate")
    return notes
