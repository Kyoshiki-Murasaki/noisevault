from __future__ import annotations

import json
from pathlib import Path


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fmt(value, digits: int = 5) -> str:
    if value is None:
        return "not available"
    numeric = float(value)
    if numeric != 0.0 and abs(numeric) < 10 ** (-digits):
        return f"{numeric:.6e}"
    return f"{numeric:.{digits}f}"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def render_pilot_report(data: dict) -> str:
    summary = data["summary"]
    a = data["experiment_a"]
    b = data["experiment_b"]
    c = data["experiment_c"]
    d = data["experiment_d"]
    auth = summary.get("live_ibm_authentication_available")
    auth_label = "available" if auth is True else "unavailable" if auth is False else "not attempted"
    ghz_reference = {
        row["num_qubits"]: row["fidelity"]
        for row in c.get("rows", [])
        if row["framework"] == "reference"
    }
    lines = [
        "# NoiseVault Pilot Results",
        "",
        f"**Overall status:** `{summary['status']}`",
        "",
        f"Run timestamp: `{summary['run_timestamp']}`  ",
        f"Execution mode: `{summary['mode']}` / `{summary['profile']}`  ",
        f"Primary snapshot: `{summary['primary_snapshot']}`  ",
        f"Data classification: **{summary['data_classification']}**",
        f"Live IBM authentication: **{auth_label}**  ",
        f"Hardware job ran: **{'yes' if summary.get('hardware_job_ran') else 'no'}**",
        "",
        "## Scope and integrity statement",
        "",
        summary["integrity_statement"],
        "",
        "## Headline numbers",
        "",
        f"- Snapshots discovered: **{summary['snapshot_count']}** across **{summary['backend_count']}** backend names.",
        f"- Frameworks completed: **{', '.join(summary['frameworks_completed']) or 'none'}**.",
        f"- Benchmark circuits with all requested framework results: **{b['complete_circuit_count']}**.",
        f"- Maximum pairwise cross-framework TVD: **{_fmt(b['max_pairwise_tvd'])}** (threshold ≤ {_fmt(b['threshold'])}; **{_verdict(b['passed'])}**).",
        f"- Native Aer reconstruction maximum TVD: **{_fmt(a.get('max_tvd'))}** (target < {_fmt(a.get('threshold'))}; **{_verdict(a['passed'])}**).",
        f"- Drift demonstration simulation TVD: **{_fmt(d.get('simulation_tvd'))}**.",
        "",
        "## Snapshot inventory",
        "",
        "| Provider | Backend | Calibration/source timestamp | Qubits | Raw hash | Path |",
        "|---|---|---:|---:|---|---|",
        *[
            f"| `{item['provider']}` | `{item['backend_name']}` | `{item['source_timestamp'] or 'unavailable'}` | {item['num_qubits']} | `{item['raw_hash']}` | `{item['path']}` |"
            for item in data.get("snapshots", [])
        ],
        "",
        "## Experiment A — Qiskit native-model reconstruction",
        "",
        f"Status: `{a['status']}` / **{_verdict(a['passed'])}**. Backend: `{a.get('backend_name', 'unavailable')}`. Data provider: `{a.get('data_provider', 'unavailable')}`. Maximum TVD: **{_fmt(a.get('max_tvd'))}**, with a strict target below **{_fmt(a.get('threshold'))}**.",
        f"The reconstructed per-gate average infidelities match Aer's model target to maximum absolute delta **{_fmt(a.get('max_calibration_error_delta'))}**. When the archived gate error lies below the relaxation floor, both models retain the physical relaxation floor rather than forcing an impossible target.",
        "",
        *[f"- {note}" for note in a.get("notes", [])],
        "",
        "## Experiment B — Cross-framework agreement",
        "",
        f"Status: `{b['status']}` / **{_verdict(b['passed'])}**. Complete circuits: **{b['complete_circuit_count']} / {b.get('expected_circuit_count', b['complete_circuit_count'])}**. Pairwise TVD maximum: **{_fmt(b['max_pairwise_tvd'])}**; mean: **{_fmt(b['mean_pairwise_tvd'])}**; threshold: **≤ {_fmt(b['threshold'])}**.",
        "",
        "The comparison executes the exported Qiskit Aer, Cirq, and PennyLane converter objects in their independent density-matrix engines. It uses one common asymmetric readout-confusion post-processor so the measured difference isolates quantum-channel conversion rather than framework-specific sampling or readout APIs.",
        "",
        "![Cross-framework agreement](figures/cross_framework_tvd.png)",
        "",
        "## Experiment C — Noise realism sanity check",
        "",
        f"Status: `{c['status']}`. The plotted quantity is classical fidelity to the ideal GHZ distribution on nested physical path `{c.get('nested_physical_path')}`. Maximum GHZ cross-framework TVD: **{_fmt(c.get('max_cross_framework_tvd'))}**.",
        "",
        *[
            f"- {size} qubits: reference fidelity **{_fmt(fidelity)}**."
            for size, fidelity in sorted(ghz_reference.items())
        ],
        "",
        "![GHZ fidelity](figures/ghz_fidelity.png)",
        "",
        "## Experiment D — Versioned drift demonstration",
        "",
        f"Status: `{d['status']}`. Older snapshot: `{d.get('older_snapshot')}`. Newer snapshot: `{d.get('newer_snapshot')}`.",
        "",
        f"Pair classification: **{d.get('pair_classification', 'not available')}**. Providers: `{', '.join(d.get('data_provider_pair', []))}`. Source timestamps: `{d.get('older_source_timestamp')}` → `{d.get('newer_source_timestamp')}`. Empirical calibration evidence: **{'yes' if d.get('empirical_calibration_evidence') else 'no'}**.",
        "",
        f"The representative circuit distribution changed by maximum TVD **{_fmt(d.get('simulation_tvd'))}** between the two distinct snapshot payloads. Older and newer probability vectors are saved for every framework so this value can be recomputed independently. Deterministic fixtures demonstrate archive semantics only and are not device-drift evidence.",
        "",
        "![Calibration drift](figures/drift_comparison.png)",
        "",
        "## Validation and failures",
        "",
    ]
    if data.get("errors"):
        lines.extend(f"- `{item['stage']}`: {item['error']}" for item in data["errors"])
    else:
        lines.append("No execution errors were recorded.")
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "Calibration-derived Markovian channels do not capture crosstalk, leakage, non-Markovian effects, coherent calibration errors, or workload-dependent drift. Cross-framework agreement proves converter consistency, not hardware accuracy. No QPU job ran in this pilot. A live hardware point is optional and must never be inferred from packaged calibrations or deterministic fixtures.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m venv .venv",
            "source .venv/bin/activate",
            "python -m pip install -e '.[pilot]'",
            "python scripts/run_pilot.py --mode auto --profile full",
            "```",
            "",
            "Machine-readable results, including per-framework probability vectors, are in `results/pilot_results.json`.",
        ]
    )
    return "\n".join(lines) + "\n"
