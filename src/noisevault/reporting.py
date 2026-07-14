from __future__ import annotations

import json
from pathlib import Path


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fmt(value, digits: int = 5) -> str:
    return "not run" if value is None else f"{float(value):.{digits}f}"


def render_pilot_report(data: dict) -> str:
    summary = data["summary"]
    a = data["experiment_a"]
    b = data["experiment_b"]
    c = data["experiment_c"]
    d = data["experiment_d"]
    lines = [
        "# NoiseVault Pilot Results",
        "",
        f"**Overall status:** `{summary['status']}`",
        "",
        f"Run timestamp: `{summary['run_timestamp']}`  ",
        f"Execution mode: `{summary['mode']}` / `{summary['profile']}`  ",
        f"Primary snapshot: `{summary['primary_snapshot']}`  ",
        f"Data classification: **{summary['data_classification']}**",
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
        f"- Maximum pairwise cross-framework TVD: **{_fmt(b['max_pairwise_tvd'])}** (threshold {_fmt(b['threshold'])}).",
        f"- Native Aer reconstruction maximum TVD: **{_fmt(a.get('max_tvd'))}** (target {_fmt(a.get('threshold'))}).",
        f"- Drift demonstration simulation TVD: **{_fmt(d.get('simulation_tvd'))}**.",
        "",
        "## Experiment A — Qiskit native-model reconstruction",
        "",
        f"Status: `{a['status']}`. Backend: `{a.get('backend_name', 'n/a')}`. Maximum TVD: **{_fmt(a.get('max_tvd'))}**.",
        "",
        *[f"- {note}" for note in a.get("notes", [])],
        "",
        "## Experiment B — Cross-framework agreement",
        "",
        f"Status: `{b['status']}`. Pairwise TVD maximum: **{_fmt(b['max_pairwise_tvd'])}**; mean: **{_fmt(b['mean_pairwise_tvd'])}**.",
        "",
        "The comparison uses the same abstract circuit, the same calibrated parameters, the same canonical Kraus channels, exact density-matrix simulation, and one common asymmetric readout-confusion post-processor. This isolates framework conversion rather than transpiler policy.",
        "",
        "![Cross-framework agreement](figures/cross_framework_tvd.png)",
        "",
        "## Experiment C — Noise realism sanity check",
        "",
        f"Status: `{c['status']}`. The plotted quantity is classical fidelity to the ideal GHZ distribution as qubit count increases.",
        "",
        "![GHZ fidelity](figures/ghz_fidelity.png)",
        "",
        "## Experiment D — Versioned drift demonstration",
        "",
        f"Status: `{d['status']}`. Older snapshot: `{d.get('older_snapshot')}`. Newer snapshot: `{d.get('newer_snapshot')}`.",
        "",
        f"Pair classification: **{d.get('pair_classification', 'not available')}**. Providers: `{', '.join(d.get('data_provider_pair', []))}`. Source timestamps: `{d.get('older_source_timestamp')}` → `{d.get('newer_source_timestamp')}`.",
        "",
        f"The representative circuit distribution changed by TVD **{_fmt(d.get('simulation_tvd'))}** between the two distinct snapshot payloads. Only a pair containing live IBM provenance is treated as empirical current-device evidence.",
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
            "Calibration-derived Markovian channels do not capture crosstalk, leakage, non-Markovian effects, coherent calibration errors, or workload-dependent drift. Cross-framework agreement proves converter consistency, not hardware accuracy. A live hardware point is intentionally treated as optional and must never be fabricated when credentials, queue access, or free allocation are unavailable.",
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
