from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from .benchmarks import CircuitSpec, mirror, pilot_benchmarks
from .metrics import hellinger_fidelity, total_variation_distance
from .native_roundtrip import compare_native_aer
from .providers.ibm import CredentialsUnavailable, harvest_ibm
from .providers.qiskit_fake import import_fake_backend, list_fake_backends
from .reporting import render_pilot_report, write_json
from .runners.cirq_runner import run_cirq
from .runners.pennylane_runner import run_pennylane
from .runners.qiskit_runner import run_qiskit
from .runners.reference import run_reference
from .schema import validate_snapshot
from .select import best_connected_subset
from .snapshot_io import load_snapshot

Runner = Callable[[object, CircuitSpec, list[int]], object]


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _timestamp(snapshot) -> datetime:
    return _parse_timestamp(snapshot.captured_at)


def _effective_timestamp(snapshot) -> datetime:
    source_timestamp = snapshot.provenance.source_timestamp
    if source_timestamp:
        try:
            return _parse_timestamp(source_timestamp)
        except ValueError:
            pass
    return _timestamp(snapshot)


def _discover_snapshots(root: Path) -> list[tuple[Path, object]]:
    discovered = []
    for path in sorted((root / "snapshots").rglob("*.json")):
        try:
            discovered.append((path, load_snapshot(path)))
        except Exception:
            continue
    return discovered


def _framework_availability() -> dict[str, bool]:
    modules = {"qiskit": "qiskit_aer", "cirq": "cirq", "pennylane": "pennylane"}
    return {name: importlib.util.find_spec(module) is not None for name, module in modules.items()}


def _choose_primary(discovered):
    rank = {"ibm": 0, "qiskit_fake": 1, "ibm_csv": 2, "demo": 3}
    return min(
        discovered,
        key=lambda item: (rank.get(item[1].provider, 9), -_timestamp(item[1]).timestamp()),
    )


def _drift_provider_score(first: str, second: str) -> int:
    providers = {first, second}
    if first == second == "ibm":
        return 0
    if "ibm" in providers and providers <= {"ibm", "qiskit_fake", "ibm_csv"}:
        return 1
    if first == second == "qiskit_fake":
        return 2
    if providers <= {"ibm_csv", "qiskit_fake"}:
        return 3
    if first == second == "demo":
        return 4
    return 5


def _choose_drift_pair(discovered):
    groups = defaultdict(list)
    for path, snapshot in discovered:
        groups[snapshot.backend_name].append((path, snapshot))

    candidates = []
    for backend, items in groups.items():
        for first, second in combinations(items, 2):
            first_snapshot = first[1]
            second_snapshot = second[1]
            # Re-importing one frozen fake backend creates a new archive timestamp but not
            # a new calibration. Such duplicates are not evidence of temporal drift.
            if first_snapshot.provenance.raw_hash == second_snapshot.provenance.raw_hash:
                continue
            ordered = sorted((first, second), key=lambda item: _effective_timestamp(item[1]))
            older, newer = ordered
            span = (_effective_timestamp(newer[1]) - _effective_timestamp(older[1])).total_seconds()
            if span <= 0:
                continue
            provider_score = _drift_provider_score(older[1].provider, newer[1].provider)
            candidates.append((provider_score, -span, backend, older, newer))
    if not candidates:
        raise RuntimeError("No backend has two distinct, genuinely dated snapshot payloads.")
    _, _, _, older, newer = min(candidates)
    return older, newer


def _load_config(root: Path) -> dict:
    return json.loads((root / "pilot_config.json").read_text(encoding="utf-8"))


def _try_fake_backend(root: Path, errors: list[dict]) -> object | None:
    try:
        names = list_fake_backends()
        priorities = ["FakeManilaV2", "FakeJakartaV2", "FakeLimaV2", "FakeAthensV2", "FakeSherbrooke"]
        name = next((candidate for candidate in priorities if candidate in names), names[0])
        import_fake_backend(name, root / "snapshots")
        import qiskit_ibm_runtime.fake_provider as fake_provider

        return getattr(fake_provider, name)()
    except Exception as exc:
        errors.append({"stage": "fake_backend_import", "error": f"{type(exc).__name__}: {exc}"})
        return None


def _try_live_harvest(root: Path, config: dict, errors: list[dict]) -> None:
    try:
        harvest_ibm(
            root / "snapshots",
            max_backends=int(config["live"]["max_backends"]),
        )
    except CredentialsUnavailable as exc:
        errors.append({"stage": "live_harvest", "error": str(exc)})
    except Exception as exc:
        errors.append({"stage": "live_harvest", "error": f"{type(exc).__name__}: {exc}"})


def _placeholder_figure(path: Path, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cross_framework(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    pairs = ["qiskit-cirq", "qiskit-pennylane", "cirq-pennylane"]
    if not rows:
        _placeholder_figure(path, "Cross-framework TVD", "No complete three-framework circuits were available.")
        return
    matrix = np.array([[row["pairwise_tvd"].get(pair, np.nan) for pair in pairs] for row in rows])
    height = max(4.5, min(14, 0.27 * len(rows) + 2.5))
    fig, ax = plt.subplots(figsize=(9, height))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(pairs)), pairs, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), [row["circuit"] for row in rows], fontsize=7)
    ax.set_title("Pairwise total variation distance")
    fig.colorbar(image, ax=ax, label="TVD")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ghz(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        _placeholder_figure(path, "GHZ fidelity", "No GHZ results were available.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    by_framework = defaultdict(list)
    for row in rows:
        by_framework[row["framework"]].append((row["num_qubits"], row["fidelity"]))
    for framework, values in sorted(by_framework.items()):
        values.sort()
        ax.plot([x for x, _ in values], [y for _, y in values], marker="o", label=framework)
    sizes = sorted({row["num_qubits"] for row in rows})
    ax.plot(sizes, [1.0] * len(sizes), linestyle="--", label="ideal")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Fidelity to ideal GHZ distribution")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _mean_two_qubit_errors(snapshot, common: list[int]) -> list[float]:
    values = []
    for a, b in zip(common, common[1:], strict=False):
        errors = [
            gate.error
            for gate in snapshot.gates
            if gate.error is not None and len(gate.qubits) == 2 and set(gate.qubits) == {a, b}
        ]
        values.append(float(min(errors)) if errors else np.nan)
    return values


def _plot_drift(older, newer, common: list[int], path: Path) -> None:
    import matplotlib.pyplot as plt

    old_q = {q.index: q for q in older.qubits}
    new_q = {q.index: q for q in newer.qubits}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(common, [old_q[q].t1_us for q in common], marker="o", label="older")
    axes[0].plot(common, [new_q[q].t1_us for q in common], marker="o", label="newer")
    axes[0].set_title("T1")
    axes[0].set_xlabel("Physical qubit")
    axes[0].set_ylabel("µs")
    axes[0].legend()

    axes[1].plot(common, [old_q[q].t2_us for q in common], marker="o", label="older")
    axes[1].plot(common, [new_q[q].t2_us for q in common], marker="o", label="newer")
    axes[1].set_title("T2")
    axes[1].set_xlabel("Physical qubit")
    axes[1].set_ylabel("µs")
    axes[1].legend()

    edges = [f"{a}-{b}" for a, b in zip(common, common[1:], strict=False)]
    x = np.arange(len(edges))
    width = 0.36
    axes[2].bar(x - width / 2, _mean_two_qubit_errors(older, common), width, label="older")
    axes[2].bar(x + width / 2, _mean_two_qubit_errors(newer, common), width, label="newer")
    axes[2].set_xticks(x, edges)
    axes[2].set_title("Best two-qubit gate error")
    axes[2].set_xlabel("Edge")
    axes[2].legend()
    fig.suptitle(f"Calibration comparison: {older.backend_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_pilot(root: str | Path, *, mode: str = "auto", profile: str = "full") -> dict:
    root = Path(root).resolve()
    config = _load_config(root)
    errors: list[dict] = []

    availability = _framework_availability()
    fake_backend = None
    if mode in {"auto", "live"}:
        _try_live_harvest(root, config, errors)
    if mode in {"auto", "offline", "live"} and availability["qiskit"]:
        fake_backend = _try_fake_backend(root, errors)

    discovered = _discover_snapshots(root)
    if not discovered:
        raise RuntimeError("No snapshots are available, including the committed offline fixtures.")
    primary_path, primary = _choose_primary(discovered)
    validation = validate_snapshot(primary)
    if not validation.valid:
        raise RuntimeError("Primary snapshot is invalid.")

    if profile == "smoke":
        subset_sizes = [2, 3]
        mirror_depths = [2]
        rb_depths = [2, 4]
    else:
        subset_sizes = [size for size in config["subset_sizes"] if size <= primary.num_qubits]
        mirror_depths = config["mirror_depths"]
        rb_depths = config["rb_depths"]
    circuits = pilot_benchmarks(subset_sizes, mirror_depths, rb_depths, config["seed"])

    runners: dict[str, Runner] = {
        "qiskit": run_qiskit,
        "cirq": run_cirq,
        "pennylane": run_pennylane,
    }
    selected_by_size = {
        size: best_connected_subset(primary, size) for size in sorted({c.num_qubits for c in circuits})
    }
    raw_results: dict[str, dict[str, object]] = defaultdict(dict)

    for circuit in circuits:
        physical = selected_by_size[circuit.num_qubits]
        raw_results[circuit.name]["reference"] = run_reference(primary, circuit, physical)
        for framework, runner in runners.items():
            if not availability[framework]:
                continue
            try:
                raw_results[circuit.name][framework] = runner(primary, circuit, physical)
            except Exception as exc:
                errors.append(
                    {
                        "stage": f"simulate:{framework}:{circuit.name}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    requested_frameworks = [name for name in runners if availability[name]]
    pairwise_rows = []
    all_pair_values = []
    for circuit in circuits:
        framework_results = raw_results[circuit.name]
        if not all(name in framework_results for name in ("qiskit", "cirq", "pennylane")):
            continue
        pairwise = {}
        for a, b in (("qiskit", "cirq"), ("qiskit", "pennylane"), ("cirq", "pennylane")):
            value = total_variation_distance(
                framework_results[a].probabilities, framework_results[b].probabilities
            )
            pairwise[f"{a}-{b}"] = value
            all_pair_values.append(value)
        pairwise_rows.append({"circuit": circuit.name, "pairwise_tvd": pairwise})

    threshold_b = float(config["thresholds"]["cross_framework_tvd_density_matrix"])
    max_pairwise = max(all_pair_values) if all_pair_values else None
    mean_pairwise = float(np.mean(all_pair_values)) if all_pair_values else None
    experiment_b = {
        "status": "complete" if pairwise_rows else "incomplete",
        "complete_circuit_count": len(pairwise_rows),
        "max_pairwise_tvd": max_pairwise,
        "mean_pairwise_tvd": mean_pairwise,
        "threshold": threshold_b,
        "passed": max_pairwise is not None and max_pairwise <= threshold_b,
        "rows": pairwise_rows,
    }

    ghz_rows = []
    for circuit in circuits:
        if circuit.family != "ghz":
            continue
        ideal = np.zeros(2**circuit.num_qubits)
        ideal[0] = ideal[-1] = 0.5
        for framework, result in raw_results[circuit.name].items():
            if framework == "reference" or framework in runners:
                ghz_rows.append(
                    {
                        "circuit": circuit.name,
                        "framework": framework,
                        "num_qubits": circuit.num_qubits,
                        "fidelity": hellinger_fidelity(result.probabilities, ideal),
                    }
                )
    experiment_c = {"status": "complete" if ghz_rows else "incomplete", "rows": ghz_rows}

    try:
        (older_path, older), (newer_path, newer) = _choose_drift_pair(discovered)
        common_indices = sorted(
            set(q.index for q in older.qubits if q.operational)
            & set(q.index for q in newer.qubits if q.operational)
        )
        drift_size = min(3, len(common_indices))
        common = common_indices[: max(drift_size, 1)]
        drift_rows = []
        if drift_size >= 2:
            drift_circuit = mirror(drift_size, 4 if profile == "smoke" else 8, config["seed"])
            for framework, runner in {"reference": run_reference, **runners}.items():
                if framework != "reference" and not availability[framework]:
                    continue
                try:
                    old_result = runner(older, drift_circuit, common[:drift_size])
                    new_result = runner(newer, drift_circuit, common[:drift_size])
                    drift_rows.append(
                        {
                            "framework": framework,
                            "tvd": total_variation_distance(
                                old_result.probabilities, new_result.probabilities
                            ),
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "stage": f"drift:{framework}",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        drift_tvd = max((row["tvd"] for row in drift_rows), default=None)
        experiment_d = {
            "status": "complete" if drift_rows else "incomplete",
            "older_snapshot": str(older_path.relative_to(root)),
            "newer_snapshot": str(newer_path.relative_to(root)),
            "backend_name": older.backend_name,
            "simulation_tvd": drift_tvd,
            "rows": drift_rows,
            "data_provider_pair": [older.provider, newer.provider],
            "pair_classification": (
                "empirical calibration comparison"
                if "ibm" in {older.provider, newer.provider}
                else "packaged calibration comparison"
                if {older.provider, newer.provider} <= {"qiskit_fake", "ibm_csv"}
                else "deterministic archive-semantics demonstration"
            ),
            "older_source_timestamp": older.provenance.source_timestamp,
            "newer_source_timestamp": newer.provenance.source_timestamp,
            "older_raw_hash": older.provenance.raw_hash,
            "newer_raw_hash": newer.provenance.raw_hash,
            "common_qubits": common,
        }
        _plot_drift(older, newer, common, root / "experiments/figures/drift_comparison.png")
    except Exception as exc:
        errors.append({"stage": "drift_pair", "error": f"{type(exc).__name__}: {exc}"})
        experiment_d = {
            "status": "incomplete",
            "older_snapshot": None,
            "newer_snapshot": None,
            "simulation_tvd": None,
            "rows": [],
        }
        _placeholder_figure(
            root / "experiments/figures/drift_comparison.png",
            "Calibration drift",
            "No valid dated snapshot pair was available.",
        )

    threshold_a = float(config["thresholds"]["roundtrip_tvd_density_matrix"])
    if fake_backend is not None:
        fake_name = getattr(fake_backend, "name", type(fake_backend).__name__)
        fake_name = fake_name() if callable(fake_name) else fake_name
        fake_candidates = [item for item in discovered if item[1].backend_name == fake_name]
        if fake_candidates:
            _, fake_snapshot = max(fake_candidates, key=lambda item: _timestamp(item[1]))
            native_result = compare_native_aer(fake_backend, fake_snapshot, num_qubits=2)
            experiment_a = {
                "status": native_result.status,
                "backend_name": native_result.backend_name,
                "max_tvd": native_result.max_tvd,
                "threshold": threshold_a,
                "passed": native_result.max_tvd is not None and native_result.max_tvd <= threshold_a,
                "circuits": native_result.circuits,
                "notes": native_result.notes,
            }
        else:
            experiment_a = {
                "status": "incomplete",
                "backend_name": str(fake_name),
                "max_tvd": None,
                "threshold": threshold_a,
                "passed": False,
                "notes": ["Imported fake backend snapshot could not be rediscovered."],
            }
    else:
        experiment_a = {
            "status": "skipped",
            "backend_name": None,
            "max_tvd": None,
            "threshold": threshold_a,
            "passed": False,
            "notes": ["No Qiskit fake backend object was available for a native Aer comparison."],
        }

    _plot_cross_framework(pairwise_rows, root / "experiments/figures/cross_framework_tvd.png")
    _plot_ghz(ghz_rows, root / "experiments/figures/ghz_fidelity.png")

    providers = {snapshot.provider for _, snapshot in discovered}
    if "ibm" in providers:
        classification = "Contains live IBM calibration snapshots; inspect provenance per file."
    elif "qiskit_fake" in providers:
        classification = "Packaged Qiskit fake-backend calibrations plus deterministic fixtures; no live hardware claim."
    else:
        classification = "Deterministic synthetic fixtures only; pipeline verification, not empirical device evidence."

    complete_frameworks = [
        name
        for name in runners
        if any(name in result_map for result_map in raw_results.values())
        and not any(item["stage"].startswith(f"simulate:{name}:") for item in errors)
    ]
    all_three = set(complete_frameworks) == set(runners)
    if all_three and experiment_b["passed"] and experiment_a["passed"]:
        if "ibm" in providers:
            status = "PILOT_COMPLETE_WITH_LIVE_CALIBRATION"
        elif "qiskit_fake" in providers or "ibm_csv" in providers:
            status = "OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS"
        else:
            status = "OFFLINE_PIPELINE_VERIFIED"
    elif all_three and max_pairwise is not None and not experiment_b["passed"]:
        status = "PILOT_EXECUTED_THRESHOLD_MISSED"
    elif all_three:
        status = "PILOT_EXECUTED_WITH_OPEN_VALIDATION_GAPS"
    else:
        status = "PARTIAL"

    summary = {
        "status": status,
        "run_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mode": mode,
        "profile": profile,
        "primary_snapshot": str(primary_path.relative_to(root)),
        "snapshot_count": len(discovered),
        "backend_count": len({snapshot.backend_name for _, snapshot in discovered}),
        "frameworks_requested": requested_frameworks,
        "frameworks_completed": complete_frameworks,
        "data_classification": classification,
        "integrity_statement": (
            "No result is labeled as live hardware evidence unless its snapshot provenance identifies the IBM Quantum Platform. "
            "Committed demo snapshots are synthetic fixtures and are retained only so CI and autonomous agents can verify the entire pipeline without credentials."
        ),
    }

    simulations = []
    circuit_lookup = {circuit.name: circuit for circuit in circuits}
    for circuit_name, framework_results in raw_results.items():
        circuit = circuit_lookup[circuit_name]
        for _framework, result in framework_results.items():
            record = result.as_record()
            record.update(
                {
                    "family": circuit.family,
                    "num_qubits": circuit.num_qubits,
                    "physical_qubits": selected_by_size[circuit.num_qubits],
                }
            )
            simulations.append(record)

    data = {
        "summary": summary,
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "experiment_c": experiment_c,
        "experiment_d": experiment_d,
        "simulations": simulations,
        "validation": {
            "warnings": [issue.__dict__ for issue in validation.warnings],
            "errors": [issue.__dict__ for issue in validation.errors],
        },
        "errors": errors,
    }
    write_json(data, root / "experiments/results/pilot_results.json")
    (root / "experiments/PILOT_RESULTS.md").write_text(render_pilot_report(data), encoding="utf-8")
    return data
