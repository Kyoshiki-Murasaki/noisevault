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
from .select import best_common_connected_path, best_connected_subset
from .snapshot_io import load_snapshot

Runner = Callable[[object, CircuitSpec, list[int]], object]


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _timestamp(snapshot) -> datetime:
    return _parse_timestamp(snapshot.captured_at)


def _calibration_timestamp(snapshot) -> datetime:
    source_timestamp = snapshot.provenance.source_timestamp
    if not source_timestamp:
        raise ValueError(
            f"{snapshot.provider}/{snapshot.backend_name} has no calibration/source timestamp."
        )
    return _parse_timestamp(source_timestamp)


def _discover_snapshots(root: Path) -> list[tuple[Path, object]]:
    discovered = []
    failures = []
    for path in sorted((root / "snapshots").rglob("*.json")):
        try:
            discovered.append((path, load_snapshot(path)))
        except Exception as exc:
            failures.append(f"{path.relative_to(root)}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("Invalid snapshot archive entries: " + " | ".join(failures))
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


def _drift_provider_score(first: str, second: str) -> int | None:
    providers = {first, second}
    if first == second == "ibm":
        return 0
    if providers == {"ibm", "qiskit_fake"}:
        return 1
    if providers == {"ibm", "ibm_csv"}:
        return 2
    if first == second == "qiskit_fake":
        return 3
    if providers <= {"ibm_csv", "qiskit_fake"}:
        return 4
    if first == second == "demo":
        return 5
    return None


def _drift_pair_classification(first: str, second: str) -> tuple[str, bool]:
    providers = {first, second}
    if first == second == "ibm":
        return "empirical live-calibration time series", True
    if providers == {"ibm", "qiskit_fake"}:
        return "live-versus-packaged calibration comparison", True
    if providers == {"ibm", "ibm_csv"}:
        return "live-versus-imported calibration comparison", True
    if first == second == "qiskit_fake":
        return "packaged calibration time series", False
    if providers <= {"ibm_csv", "qiskit_fake"}:
        return "packaged/imported calibration comparison", False
    if first == second == "demo":
        return "deterministic synthetic archive-semantics demonstration", False
    raise ValueError(f"Unsupported drift provenance pair: {first}, {second}")


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
            provider_score = _drift_provider_score(
                first_snapshot.provider, second_snapshot.provider
            )
            if provider_score is None:
                continue
            try:
                ordered = sorted(
                    (first, second), key=lambda item: _calibration_timestamp(item[1])
                )
            except ValueError:
                continue
            older, newer = ordered
            span = (
                _calibration_timestamp(newer[1]) - _calibration_timestamp(older[1])
            ).total_seconds()
            if span <= 0:
                continue
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


def _try_live_harvest(
    root: Path, config: dict, errors: list[dict]
) -> tuple[bool | None, list[Path]]:
    try:
        paths = harvest_ibm(
            root / "snapshots",
            max_backends=int(config["live"]["max_backends"]),
        )
        return True, paths
    except CredentialsUnavailable as exc:
        errors.append({"stage": "live_harvest", "error": str(exc)})
        return False, []
    except Exception as exc:
        errors.append({"stage": "live_harvest", "error": f"{type(exc).__name__}: {exc}"})
        return None, []


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


def _plot_cross_framework(
    rows: list[dict], path: Path, threshold: float, data_label: str, passed: bool
) -> None:
    import matplotlib.pyplot as plt

    pairs = ["qiskit-cirq", "qiskit-pennylane", "cirq-pennylane"]
    if not rows:
        _placeholder_figure(path, "Cross-framework TVD", "No complete three-framework circuits were available.")
        return
    matrix = np.array([[row["pairwise_tvd"].get(pair, np.nan) for pair in pairs] for row in rows])
    height = max(4.5, min(14, 0.27 * len(rows) + 2.5))
    fig, ax = plt.subplots(figsize=(9, height))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=threshold)
    ax.set_xticks(range(len(pairs)), pairs, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), [row["circuit"] for row in rows], fontsize=7)
    maximum = float(np.nanmax(matrix))
    verdict = "PASS" if passed else "FAIL"
    ax.set_title(
        f"Cross-framework TVD — {verdict} (max {maximum:.3e} ≤ {threshold:.3g})\n{data_label}"
    )
    fig.colorbar(image, ax=ax, label="TVD")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ghz(
    rows: list[dict],
    path: Path,
    data_label: str,
    selected_path: list[int],
    max_framework_tvd: float | None,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        _placeholder_figure(path, "GHZ fidelity", "No GHZ results were available.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    by_framework = defaultdict(list)
    for row in rows:
        by_framework[row["framework"]].append((row["num_qubits"], row["fidelity"]))
    styles = {
        "reference": ("--", "D"),
        "qiskit": ("-", "o"),
        "cirq": ("-.", "s"),
        "pennylane": (":", "^"),
    }
    for framework, values in sorted(by_framework.items()):
        values.sort()
        linestyle, marker = styles.get(framework, ("-", "o"))
        ax.plot(
            [x for x, _ in values],
            [y for _, y in values],
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="none",
            linewidth=1.7,
            label=framework,
        )
    sizes = sorted({row["num_qubits"] for row in rows})
    ax.plot(sizes, [1.0] * len(sizes), linestyle="--", label="ideal")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Fidelity to ideal GHZ distribution")
    ax.set_ylim(0, 1.02)
    deviation = "n/a" if max_framework_tvd is None else f"{max_framework_tvd:.3e}"
    ax.set_title(
        "GHZ distribution fidelity under archived noise\n"
        f"{data_label}\n"
        f"Nested path {selected_path}; max framework TVD {deviation}",
        fontsize=10.5,
    )
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
            if gate.operational
            and gate.error is not None
            and len(gate.qubits) == 2
            and tuple(gate.qubits) == (a, b)
            and gate.name in {"cx", "ecr", "cz"}
        ]
        values.append(float(min(errors)) if errors else np.nan)
    return values


def _plot_drift(
    older,
    newer,
    common: list[int],
    path: Path,
    pair_classification: str,
    simulation_tvd: float | None,
) -> None:
    import matplotlib.pyplot as plt

    old_q = {q.index: q for q in older.qubits}
    new_q = {q.index: q for q in newer.qubits}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    older_label = f"{older.provider}: {older.provenance.source_timestamp}"
    newer_label = f"{newer.provider}: {newer.provenance.source_timestamp}"
    axes[0].plot(common, [old_q[q].t1_us for q in common], marker="o", label=older_label)
    axes[0].plot(common, [new_q[q].t1_us for q in common], marker="o", label=newer_label)
    axes[0].set_title("T1")
    axes[0].set_xlabel("Physical qubit")
    axes[0].set_ylabel("µs")
    axes[0].set_xticks(common)
    axes[0].legend()

    axes[1].plot(common, [old_q[q].t2_us for q in common], marker="o", label=older_label)
    axes[1].plot(common, [new_q[q].t2_us for q in common], marker="o", label=newer_label)
    axes[1].set_title("T2")
    axes[1].set_xlabel("Physical qubit")
    axes[1].set_ylabel("µs")
    axes[1].set_xticks(common)
    axes[1].legend()

    edges = [f"{a}-{b}" for a, b in zip(common, common[1:], strict=False)]
    x = np.arange(len(edges))
    width = 0.36
    axes[2].bar(
        x - width / 2, _mean_two_qubit_errors(older, common), width, label=older_label
    )
    axes[2].bar(
        x + width / 2, _mean_two_qubit_errors(newer, common), width, label=newer_label
    )
    axes[2].set_xticks(x, edges)
    axes[2].set_title("Best two-qubit gate error")
    axes[2].set_xlabel("Edge")
    axes[2].legend()
    tvd_label = "not available" if simulation_tvd is None else f"{simulation_tvd:.6g}"
    provenance_warning = (
        "DETERMINISTIC SYNTHETIC FIXTURES"
        if older.provider == newer.provider == "demo"
        else pair_classification.upper()
    )
    fig.suptitle(
        f"Calibration comparison: {older.backend_name} — {provenance_warning}\n"
        f"{pair_classification}; representative-circuit TVD {tvd_label}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_pilot(root: str | Path, *, mode: str = "auto", profile: str = "full") -> dict:
    root = Path(root).resolve()
    config = _load_config(root)
    errors: list[dict] = []

    availability = _framework_availability()
    fake_backend = None
    live_authentication_available: bool | None = None
    live_paths: list[Path] = []
    if mode in {"auto", "live"}:
        live_authentication_available, live_paths = _try_live_harvest(root, config, errors)
    if mode in {"auto", "offline", "live"} and availability["qiskit"]:
        fake_backend = _try_fake_backend(root, errors)

    discovered = _discover_snapshots(root)
    if not discovered:
        raise RuntimeError("No snapshots are available, including the committed offline fixtures.")
    primary_path, primary = _choose_primary(discovered)
    validation = validate_snapshot(primary)
    if not validation.valid:
        raise RuntimeError("Primary snapshot is invalid.")
    archive_validation = [
        {
            "path": str(path.relative_to(root)),
            "warnings": [issue.__dict__ for issue in validate_snapshot(snapshot).warnings],
            "errors": [issue.__dict__ for issue in validate_snapshot(snapshot).errors],
        }
        for path, snapshot in discovered
    ]

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
    circuit_sizes = sorted({c.num_qubits for c in circuits})
    longest_path = best_connected_subset(primary, max(circuit_sizes))
    selected_by_size = {size: longest_path[:size] for size in circuit_sizes}
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

    requested_frameworks = list(runners)
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
    experiment_b_complete = (
        all(availability.values()) and len(pairwise_rows) == len(circuits)
    )
    experiment_b = {
        "status": "complete" if experiment_b_complete else "incomplete",
        "expected_circuit_count": len(circuits),
        "complete_circuit_count": len(pairwise_rows),
        "max_pairwise_tvd": max_pairwise,
        "mean_pairwise_tvd": mean_pairwise,
        "threshold": threshold_b,
        "passed": (
            experiment_b_complete
            and max_pairwise is not None
            and max_pairwise <= threshold_b
        ),
        "rows": pairwise_rows,
    }

    ghz_rows = []
    ghz_pair_values = []
    ghz_circuit_count = 0
    for circuit in circuits:
        if circuit.family != "ghz":
            continue
        ghz_circuit_count += 1
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
        framework_results = raw_results[circuit.name]
        if all(name in framework_results for name in runners):
            for first, second in combinations(runners, 2):
                ghz_pair_values.append(
                    total_variation_distance(
                        framework_results[first].probabilities,
                        framework_results[second].probabilities,
                    )
                )
    experiment_c_complete = len(ghz_rows) == ghz_circuit_count * (len(runners) + 1)
    experiment_c = {
        "status": "complete" if experiment_c_complete else "incomplete",
        "expected_row_count": ghz_circuit_count * (len(runners) + 1),
        "max_cross_framework_tvd": max(ghz_pair_values, default=None),
        "nested_physical_path": longest_path,
        "rows": ghz_rows,
    }

    try:
        (older_path, older), (newer_path, newer) = _choose_drift_pair(discovered)
        maximum_drift_size = min(3, older.num_qubits, newer.num_qubits)
        common = None
        for candidate_size in range(maximum_drift_size, 1, -1):
            try:
                common = best_common_connected_path(older, newer, candidate_size)
                break
            except ValueError:
                continue
        if common is None:
            raise RuntimeError(
                "The selected dated pair has no shared directed calibrated path of at least two qubits."
            )
        drift_size = len(common)
        drift_rows = []
        drift_circuit = mirror(
            drift_size, 4 if profile == "smoke" else 8, config["seed"]
        )
        for framework, runner in {"reference": run_reference, **runners}.items():
            if framework != "reference" and not availability[framework]:
                continue
            try:
                old_result = runner(older, drift_circuit, common)
                new_result = runner(newer, drift_circuit, common)
                drift_rows.append(
                    {
                        "framework": framework,
                        "circuit_name": drift_circuit.name,
                        "num_qubits": drift_size,
                        "physical_qubits": common,
                        "older_probabilities": old_result.probabilities.tolist(),
                        "newer_probabilities": new_result.probabilities.tolist(),
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
        pair_classification, empirical_calibration = _drift_pair_classification(
            older.provider, newer.provider
        )
        expected_drift_rows = 1 + sum(availability.values())
        drift_complete = len(drift_rows) == expected_drift_rows
        experiment_d = {
            "status": "complete" if drift_complete else "incomplete",
            "expected_row_count": expected_drift_rows,
            "older_snapshot": str(older_path.relative_to(root)),
            "newer_snapshot": str(newer_path.relative_to(root)),
            "backend_name": older.backend_name,
            "simulation_tvd": drift_tvd,
            "rows": drift_rows,
            "data_provider_pair": [older.provider, newer.provider],
            "pair_classification": pair_classification,
            "empirical_calibration_evidence": empirical_calibration,
            "older_source_timestamp": older.provenance.source_timestamp,
            "newer_source_timestamp": newer.provenance.source_timestamp,
            "older_raw_hash": older.provenance.raw_hash,
            "newer_raw_hash": newer.provenance.raw_hash,
            "common_qubits": common,
        }
        _plot_drift(
            older,
            newer,
            common,
            root / "experiments/figures/drift_comparison.png",
            pair_classification,
            drift_tvd,
        )
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
            fake_path, fake_snapshot = max(
                fake_candidates, key=lambda item: _timestamp(item[1])
            )
            native_result = compare_native_aer(fake_backend, fake_snapshot, num_qubits=2)
            max_calibration_delta = max(
                (
                    float(row["absolute_difference"])
                    for row in native_result.calibration_checks
                ),
                default=None,
            )
            calibration_tolerance = 1e-10
            experiment_a = {
                "status": native_result.status,
                "backend_name": native_result.backend_name,
                "snapshot": str(fake_path.relative_to(root)),
                "data_provider": fake_snapshot.provider,
                "max_tvd": native_result.max_tvd,
                "threshold": threshold_a,
                "threshold_comparison": "strictly_below",
                "max_calibration_error_delta": max_calibration_delta,
                "calibration_error_tolerance": calibration_tolerance,
                "calibration_target_passed": (
                    max_calibration_delta is not None
                    and max_calibration_delta <= calibration_tolerance
                ),
                "passed": (
                    native_result.max_tvd is not None
                    and native_result.max_tvd < threshold_a
                    and max_calibration_delta is not None
                    and max_calibration_delta <= calibration_tolerance
                ),
                "circuits": native_result.circuits,
                "calibration_checks": native_result.calibration_checks,
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

    providers = {snapshot.provider for _, snapshot in discovered}
    if "ibm" in providers:
        classification = "Contains live IBM calibration snapshots; inspect provenance per file."
    elif "qiskit_fake" in providers:
        classification = "Packaged Qiskit fake-backend calibrations plus deterministic fixtures; no live hardware claim."
    else:
        classification = "Deterministic synthetic fixtures only; pipeline verification, not empirical device evidence."

    primary_label = (
        f"{primary.provider}/{primary.backend_name}; calibration "
        f"{primary.provenance.source_timestamp or 'timestamp unavailable'}"
    )
    _plot_cross_framework(
        pairwise_rows,
        root / "experiments/figures/cross_framework_tvd.png",
        threshold_b,
        primary_label,
        bool(experiment_b["passed"]),
    )
    _plot_ghz(
        ghz_rows,
        root / "experiments/figures/ghz_fidelity.png",
        primary_label,
        longest_path,
        experiment_c["max_cross_framework_tvd"],
    )

    complete_frameworks = [
        name
        for name in runners
        if any(name in result_map for result_map in raw_results.values())
        and not any(item["stage"].startswith(f"simulate:{name}:") for item in errors)
    ]
    all_three = set(complete_frameworks) == set(runners)
    required_complete = (
        all_three
        and experiment_a["passed"]
        and experiment_b["passed"]
        and experiment_c["status"] == "complete"
        and experiment_d["status"] == "complete"
    )
    threshold_missed = (
        experiment_a.get("status") == "complete"
        and experiment_a.get("max_tvd") is not None
        and experiment_a["max_tvd"] >= threshold_a
    ) or (
        experiment_b["status"] == "complete"
        and max_pairwise is not None
        and max_pairwise > threshold_b
    )
    if required_complete:
        if "ibm" in providers:
            status = "PILOT_VERIFIED_WITH_LIVE_CALIBRATION_METADATA"
        elif "qiskit_fake" in providers or "ibm_csv" in providers:
            status = "OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS"
        else:
            status = "OFFLINE_PIPELINE_VERIFIED"
    elif all_three and threshold_missed:
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
        "live_ibm_snapshot_count": sum(
            snapshot.provider == "ibm" for _, snapshot in discovered
        ),
        "live_snapshots_harvested_this_run": len(live_paths),
        "live_ibm_authentication_available": live_authentication_available,
        "hardware_job_ran": False,
        "frameworks_requested": requested_frameworks,
        "frameworks_completed": complete_frameworks,
        "data_classification": classification,
        "integrity_statement": (
            "No result is labeled as live hardware evidence unless its snapshot provenance identifies the IBM Quantum Platform. "
            "Committed demo snapshots are synthetic fixtures, kept only so the full pipeline can be tested, in CI or locally, without needing real credentials."
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
        "snapshots": [
            {
                "path": str(path.relative_to(root)),
                "provider": snapshot.provider,
                "backend_name": snapshot.backend_name,
                "captured_at": snapshot.captured_at,
                "source_timestamp": snapshot.provenance.source_timestamp,
                "raw_hash": snapshot.provenance.raw_hash,
                "num_qubits": snapshot.num_qubits,
            }
            for path, snapshot in discovered
        ],
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "experiment_c": experiment_c,
        "experiment_d": experiment_d,
        "simulations": simulations,
        "validation": {
            "warnings": [issue.__dict__ for issue in validation.warnings],
            "errors": [issue.__dict__ for issue in validation.errors],
            "archive": archive_validation,
        },
        "errors": errors,
    }
    write_json(data, root / "experiments/results/pilot_results.json")
    (root / "experiments/PILOT_RESULTS.md").write_text(render_pilot_report(data), encoding="utf-8")
    return data
