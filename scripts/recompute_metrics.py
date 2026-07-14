#!/usr/bin/env python3
"""Independently recompute NoiseVault headline metrics from saved vectors only."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def normalize(values: list[float]) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Probability vectors must be nonempty and finite.")
    if any(value < -1e-12 for value in values):
        raise ValueError("Probability vectors contain materially negative values.")
    clipped = [max(0.0, value) for value in values]
    total = sum(clipped)
    if total <= 0:
        raise ValueError("Probability vector has zero mass.")
    return [value / total for value in clipped]


def tvd(first: list[float], second: list[float]) -> float:
    first = normalize(first)
    second = normalize(second)
    if len(first) != len(second):
        raise ValueError("Probability-vector length mismatch.")
    return 0.5 * sum(abs(a - b) for a, b in zip(first, second, strict=True))


def hellinger_fidelity(first: list[float], second: list[float]) -> float:
    first = normalize(first)
    second = normalize(second)
    if len(first) != len(second):
        raise ValueError("Probability-vector length mismatch.")
    return sum(math.sqrt(a * b) for a, b in zip(first, second, strict=True)) ** 2


def close(first: float, second: float, tolerance: float = 1e-12) -> bool:
    return abs(first - second) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("experiments/results/pilot_results.json"),
    )
    args = parser.parse_args()
    data = json.loads(args.path.read_text(encoding="utf-8"))
    issues: list[str] = []

    normalization_errors = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for record in data["simulations"]:
        probabilities = record["probabilities"]
        normalize(probabilities)
        normalization_errors.append(abs(sum(probabilities) - 1.0))
        grouped[record["circuit_name"]][record["framework"]] = probabilities

    a_values = []
    for row in data["experiment_a"]["circuits"]:
        value = tvd(row["native"], row["reconstructed"])
        a_values.append(value)
        if not close(value, row["tvd"]):
            issues.append(f"Experiment A mismatch: {row['name']}")
    a_max = max(a_values)
    if not close(a_max, data["experiment_a"]["max_tvd"]):
        issues.append("Experiment A maximum mismatch")

    b_values = []
    complete_circuits = 0
    for frameworks in grouped.values():
        if all(name in frameworks for name in ("qiskit", "cirq", "pennylane")):
            complete_circuits += 1
            for first, second in combinations(("qiskit", "cirq", "pennylane"), 2):
                b_values.append(tvd(frameworks[first], frameworks[second]))
    b_max = max(b_values)
    b_mean = sum(b_values) / len(b_values)
    if not close(b_max, data["experiment_b"]["max_pairwise_tvd"]):
        issues.append("Experiment B maximum mismatch")
    if not close(b_mean, data["experiment_b"]["mean_pairwise_tvd"]):
        issues.append("Experiment B mean mismatch")

    c_values = []
    for row in data["experiment_c"]["rows"]:
        probabilities = grouped[row["circuit"]][row["framework"]]
        ideal = [0.0] * len(probabilities)
        ideal[0] = ideal[-1] = 0.5
        value = hellinger_fidelity(probabilities, ideal)
        c_values.append(value)
        if not close(value, row["fidelity"]):
            issues.append(
                f"Experiment C mismatch: {row['circuit']}/{row['framework']}"
            )

    d_values = []
    for row in data["experiment_d"]["rows"]:
        value = tvd(row["older_probabilities"], row["newer_probabilities"])
        d_values.append(value)
        if not close(value, row["tvd"]):
            issues.append(f"Experiment D mismatch: {row['framework']}")
    d_max = max(d_values)
    if not close(d_max, data["experiment_d"]["simulation_tvd"]):
        issues.append("Experiment D maximum mismatch")

    result = {
        "status": "PASS" if not issues else "FAIL",
        "input": str(args.path),
        "saved_simulation_vector_count": len(data["simulations"]),
        "maximum_normalization_error": max(normalization_errors, default=0.0),
        "experiment_a_recomputed_max_tvd": a_max,
        "experiment_a_threshold": data["experiment_a"]["threshold"],
        "experiment_a_passed": a_max < data["experiment_a"]["threshold"],
        "experiment_b_complete_circuit_count": complete_circuits,
        "experiment_b_recomputed_max_tvd": b_max,
        "experiment_b_recomputed_mean_tvd": b_mean,
        "experiment_b_threshold": data["experiment_b"]["threshold"],
        "experiment_b_passed": b_max <= data["experiment_b"]["threshold"],
        "experiment_c_recomputed_row_count": len(c_values),
        "experiment_d_recomputed_max_tvd": d_max,
        "issues": issues,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
