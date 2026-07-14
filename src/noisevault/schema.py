from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite

from .models import DeviceNoiseSnapshot


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, code, path, message))


def _check_probability(report: ValidationReport, value: float | None, path: str) -> None:
    if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
        report.add("error", "probability_out_of_range", path, f"Expected [0, 1], got {value}.")


def _check_finite(report: ValidationReport, value: float | None, path: str) -> bool:
    if value is not None and not isfinite(value):
        report.add("error", "nonfinite_value", path, f"Expected a finite value, got {value}.")
        return False
    return True


def _check_timestamp(report: ValidationReport, value: str, path: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            report.add("error", "naive_timestamp", path, "Timestamp must include a timezone.")
    except ValueError:
        report.add("error", "invalid_timestamp", path, "Timestamp is not ISO-8601.")


def validate_snapshot(snapshot: DeviceNoiseSnapshot) -> ValidationReport:
    report = ValidationReport()

    if snapshot.schema_version != "0.1":
        report.add(
            "warning",
            "unknown_schema_version",
            "schema_version",
            f"Pilot code was authored for schema 0.1, got {snapshot.schema_version}.",
        )
    if snapshot.num_qubits <= 0:
        report.add("error", "invalid_qubit_count", "num_qubits", "num_qubits must be positive.")
    _check_timestamp(report, snapshot.captured_at, "captured_at")
    if snapshot.provenance.source_timestamp is not None:
        _check_timestamp(
            report, snapshot.provenance.source_timestamp, "provenance.source_timestamp"
        )

    indices = [qubit.index for qubit in snapshot.qubits]
    if len(indices) != len(set(indices)):
        report.add("error", "duplicate_qubit", "qubits", "Qubit indices must be unique.")
    expected = set(range(snapshot.num_qubits))
    actual = set(indices)
    if actual != expected:
        report.add(
            "error",
            "incomplete_qubit_table",
            "qubits",
            f"Expected indices {min(expected, default=0)}..{max(expected, default=0)}; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}.",
        )

    for i, qubit in enumerate(snapshot.qubits):
        path = f"qubits[{i}]"
        t1_finite = _check_finite(report, qubit.t1_us, f"{path}.t1_us")
        t2_finite = _check_finite(report, qubit.t2_us, f"{path}.t2_us")
        frequency_finite = _check_finite(
            report, qubit.frequency_ghz, f"{path}.frequency_ghz"
        )
        if t1_finite and qubit.t1_us is not None and qubit.t1_us <= 0:
            report.add("error", "nonpositive_t1", f"{path}.t1_us", "T1 must be positive.")
        if t2_finite and qubit.t2_us is not None and qubit.t2_us <= 0:
            report.add("error", "nonpositive_t2", f"{path}.t2_us", "T2 must be positive.")
        if frequency_finite and qubit.frequency_ghz is not None and qubit.frequency_ghz <= 0:
            report.add(
                "error",
                "nonpositive_frequency",
                f"{path}.frequency_ghz",
                "Frequency must be positive.",
            )
        if (
            t1_finite
            and t2_finite
            and qubit.t1_us is not None
            and qubit.t2_us is not None
            and qubit.t2_us > 2.0 * qubit.t1_us
        ):
            report.add(
                "warning",
                "unphysical_t2",
                f"{path}.t2_us",
                f"T2={qubit.t2_us} exceeds 2*T1={2*qubit.t1_us}; converters clamp T2 for simulation but preserve the raw value.",
            )
        _check_probability(report, qubit.readout_error, f"{path}.readout_error")
        _check_probability(report, qubit.prob_meas0_prep1, f"{path}.prob_meas0_prep1")
        _check_probability(report, qubit.prob_meas1_prep0, f"{path}.prob_meas1_prep0")

    valid_indices = set(range(snapshot.num_qubits))
    seen_gates: set[tuple[str, tuple[int, ...]]] = set()
    for i, gate in enumerate(snapshot.gates):
        path = f"gates[{i}]"
        key = (gate.name, tuple(gate.qubits))
        if key in seen_gates:
            report.add("warning", "duplicate_gate", path, f"Duplicate calibration for {key}.")
        seen_gates.add(key)
        if not gate.qubits:
            report.add("error", "empty_gate_qubits", f"{path}.qubits", "Gate must target qubits.")
        if not set(gate.qubits).issubset(valid_indices):
            report.add("error", "gate_qubit_out_of_range", f"{path}.qubits", str(gate.qubits))
        _check_probability(report, gate.error, f"{path}.error")
        duration_finite = _check_finite(report, gate.duration_ns, f"{path}.duration_ns")
        if duration_finite and gate.duration_ns is not None and gate.duration_ns < 0:
            report.add("error", "negative_gate_duration", f"{path}.duration_ns", "Duration cannot be negative.")
        elif gate.duration_ns == 0:
            report.add(
                "warning",
                "zero_gate_duration",
                f"{path}.duration_ns",
                "Zero duration is acceptable for virtual gates but produces no relaxation.",
            )

    edge_set = {tuple(edge) for edge in snapshot.coupling_map}
    for i, edge in enumerate(snapshot.coupling_map):
        if len(edge) != 2:
            report.add("error", "invalid_coupling_edge", f"coupling_map[{i}]", str(edge))
        elif not set(edge).issubset(valid_indices):
            report.add("error", "coupling_qubit_out_of_range", f"coupling_map[{i}]", str(edge))
    for gate in snapshot.gates:
        if len(gate.qubits) == 2 and tuple(gate.qubits) not in edge_set and tuple(reversed(gate.qubits)) not in edge_set:
            report.add(
                "warning",
                "gate_missing_from_coupling_map",
                "gates",
                f"{gate.name}{gate.qubits} has no corresponding coupling edge.",
            )

    return report
