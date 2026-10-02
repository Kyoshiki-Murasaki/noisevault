"""The values are invented."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).with_name("train-00000-of-00001.parquet")

ENVIRONMENT = (
    "latitude",
    "longitude",
    "solar_zenith_deg",
    "temperature_c",
    "pressure_hpa",
    "humidity_pct",
    "bz_gsm_nt",
    "neutron_flux",
    "kp_index",
    "ap_index",
    "Ap_daily",
    "SN",
    "f107_observed_sfu",
    "f107_adjusted_sfu",
    "solar_flux_sfu",
    "dst_nt",
)
TIME = pa.timestamp("us", tz="UTC")
SCHEMA = pa.schema(
    [
        ("backend", pa.large_string()),
        ("property_family", pa.large_string()),
        ("property", pa.large_string()),
        ("qubit_a", pa.int64()),
        ("qubit_b", pa.float64()),
        ("value", pa.float64()),
        ("unit", pa.large_string()),
        ("scope", pa.large_string()),
        ("is_failure_ceiling", pa.bool_()),
        ("observed_time", TIME),
        ("calibrated_time", TIME),
        ("snapshot_update_time", TIME),
        ("calibration_age_seconds", pa.float64()),
        ("is_new_measurement", pa.bool_()),
        ("chipwide_recal_event_id", pa.large_string()),
        *((name, pa.float64()) for name in ENVIRONMENT),
    ]
)
QUBIT_PROPERTIES = {
    "T1",
    "T2",
    "readout_error",
    "prob_meas0_prep1",
    "prob_meas1_prep0",
    "readout_length",
    "init_error",
}

FIRST_SEEN = datetime(2026, 1, 31, 17, 5, 3, tzinfo=UTC)
LEGACY_SEEN = datetime(2026, 1, 31, 17, 5, 3, 496046, tzinfo=UTC)
LEGACY = datetime(2026, 1, 31, 13, 38, 30, tzinfo=UTC)
BACKFILL = datetime(2026, 1, 29, 9, 44, 10, tzinfo=UTC)
A_SEEN = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
A = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
B_SEEN = datetime(2026, 6, 2, 9, 0, tzinfo=UTC)
B = datetime(2026, 6, 2, 8, 0, tzinfo=UTC)
B_CALIBRATED_AFTER_SEEN = datetime(2026, 6, 2, 9, 0, 14, 800000, tzinfo=UTC)
TORINO_LAST_SEEN = datetime(2026, 4, 1, 13, 20, 0, tzinfo=UTC)
TORINO_LAST = datetime(2026, 4, 1, 12, 56, 1, tzinfo=UTC)

FEZ_QUBITS = range(4)
FEZ_PAIRS = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]


def row(
    backend: str,
    prop: str,
    locus: tuple[int, ...],
    value: float,
    unit: str,
    calibrated: datetime,
    seen: datetime,
    generation: str,
    *,
    ceiling: bool = False,
) -> dict[str, Any]:
    a, *b = locus
    return {
        "backend": backend,
        "property_family": prop.replace("_gate", ""),
        "property": prop,
        "qubit_a": a,
        "qubit_b": float(b[0]) if b else None,
        "value": value,
        "unit": None if generation == "legacy" else unit,
        "scope": None
        if generation == "legacy"
        else ("qubit" if prop in QUBIT_PROPERTIES else f"gate{len(locus)}q"),
        "is_failure_ceiling": ceiling,
        "observed_time": seen,
        "calibrated_time": calibrated,
        "snapshot_update_time": None if generation == "legacy" else seen,
        "calibration_age_seconds": (seen - calibrated).total_seconds(),
        "is_new_measurement": True if generation == "polled" else None,
        "chipwide_recal_event_id": None,
        **dict.fromkeys(ENVIRONMENT),
        "SN": 120.0,
    }


def legacy(backend: str, qubits: range, pairs: list, calibrated: datetime, seen: datetime) -> list:
    out = []
    for q in qubits:
        for prop, value in (
            ("T1", 4.854e-05 + q * 1e-06),
            ("T2", 3.0e-05 + q * 1e-06),
            ("readout_error", 0.02),
            ("prob_meas0_prep1", 0.025),
            ("prob_meas1_prep0", 0.015),
            ("sx_gate_error", 4e-04),
        ):
            out.append(row(backend, prop, (q,), value, "", calibrated, seen, "legacy"))
    for pair in pairs:
        out.append(row(backend, "cz_gate_error", pair, 0.0439, "", calibrated, seen, "legacy"))
    return out


def backfill() -> list:
    out = []
    for q in FEZ_QUBITS:
        for prop, value, unit in (
            ("sx_gate_length", 32.0, "ns"),
            ("x_gate_error", 4e-04, ""),
            ("x_gate_length", 32.0, "ns"),
            ("rz_gate_error", 0.0, ""),
            ("rz_gate_length", 0.0, "ns"),
            ("readout_length", 1560.0, "ns"),
        ):
            out.append(row("ibm_fez", prop, (q,), value, unit, BACKFILL, FIRST_SEEN, "backfill"))
    for pair in FEZ_PAIRS:
        out.append(
            row("ibm_fez", "cz_gate_length", pair, 68.0, "ns", BACKFILL, FIRST_SEEN, "backfill")
        )
    return out


def calibration_a() -> list:
    out = []
    for q in FEZ_QUBITS:
        for prop, value, unit in (
            ("T1", 100.0 + q, "us"),
            ("T2", 80.0 + q, "us"),
            ("readout_error", 0.01, ""),
            ("prob_meas0_prep1", 0.012, ""),
            ("prob_meas1_prep0", 0.008, ""),
            ("readout_length", 1560.0, "ns"),
            ("init_error", 0.001, ""),
            ("measure_gate_error", 0.01, ""),
            ("measure_gate_length", 1560.0, "ns"),
            ("measure_threshold", -3.1e08, ""),
            ("measure_2_gate_length", 1760.0, "ns"),
            ("measure_2_threshold", 1.9e08, ""),
            ("sx_gate_error", 2e-04 + q * 1e-05, ""),
            ("sx_gate_length", 32.0, "ns"),
            ("x_gate_error", 2e-04 + q * 1e-05, ""),
            ("x_gate_length", 32.0, "ns"),
            ("id_gate_error", 2e-04, ""),
            ("id_gate_length", 32.0, "ns"),
            ("rz_gate_error", 0.0, ""),
            ("rz_gate_length", 0.0, "ns"),
        ):
            out.append(row("ibm_fez", prop, (q,), value, unit, A, A_SEEN, "polled"))
    cz = {(0, 1): 0.003, (1, 0): 0.003, (1, 2): 0.004, (2, 1): 0.006, (2, 3): 0.005, (3, 2): 0.005}
    for pair, error in cz.items():
        for prop, value, unit in (
            ("cz_gate_error", error, ""),
            ("cz_gate_length", 68.0, "ns"),
            ("rzz_gate_error", 0.004, ""),
            ("rzz_gate_length", 68.0, "ns"),
        ):
            out.append(row("ibm_fez", prop, pair, value, unit, A, A_SEEN, "polled"))
    return out


def calibration_b() -> list:
    return [
        row("ibm_fez", "T1", (0,), 120.0, "us", B, B_SEEN, "polled"),
        row("ibm_fez", "T2", (2,), 90.0, "us", B_CALIBRATED_AFTER_SEEN, B_SEEN, "polled"),
        row("ibm_fez", "sx_gate_error", (2,), 1.0, "", B, B_SEEN, "polled", ceiling=True),
        row("ibm_fez", "prob_meas1_prep0", (3,), 1.0, "", B, B_SEEN, "polled", ceiling=True),
        row("ibm_fez", "cz_gate_error", (0, 1), 0.002, "", B, B_SEEN, "polled"),
        row("ibm_fez", "cz_gate_error", (1, 0), 0.002, "", B, B_SEEN, "polled"),
    ]


def main() -> None:
    torino_pairs = [(0, 1), (1, 0)]
    rows = [
        *legacy("ibm_fez", FEZ_QUBITS, FEZ_PAIRS, LEGACY, LEGACY_SEEN),
        *backfill(),
        *calibration_a(),
        *calibration_b(),
        *legacy("ibm_torino", range(2), torino_pairs, LEGACY, LEGACY_SEEN),
        *legacy("ibm_torino", range(2), torino_pairs, TORINO_LAST, TORINO_LAST_SEEN),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), OUT)


if __name__ == "__main__":
    main()
