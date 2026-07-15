from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class QubitCalibration(StrictModel):
    index: int
    t1_us: float | None = None
    t2_us: float | None = None
    frequency_ghz: float | None = None
    readout_error: float | None = None
    prob_meas0_prep1: float | None = None
    prob_meas1_prep0: float | None = None
    operational: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class GateCalibration(StrictModel):
    name: str
    qubits: list[int]
    error: float | None = None
    duration_ns: float | None = None
    operational: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provenance(StrictModel):
    source: str
    raw_hash: str
    harvester_version: str = "0.1.0"
    source_timestamp: str | None = None
    source_url: str | None = None
    notes: list[str] = Field(default_factory=list)


class DeviceNoiseSnapshot(StrictModel):
    schema_version: str = "0.1"
    provider: str
    backend_name: str
    captured_at: str
    num_qubits: int
    coupling_map: list[list[int]]
    basis_gates: list[str]
    qubits: list[QubitCalibration]
    gates: list[GateCalibration]
    provenance: Provenance
    extensions: dict[str, Any] = Field(default_factory=dict)
