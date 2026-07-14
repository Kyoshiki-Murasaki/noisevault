# NoiseVault schema 0.1

A snapshot is an immutable description of one provider/backend calibration state at one timestamp.

Required top-level fields:

- `schema_version`: currently `0.1`.
- `provider`: source namespace such as `ibm`, `qiskit_fake`, `ibm_csv`, or `demo`.
- `backend_name`: provider-visible backend identifier.
- `captured_at`: timezone-qualified ISO-8601 timestamp recording when NoiseVault captured the source.
- `num_qubits`, `coupling_map`, and `basis_gates`.
- `qubits`: one record per physical qubit.
- `gates`: calibrated instruction instances with explicit physical qargs.
- `provenance`: source description, raw source hash, harvester version, and optional source timestamp.
- `extensions`: reserved namespaced data.

## Qubit fields

`index`, `t1_us`, `t2_us`, `frequency_ghz`, `readout_error`, `prob_meas0_prep1`, `prob_meas1_prep0`, and `operational`.

Null means unavailable. Null never means zero. Raw unphysical values are preserved, flagged during validation, and only clamped at the simulation boundary where a completely positive channel requires it.

## Gate fields

`name`, ordered `qubits`, average `error`, `duration_ns`, and `operational`.

A zero gate duration is allowed for virtual operations such as `rz`. Missing duration invokes a documented conservative pilot default at conversion time; this is recorded as an approximation, not source data.

Two-qubit `qubits` order is semantic. Converters and subset selection do not assume that `[control, target]` can borrow calibration data from `[target, control]`. A requested directed entangler without a matching ordered calibration is rejected.

## Provenance and hashing

`raw_hash` hashes the provider payload before normalization. It is not a signature and does not establish provider authenticity. It establishes that two NoiseVault imports used byte-equivalent normalized source content.

Each archived filename includes a full timestamp rather than only a date so intraday calibration changes cannot overwrite one another.

`captured_at` records the archive/import event. Temporal-drift experiments require a distinct timezone-qualified `provenance.source_timestamp` that identifies the underlying calibration state; an undated CSV import cannot use its import time as a calibration date.
