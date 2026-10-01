# Changelog

All notable changes to NoiseVault. Versions follow [Semantic Versioning](https://semver.org).

## 0.2.0 (2026-10-01)

A rebuild around one hardware-agnostic file format. Code written for 0.1 needs changes; 0.1
profile files still load.

### Added

- **Profile format 1.0.** One JSON file per device calibration, for superconducting,
  trapped-ion, neutral-atom, spin and photonic qubits. Each error metric has its own key
  (`avg_infidelity`, `process_infidelity`, `depolarizing_param`, `pauli`), so no number is
  guessed. Per-qubit and per-gate records override device-wide defaults. Disabled gates and
  qubits, directed connectivity, provenance and license fields are part of the format. See
  [docs/profile-format.md](docs/profile-format.md).
- **Fingerprints.** Every profile has a SHA-256 fingerprint of its physics.
  `nv.load(ref, expect="nv:...")` fails if the profile changed, and `nv cite` prints a
  citation with the full fingerprint.
- **25 bundled profiles that load offline**: 18 IBM devices from qiskit-ibm-runtime, 5
  Quantinuum machines from Quantinuum's published benchmark data, and Google's Rainbow and
  Weber from cirq-google. All are Apache-2.0 data.
- **Four exports**, each returning the framework's own object with a `.report` of what it
  reproduces exactly, approximates or omits:
  - `to_qiskit()`: an `AerSimulator` with a device `Target`, so `transpile` compiles to the
    device's natives and avoids disabled gates.
  - `to_cirq()`: a `cirq.NoiseModel`. `GridQubit(r, c)` addresses the qubit at those coords.
  - `to_pennylane()`: a `qml.NoiseModel` for `qml.add_noise`, with gradients through the noise.
  - `to_stim(circuit)`: a noisy copy of a Stim circuit for QEC-size sampling and decoding.
- **Live pulls without an account**: IBM's public calibration endpoint, with history through
  `--at`, and IonQ's published characterizations. Pulls through an IBM account are also
  supported. Pulled profiles are saved to `~/.noisevault/profiles`.
- **Importers**: `from_qiskit_backend`, `from_ibm_csv`, `from_braket`, `from_cirq_google`, and
  Quantinuum's dated datasets.
- **Hypothetical devices** with `Profile.uniform`, and `profile.suggest_layout(n)` to pick a
  well-calibrated chain of qubits.
- **Command line** `nv` (also `noisevault`): `list`, `show`, `pull`, `diff`, `check`, `cite`,
  `validate`, `doctor` and `schema`.
- Documentation in [docs/](docs/) and runnable [examples](examples/).

### Changed

- The core install needs only numpy, pydantic, typer and rich. Qiskit, Cirq, PennyLane and
  Stim are extras (`pip install 'noisevault[all]'`), and importing `noisevault` imports none of
  them.
- Supports Python 3.11 to 3.14.
- Format 0.1 files are upgraded in memory with a `MigrationWarning`. Save them to keep the 1.0
  form.

### Removed

- The 0.1 Python API, command line, snapshot folder and experiment scripts.

## 0.1.0 (2026-07-14)

Initial release.

- Schema 0.1: dated JSON snapshots of IBM device calibrations with provenance and raw-payload
  hashes.
- Importers for IBM Quantum accounts, Qiskit fake backends and IBM calibration CSV files.
- Converters to Qiskit Aer, Cirq and PennyLane noise models, compared on small benchmark
  circuits with exact density-matrix simulation.
