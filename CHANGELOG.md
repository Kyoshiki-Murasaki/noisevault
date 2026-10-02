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
  citation with the full fingerprint, the NoiseVault version and the ref that loads the cited
  calibration. On a mismatch, the error names the calibration the ref loaded. If you have the
  pinned calibration, the hint is the `nv.load` call, with the same pin, that loads it. A dated
  ref with no calibration on that UTC day fails with "no ibm_fez profile calibrated on
  2025-03-01 UTC", and for a device that `nv pull` serves, the hint is
  `nv pull ibm_fez --at 2025-03-01T23:59:59Z` and says to load the ref that pull prints.
- **25 bundled profiles that load offline**: 18 IBM devices from qiskit-ibm-runtime, 5
  Quantinuum machines from Quantinuum's published benchmark data, and Google's Rainbow and
  Weber from cirq-google. All are Apache-2.0 data, and the files are byte-identical on
  Python 3.11 to 3.14.
- **Four exports**, each returning the framework's own object with a `.report` of what it
  reproduces exactly, approximates or omits:
  - `to_qiskit()`: an `AerSimulator` with a device `Target`, so `transpile` compiles to the
    device's natives and avoids disabled gates.
  - `to_cirq()`: a `cirq.NoiseModel`. `GridQubit(r, c)` addresses the qubit at those coords.
  - `to_pennylane()`: a `qml.NoiseModel` for `qml.add_noise`, with gradients through the noise.
    A broadcast whose angles need different natives raises and names the fix,
    `qml.transforms.broadcast_expand`. Operator arithmetic such as `qml.prod` gets the noise of
    the gates it decomposes into.
  - `to_stim(circuit)`: a noisy copy of a Stim circuit for QEC-size sampling and decoding.
    `CXSWAP`, `SWAPCX` and `CZSWAP` are the registry gates `cxswap`, `swapcx` and `czswap`.
    Each takes the profile's calibration of that gate. Without one, the export asks you to
    decompose it.
- **One rule for fixed-angle gates in every export.** A gate such as `s`, `sx` or `ms` takes
  its own native's noise if the profile has it, else the noise of the rotation it equals.
- **No typical noise for gates that need a decomposition.** A gate that needs several native
  entanglers, such as `swap` or `ccx`, or acts on more than two qubits, never takes the typical
  native gate's noise. If the profile does not calibrate it, the export raises
  `MissingCalibrationError` with either `unknown_gates` value and asks you to decompose it.
- **Readable reports.** Approximation warnings point at your own line of code, and
  `report.summary()` states counts as sentences, such as "cx took the typical native gate's
  noise 40 times". The report's usage counts print on one line that starts with `used:`, and
  the summary names included errors in plain words, such as "single-qubit gate error".
- **Live pulls without an account**: IBM's public calibration endpoint, with history through
  `--at`, and IonQ's published characterizations. Pulls through an IBM account are also
  supported. Pulled profiles are saved to `~/.noisevault/profiles`.
- **Importers**: `from_qiskit_backend`, `from_ibm_csv`, `from_braket`, `from_cirq_google`, and
  Quantinuum's dated datasets. `from_braket` maps each native to the gate with the same
  matrix. `from_qiskit_backend` allows a gate only on the qubits its `Target` lists, takes the
  technology from the backend, and labels a fake whose snapshot is a model, such as
  `FakeNighthawk`, as `vendor_model`. An IBM gate error missing from a CSV or a pull takes the
  device median, and `provenance.notes` names those qubits and pairs. `from_ibm_csv` reads
  IBM's CSV formats from 2023 to 2026 and treats a cell that says `undefined` as blank. It
  refuses a file in an older format, a copy that a spreadsheet saved, or a file with two
  columns for the same value, each with a one-line error that says why.
- **Hypothetical devices** with `Profile.uniform`, and `profile.suggest_layout(n)` to pick a
  well-calibrated chain of qubits that each have every one-qubit native the device has.
- **Command line** `nv` (also `noisevault`): `list`, `show`, `pull`, `diff`, `check`, `cite`,
  `validate`, `doctor` and `schema`.
  - A bare `nv` prints the help, which ends with three commands to start with. A usage
    mistake prints one line with the closest match.
  - A failure prints what went wrong on an `error:` line and the next step on a `hint:` line.
    In Python, a `NoiseVaultError` keeps that step in `hint`, and `str(error)` ends with it.
  - `nv show` and `nv diff` label gate errors as average gate infidelity.
    When you have several calibrations of a device, `nv show` and `nv check` say which one a
    bare id loaded, and `nv check` names the calibration it checked.
  - `nv check` lists missing frameworks with one install command, and with no framework
    installed it prints one error and one install command. It counts a circuit that ran with
    gates removed as reduced. It also samples a measurement-only circuit through each
    framework's own readout. Each row shows the TVD and the tolerance of the circuit with the
    highest ratio of TVD to tolerance. `--json` reports that circuit's name, TVD and tolerance
    under `worst`.
  - `nv list` keeps one device's calibrations together, newest first, and shows the time when
    two calibrations share a date. It never cuts an id or a date. In a narrow terminal it
    leaves out the source, then the processor, then the license column, and its last line
    then names each license. A license all profiles share is stated once.
  - `nv validate` says what an unknown or a missing key means.
  - Output writes commands plainly, without backticks, so they paste into a shell as they are,
    and no line ends in padding spaces.
  - `nv diff` marks values as new or gone and compares both orders of a symmetric pair.
  - A damaged or unreadable file in the vault is skipped with a warning, and the other
    profiles still list and load.
- Documentation in [docs/](docs/) and runnable [examples](examples/).

### Changed

- Install from GitHub with the extra for your framework:
  `pip install "noisevault[qiskit] @ git+https://github.com/Kyoshiki-Murasaki/noisevault"`.
  To try the command line without installing, run
  `uvx --from git+https://github.com/Kyoshiki-Murasaki/noisevault nv show ibm_fez`.
- The core install needs only numpy, pydantic, typer and rich. Qiskit, Cirq, PennyLane and
  Stim are extras, and importing `noisevault` imports none of them.
- Minimum versions are pennylane 0.43.3, stim 1.15, typer 0.27 and rich 13.8. CI installs
  every direct dependency at its declared minimum and runs the tests.
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
