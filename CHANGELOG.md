# Changelog

All notable changes to NoiseVault. Versions follow [Semantic Versioning](https://semver.org).

## Unreleased

### Added

- **Factors for the error a calibration leaves out.** A profile can set `unmodeled_error`, with
  a factor on its gate error rates, a factor on its readout error rates, or both. Every export
  and the `nv check` reference apply the factors, and `nv diff` compares the scaled errors. A
  factor raises each error channel to that power, so 1 keeps the stated error and no factor
  takes an error out of its physical range. T1, T2 and preparation error are never scaled. An
  error with no valid power, such as a readout pair no better than chance, stays as stated.
  - Each export's report states the factors on its second line, which starts with
    `unmodeled error:` and names each error left as stated. `report.to_dict()` holds the same
    text under `unmodeled_error`.
  - `nv show` prints the calibration as stated, then the factors on an `unmodeled` row.
  - The field is part of the fingerprint, and `nv cite` names the factors.
    `profile.uncorrected()` returns the profile without the field, with the calibration's
    fingerprint. Profiles without the field keep their fingerprints.
  - NoiseVault 0.2.0 refuses a profile that sets the field.

  See [Unmodeled error](docs/profile-format.md#unmodeled-error).
- **IBM calibration history from Hugging Face.** `nv.from_calibration_archive(path, device,
  at=...)` reads a local copy of the Hugging Face dataset `phanerozoic/qiskit-calibration-drift`
  and returns IBM's calibration of `ibm_fez`, `ibm_kingston`, `ibm_marrakesh` or `ibm_torino`
  as it stood at `at`. A provenance note names the values calibrated more than 7 days before
  `at`, or before the newest calibration when you give no `at`.
  `nv.calibration_archive_devices(path)` gives the range of times each device covers. Install
  `noisevault[hf]` (pyarrow 14.0.1 or later). `nv doctor` lists pyarrow. See
  [IBM calibration archive on Hugging Face](docs/data-sources.md#ibm-calibration-archive-on-hugging-face).
- **Scoring a profile on counts from a device.** `nv compare REF COUNTS` scores a profile on
  counts that a device measured. It fits one factor on every gate error rate and one on every
  readout error rate, each with a 95% interval, and tests whether one pair of factors explains
  every circuit. It exits with status 0 whatever the fit. `-o FILE` saves the profile with the
  fitted factors in `unmodeled_error`, and `--json` prints the result as JSON. Before the fit,
  `-o` refuses the counts file, the profile file and any path in the vault or among the bundled
  profiles. In Python, `profile.compare(counts)` returns the same result, and
  `result.fitted_profile()` returns the profile with the factors.
  - `noisevault.counts` reads and writes counts files. `load_counts(path)` reads one as a
    `MeasuredCounts`. `plan(profile)` returns the circuits to run, with every wait written as a
    `delay`. `simulate(profile, circuits, shots=..., seed=...)` draws counts from a profile.
  - `load_counts` raises `nv.CountsError` for a file it refuses. The message names the file and
    the field, and `hint` says how to fix it.
  - A command given a counts file where it takes a profile says so. With the two arguments
    swapped, `nv compare` names the order it takes them in.
  - `scripts/run_on_ibm.py` runs the `nv compare` circuits on an IBM device through your IBM
    Quantum account. It writes a counts file bound to the calibration in effect when the job ran.

  See [Counts format](docs/counts-format.md) and
  [Measure a profile against hardware](docs/recipes.md#measure-a-profile-against-hardware).

### Fixed

- **Revalidating a profile after reading its fingerprint.** `Profile.model_validate(profile)`, a
  pydantic model or `TypeAdapter` with a `Profile` field, and `Profile(**dict(profile))` raised
  `Extra inputs are not permitted` once `fingerprint`, `artifact_hash` or `table` had been read.
  They now accept the profile. With pydantic 2.13, assigning to `fingerprint`, `artifact_hash` or
  `table` replaced the stored value. The assignment now raises `ValidationError`, like any other
  change to a frozen profile.
- **Loading from a vault whose index disagrees with its files.** `nv.load`, the commands that
  take a ref, and `nv pull` trusted the vault's index, `.index.json`, without checking it
  against the files. An index entry with the wrong date let `ibm_manila@2024-06-04` load a
  calibration from 2024-06-03, and `nv pull` then saved over that calibration. A load could also
  return another calibration when a file was saved over during the load. NoiseVault now uses the
  index only when its version and digest show that NoiseVault wrote it, and otherwise reads the
  files again. `nv.load` checks the profile it read against the ref and the `expect=` pin, and
  reads the vault again when they differ. An `.index.json` that is not a regular file no longer
  stops `nv list`.
- **Importer input that is damaged or incomplete.** Three importers returned wrong values with
  no error. Each now raises `SourceDataError` that names the file and the field or line.
  - `from_ibm_csv` lost every row after a quote left open, so a file could import with no
    two-qubit records. A row cut short read its missing cells as blank, and a blank Operational
    cell enabled a disabled qubit. Both are now refused. A CSV whose lines end in a bare carriage
    return failed with a `csv` module error, and it now reads like any other.
  - `from_braket` read a time with no `unit` as seconds, so a T1 of 18.5 us became 18,500,000
    us. It read a standardized v1 or v2 two-qubit fidelity with no `fidelityType` as randomized
    benchmarking. On one IQM pair, the misread changed the error from 0.009 to 0.013. Braket's
    schemas require both fields, and `from_braket` now refuses a file that leaves one out. A v3
    fidelity with no type still reads as randomized benchmarking, the only type v3 defines.
  - `noisevault.sources.quantinuum.from_repository` accepted counts that cannot be right. A shot
    count of 1 in H2-2's single-qubit file gave an error of 4.4e-16 instead of 7.85e-5, and an
    empty first zone gave 0.5. It now refuses a shot count that is not a positive whole number, a
    count outside 0 to the shot count, a sequence length that is not a whole number, a zone with
    no sequence lengths, and an empty or misshapen map.
- **Damaged profiles and vendor replies.** Each case below now gives one error that names the
  damaged input. Before, a command printed an unexpected error and asked for a bug report, or
  printed a message such as `Expecting value: line 1 column 1 (char 0)` that named nothing.
  - A profile file nested deeper than the JSON parser takes. Loading one raises the
    `json.JSONDecodeError` that any damaged file gives, with the depth and the position of the
    deepest bracket, and the commands call the file damaged.
  - Free-form data in `benchmarks`, `extensions` or `provenance.extra` nested more than 256
    levels deep. The format now limits that data to 64 levels, so a profile nested deeper is
    invalid and `nv validate` names the field. A profile nested 65 to 256 levels deep loaded
    before and is now refused too.
  - A reply from IBM's public endpoint or IonQ's API that is not JSON, or is JSON of the wrong
    shape such as `[]` or `"maintenance"`. The error names the URL, and the first wrong field
    when there is one. The hint says to try again later. A damaged IBM device list or
    configuration is the exception. Those replies only add detail, such as the processor name,
    so `nv pull` leaves the detail out and gives no error. Before, some IonQ replies of the wrong
    shape gave a profile from an older record, or one dated 1970, with no error.
  - An IBM time in a unit NoiseVault does not read, such as a T1 in `min`, from `nv pull`,
    `nv.from_qiskit_backend` or `nv.from_calibration_archive`. The error names the time, the
    qubit or gate, and the unit.
  - A vendor value that no profile can hold, such as a readout error of 1.5. `nv pull` and every
    importer raise `nv.SourceDataError`, which names the source and the first such value with
    its qubit or gate. When the source takes a date (`--at`, or `at=` in Python), the hint says
    to pass an earlier one. For an IBM CSV, a Braket file or a Quantinuum spec sheet, it says to
    correct the value in the file. Before, `nv pull` said the profile was not valid and to run
    `nv validate FILE`, though a pull has no file, and the importers raised pydantic's
    `ValidationError`.
  - An IonQ record whose qubit count is missing from the record and from IonQ's backend listing.
    The error names the record, and the hint says to pass an earlier `--at`.
  - An IonQ backend listing with no QPU in it. The error ended with "it lists" and named nothing.
    It now says that the listing names no QPU, and the hint says to try again later.
- **Format 0.1 files that mark a gate or qubit not operational with a string.** The upgrade read
  `operational` by truthiness, so `"false"`, `"no"`, `"off"` and `"0"` left the gate or qubit
  enabled. It now reads the flag as NoiseVault 0.1 did, so these values disable it. A value that
  rule cannot read raises a `ValueError` that names the field.
- **Saving profiles from several threads.** `profile.save(path)` read the umask by setting the
  process umask to 0 and back. A file that another thread created in that moment did not get
  the umask, and two saves at once could leave the umask at 0. `profile.save` no longer changes
  the umask. A new file still gets the mode the umask allows, and a replaced file keeps its mode.
- **A circuit wider than the device on the Qiskit export.** The simulator from `to_qiskit()`
  refused a circuit with more qubits than the device and said to transpile it, but `transpile`
  refuses that circuit too. The hint now names a step that works. For idle extra qubits, it says
  to build the circuit on at most the device's qubit count. For a circuit transpiled for a larger
  backend, it says to transpile the original circuit. For a circuit that uses more qubits than
  the device has, it says to load a profile with enough qubits.
- **Command line.**
  - When IBM had no calibration of a device before the `--at` date,
    `nv pull --source ibm-account` gave advice for retired devices. It now says that IBM returned
    no calibration before that date, and the hint says to pick a later date.
  - `nv show` counted gate loci with no error metric as "without error", a label that read as
    free of error. Exports give those loci the typical native gate's noise. `nv show` now counts
    them as "uncalibrated".
  - `nv diff` warned that the second profile was older than the first even for two different
    devices. For two calibrations of one device on one day, the warning named both by that date,
    as in "ibm_manila@2024-05-27 is older than ibm_manila@2024-05-27". It now warns only about one
    device, and it gives the times when the dates match.
  - For a `.json` or `.json.gz` file that is not valid JSON, the hint said to give a profile file
    with one of those names. It now says that the file is damaged or cut short. A file cut short
    inside a string no longer reads "Unterminated string starting at at line 1".
  - When a vault copy replaced a bundled calibration, a dated ref with no calibration on that day
    listed the replaced calibration twice. It now lists each calibration once.
  - In a narrow terminal, `nv check` wrapped cells such as "exact + 20000 shots" over several
    lines. It now leaves out the tolerance column, then the circuits column, and keeps each cell
    on one line.

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
    Each takes the profile's calibration of that gate. Without one, the export raises
    `MissingCalibrationError` and asks you to decompose it.
- **One rule for fixed-angle gates in every export.** A gate such as `s`, `sx` or `ms` takes
  its own native's noise if the profile has it, else the noise of the rotation it equals.
- **No typical noise for gates that need a decomposition.** A gate that needs several native
  entanglers, such as `swap` or `ccx`, never takes the typical native gate's noise. Neither
  does a gate on more than two qubits. If the profile does not calibrate it, the export raises
  `MissingCalibrationError` with either `unknown_gates` value and asks you to decompose it.
- **Readable reports.** Approximation warnings point at your own line of code.
  `report.summary()` states the usage counts as sentences on one line that starts with `used:`,
  such as "cx took the typical native gate's noise 40 times". It names each `includes` item in
  plain words, such as "single-qubit gate error".
- **Live pulls without an account.** `nv pull` reads IBM's public calibration endpoint, with
  history through `--at`, and IonQ's published characterizations. Pulls through an IBM account
  also work. `nv pull` saves pulled profiles to `~/.noisevault/profiles`.
- **Importers.** Five importers read vendor data: `from_qiskit_backend`, `from_ibm_csv`,
  `from_braket`, `from_cirq_google`, and Quantinuum's dated datasets. `from_braket` maps each
  native to the gate with the same matrix. `from_qiskit_backend` allows a gate only on the
  qubits its `Target` lists, takes the technology from the backend, and labels a fake whose
  snapshot is a model, such as `FakeNighthawk`, as `vendor_model`. An IBM gate error missing
  from a CSV or a pull takes the device median, and `provenance.notes` names those qubits and
  pairs. `from_ibm_csv` reads IBM's CSV formats from 2023 to 2026 and treats a cell that says
  `undefined` as blank. It refuses a file in an older format, a copy in which a spreadsheet
  turned `partner:value` cells into times, and a file with two columns for the same value. Each
  refusal is a one-line error that says why. Importers raise `SourceDataError` for calibration
  data they cannot read. `SourceDataError` is both a `NoiseVaultError` and a `ValueError`. Its
  `hint` holds the next step, if there is one. `from_qiskit_backend` also raises
  `SourceDataError` for a backend with no fixed qubit count.
- **Hypothetical devices** with `Profile.uniform`, and `profile.suggest_layout(n)` to pick a
  well-calibrated chain of qubits that each have every one-qubit native the device has.
- **Command line** `nv` (also `noisevault`): `list`, `show`, `pull`, `diff`, `check`, `cite`,
  `validate`, `doctor` and `schema`.
  - A bare `nv` prints the help, which ends with three commands to start with. A usage
    mistake prints one line with the closest match.
  - A failure prints what went wrong on an `error:` line, and the next step, if there is one,
    on a `hint:` line. In Python, a `NoiseVaultError` keeps that step in `hint`. `str(error)`
    ends with it.
  - `nv show` and `nv diff` label gate errors as average gate infidelity.
  - When you have several calibrations of a device, `nv show` and `nv check` say which one a
    bare id loaded. `nv check` names the calibration it checked on its first line.
  - `nv check` lists missing frameworks with one install command. With no framework installed,
    it prints one error and one install command. When an export refuses the profile,
    `nv check` lists that framework as skipped with the reason and still checks the others.
  - `nv check` counts a circuit that ran with gates removed as reduced. It also samples a
    measurement-only circuit through each framework's own readout. Each row shows the TVD and
    the tolerance of the circuit with the highest ratio of TVD to tolerance. `--json` reports
    that circuit's name, TVD and tolerance under `worst`.
  - `nv list` keeps one device's calibrations together, newest first. It shows the time when
    two calibrations share a date. It never cuts an id or a date.
  - In a narrow terminal, `nv list` leaves out the source column, then the processor column,
    then the license column. Without the license column, the line that counts the profiles
    names the most common license, then each other license with its profiles. When every
    profile has the same license, that line names it and the table has no license column.
  - `nv validate` says what an unknown or a missing key means.
  - `nv` prints commands without backticks, so you can paste them into a shell as they are. No
    output line ends in padding spaces.
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
