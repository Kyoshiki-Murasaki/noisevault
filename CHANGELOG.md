# Changelog

This file lists all notable changes to NoiseVault. Versions follow
[Semantic Versioning](https://semver.org).

## Unreleased

### Added

- **Factors for the error a calibration leaves out.** A profile can set `unmodeled_error`, with
  a factor on its gate error rates, a factor on its readout error rates, or both. Every export
  and the `nv check` reference apply the factors. `nv diff` compares the errors after it applies
  the factors. A factor raises each error channel to that power. As a result, a factor of 1
  keeps the stated error, and no factor takes an error out of its physical range. No factor
  changes T1, T2 or preparation error. An error with no valid power, such as a readout pair no
  better than chance, stays as stated.
  - Each export's report states the factors on its second line, which starts with
    `unmodeled error:` and names each error left as stated. `report.to_dict()` holds the same
    text under `unmodeled_error`.
  - `nv show` prints the calibration as stated, then the factors on an `unmodeled` row.
  - A factor and its interval print with three significant digits, or with more when three
    would print two different values alike, as in `x1.008 (95% interval 1.001 to 1.015)`.
  - The field is part of the fingerprint, and `nv cite` names the factors.
    `profile.uncorrected()` returns the profile without the field, with the calibration's
    fingerprint. `profile.calibration_fingerprint` is that fingerprint, the one a counts file
    names. Profiles without the field keep their fingerprints.
  - A NoiseVault install that does not know the field refuses a profile that sets it, because
    format 1.0 forbids unknown keys.

  See [Unmodeled error](docs/profile-format.md#unmodeled-error).
- **IBM calibration history from Hugging Face.** `nv.from_calibration_archive(path, device,
  at=...)` reads a local copy of the Hugging Face dataset `phanerozoic/qiskit-calibration-drift`.
  The function returns the IBM calibration of `ibm_fez`, `ibm_kingston`, `ibm_marrakesh` or
  `ibm_torino` that was in effect at `at`. A provenance note names the values calibrated more
  than 7 days before `at`, or before the newest calibration when you give no `at`.
  `nv.calibration_archive_devices(path)` gives the range of times each device covers. Install
  `noisevault[hf]` (pyarrow 14.0.1 or later). `nv doctor` lists pyarrow. See
  [IBM calibration archive on Hugging Face](docs/data-sources.md#ibm-calibration-archive-on-hugging-face).
- **Scoring a profile on counts from a device.** `nv compare REF COUNTS` scores a profile on counts
  that a device measured. It fits one factor on every gate error rate and one factor on every
  readout error rate. Each factor has a 95% interval. `nv compare` also tests whether one pair of
  factors explains every circuit. It exits with status 0 for every fit result. `-o FILE` saves the
  profile with the fitted factors in `unmodeled_error`, and `--json` prints the result as JSON.
  Before the fit, `-o` refuses the counts file, the profile file and any path in the vault or among
  the bundled profiles. In Python, `profile.compare(counts)` returns the same result, and
  `result.fitted_profile()` returns the profile with the factors. `result.summary()` returns the
  text that `nv compare` prints. `result.summary_lines()` returns the same lines, each with the
  number of leading characters that `nv compare` prints in bold, for a program that styles the
  output itself.
  - `noisevault.counts` reads and writes counts files. `load_counts(path)` reads one as a
    `MeasuredCounts`. `plan(profile)` returns the circuits to run, with every wait written as a
    `delay`. `simulate(profile, circuits, shots=..., seed=...)` draws counts from a profile.
    `SAMPLER_V2_OPTIONS` maps each Qiskit Runtime `SamplerV2` option that `load_counts` checks,
    such as `twirling.enable_gates`, to the value that `load_counts` requires. The map is for
    code that runs the circuits through `SamplerV2` itself.
  - `load_counts` raises `nv.CountsError` for a file it refuses. The message names the file and
    the field, and `hint` says how to correct the file.
  - When a command that takes a profile gets a counts file, the error says that the file is a
    counts file and not a profile. When the two arguments are in the wrong order, `nv compare`
    names the correct order.
  - `scripts/run_on_ibm.py` runs the `nv compare` circuits on an IBM device through your IBM
    Quantum account. It writes a counts file bound to the calibration in effect when the job ran.

  See [Counts format](docs/counts-format.md),
  [Measure a profile against hardware](docs/recipes.md#measure-a-profile-against-hardware) and
  [How nv compare fits the factors](docs/limitations.md#how-nv-compare-fits-the-factors).

### Changed

- **Messages and help text.** Error messages, warnings, hints and help text now follow one
  writing standard, ASD-STE100 Simplified Technical English. Each concept has one term. Each
  message names qubits in one form: `qubit 3`, `qubits 0-2` for a pair, and
  `qubits 0-1, 1-2 and 2-3` for a list of pairs. Code that matches the text of a message must
  match the new text. Examples:
  - `cz has no calibration on (0, 2) and connectivity does not allow it` is now
    `cz has no calibration on qubits 0-2, and the connectivity does not allow cz there`.
  - `'ibm_fez@2025-13-40': 2025-13-40 is not a calendar date; give one such as 2025-02-26` is now
    `'ibm_fez@2025-13-40': 2025-13-40 is not a calendar date. Give one such as 2025-02-26`.
  - `nv list --help` says `for example trapped_ion` in place of `e.g. trapped_ion`.

### Fixed

- **Revalidating a profile after reading its fingerprint.** `Profile.model_validate(profile)`, a
  pydantic model or `TypeAdapter` with a `Profile` field, and `Profile(**dict(profile))` raised
  `Extra inputs are not permitted` after a read of `fingerprint`, `artifact_hash` or `table`.
  These three calls now accept the profile. With pydantic 2.13, an assignment to `fingerprint`,
  `artifact_hash` or `table` replaced the stored value. The assignment now raises
  `ValidationError`, like any other change to a frozen profile.
- **Loading from a vault whose index disagrees with its files.** `nv.load`, the commands that
  take a ref, and `nv pull` used the vault's index, `.index.json`, and did not check the index
  against the files. An index entry with the wrong date let `ibm_manila@2024-06-04` load a
  calibration from 2024-06-03, and `nv pull` then saved over that calibration. A load could also
  return another calibration when a save replaced a file during the load. NoiseVault now uses the
  index only when its version and digest show that NoiseVault wrote it, and otherwise reads the
  files again. `nv.load` checks the profile that it read against the calibration that the ref
  selected and against the `expect=` pin. If the profile does not match, `nv.load` reads the
  vault again. An `.index.json` that is not a regular file no longer stops `nv list`.
- **Importer input that is damaged or incomplete.** Three importers returned wrong values with
  no error. Each now raises `SourceDataError` that names the file and the field or line.
  - `from_ibm_csv` lost every row after a quote left open, so a file could import with no
    two-qubit records. For a row cut short, `from_ibm_csv` read the missing cells as blank, and a
    blank Operational cell enabled a disabled qubit. `from_ibm_csv` now refuses a file with a quote
    left open or a row cut short. A CSV whose lines end in a bare carriage return failed with a
    `csv` module error. Such a CSV now reads like any other CSV.
  - `from_braket` read a time with no `unit` as seconds, so a T1 of 18.5 us became 18,500,000
    us. It read a standardized v1 or v2 two-qubit fidelity with no `fidelityType` as randomized
    benchmarking. On one IQM pair, the misread changed the error from 0.009 to 0.013. Braket's
    schemas require both fields, and `from_braket` now refuses a file that leaves one out. A v3
    fidelity with no type still reads as randomized benchmarking, the only type v3 defines.
  - `noisevault.sources.quantinuum.from_repository` accepted counts that cannot be right. A shot
    count of 1 in H2-2's single-qubit file gave an error of 4.4e-16 instead of 7.85e-5, and an
    empty first zone gave 0.5. The importer now refuses a shot count that is not a positive whole
    number and a count outside 0 to the shot count. It also refuses a sequence length that is
    not a whole number, a zone with no sequence lengths, and an empty or misshapen map.
- **Gate order of a profile from a Qiskit backend.** `nv.from_qiskit_backend` wrote the gates in
  an order that changed from one Python process to the next. As a result, one calibration could
  give different file bytes. The importer now reads the backend's gates in name order. The
  fingerprint and the bundled files do not change.
- **Vendor values and qubit indices of the wrong type or value.** The Braket, IonQ, IBM and
  Hugging Face importers used some values with no check. Such a value could change with no
  error, go missing with no note, or stop the import with an unexpected Python error. A Braket
  T1 of `true` became 1 s. An archive `qubit_b` of 2.5 became qubit 2. A negative IonQ gate time
  gave no gate time. An IBM gate on the wrong number of qubits stopped `nv pull` with a
  `ValueError`. An IBM `rz` gate on qubit 99 of a 5-qubit device gave no error. A Braket
  `updatedAt` of `false` gave the time of the service refresh. Each importer now checks these
  values where it reads them, also in metadata, headers, timestamps and the gates that it skips.
  A bad value raises `nv.SourceDataError`, which names the source, the field and the value. A
  Braket qubit id that is not a whole number, such as `q1`, raises the error. A value of the
  wrong type in an IonQ reply gives the error for a reply of the wrong shape. An IonQ median
  above 1 is corrupt data, and a provenance note says so. A provenance note also counts the
  archive rows with no property or no `calibrated_time`, which the profile does not use. An
  archive device with no such row, or with no `observed_time`, raises the error. Archive rows
  with no `backend` also raise the error. An archive timestamp with no time zone is UTC, as the
  dataset stores it. Before, such a timestamp stopped an import with `at=` with a `TypeError`.
- **Damaged profiles and vendor replies.** Each case below now gives one error that names the
  damaged input. Before, a command printed an unexpected error and asked for a bug report, or
  printed a message such as `Expecting value: line 1 column 1 (char 0)` that named nothing.
  - A profile file nested deeper than the JSON parser takes. A load of such a file raises the
    `json.JSONDecodeError` that any damaged file gives. The error gives the depth and the
    position of the deepest bracket. The commands call the file damaged.
  - Free-form data in `benchmarks`, `extensions` or `provenance.extra` nested more than 256
    levels deep. The format now limits that data to 64 levels, so a profile nested deeper is
    invalid and `nv validate` names the field. A profile nested 65 to 256 levels deep loaded
    before, and NoiseVault now refuses that profile too.
  - An importer's file that is not UTF-8 text, or a Braket or Quantinuum file that is not JSON.
    `from_braket`, `from_ibm_csv` and the Quantinuum importers raised `UnicodeDecodeError`,
    `json.JSONDecodeError` or, for an integer too long to parse, `ValueError`. These importers
    now raise `nv.SourceDataError` that names the file and what is wrong with it, as in
    `saved.json is not JSON (expecting value at line 1, column 7)` or
    `saved.json is not UTF-8 text (byte 0xb5 on line 2)`.
    `noisevault.sources.quantinuum.from_spec_csv` also refuses a quoted cell that continues on
    the next line, and the error names the line, as `from_ibm_csv` does.
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
    to pass an earlier one. For an IBM CSV, a Braket file or a Quantinuum spec sheet, the hint
    says to correct the value in the file. Before, `nv pull` said that the profile was not valid
    and to run `nv validate FILE`, but a pull has no file. The importers raised pydantic's
    `ValidationError`.
  - An IonQ record whose qubit count is missing from the record and from IonQ's backend listing.
    The error names the record, and the hint says to pass an earlier `--at`.
  - An IonQ backend listing with no `qpu.` backend in it. The error ended with "it lists" and
    named nothing. The error now says that the listing "names no QPU", and the hint says to try
    again later.
- **Format 0.1 files that mark a gate or qubit not operational with a string.** The upgrade read
  `operational` by truthiness, so `"false"`, `"no"`, `"off"` and `"0"` left the gate or qubit
  enabled. The upgrade now reads the flag as NoiseVault 0.1 did, so these values disable the gate
  or qubit. A value that rule cannot read raises a `ValueError` that names the field.
- **Saving profiles from several threads.** `profile.save(path)` read the umask by setting the
  process umask to 0 and back. A file that another thread created in that moment did not get
  the umask, and two saves at once could leave the umask at 0. `profile.save` no longer changes
  the umask. A new file still gets the mode the umask allows, and a replaced file keeps its mode.
- **A circuit wider than the device on the Qiskit export.** The simulator from `to_qiskit()`
  refused a circuit with more qubits than the device and said to transpile it, but `transpile`
  refuses that circuit too. The hint now names a step that works. For idle extra qubits, the hint
  says to build the circuit on at most the device's qubit count. For a circuit transpiled for a
  larger backend, the hint says to transpile the original circuit. For a circuit that uses more
  qubits than the device has, the hint says to load a profile with enough qubits.
- **Command line.**
  - When IBM had no calibration of a device before the `--at` date,
    `nv pull --source ibm-account` gave advice for retired devices. It now says that IBM returned
    no calibration before that date, and the hint says to pick a later date.
  - `nv show` counted gate loci with no error metric as "without error", a label that read as
    free of error. Exports give those loci the typical native gate's noise. `nv show` now counts
    them as "uncalibrated".
  - `nv diff` warned that the second profile was older than the first even for two different
    devices. For two calibrations of one device on one day, the warning named both by that date,
    as in "ibm_manila@2024-05-27 is older than ibm_manila@2024-05-27". It now warns only when both
    profiles are of one device. When the dates match, the warning gives the times.
  - For a `.json` or `.json.gz` file that is not valid JSON, the hint said to give a profile file
    with one of those names. For a damaged gzip file, the hint said to copy or pull the file
    again. Both hints now say that the file is damaged or truncated and to pull or export the
    profile again. A file with another name gets the hint to give a profile file. For a file that
    is not UTF-8 text, the error said "not JSON (not UTF-8 text)". The error now names the first
    byte that is not UTF-8 and its line, as in `bad.json is not UTF-8 text (byte 0xff on line 2)`.
    For a file cut short inside a string, the error no longer reads
    "Unterminated string starting at at line 1".
  - When a vault copy replaced a bundled calibration, the error for a dated ref with no
    calibration on that day listed the replaced calibration twice. The error now lists each
    calibration once.
  - In a narrow terminal, `nv check` wrapped cells such as "exact + 20000 shots" over several
    lines. It now leaves out the tolerance column, then the circuits column, and keeps each cell
    on one line.

## 0.2.0 (2026-10-01)

Version 0.2.0 is a rebuild around one hardware-agnostic file format. Code written for 0.1 needs
changes. The 0.1 profile files still load.

### Added

- **Profile format 1.0.** A profile is one JSON file for one calibration of one device. The format
  covers superconducting, trapped-ion, neutral-atom, spin and photonic qubits. Each error metric has
  its own key (`avg_infidelity`, `process_infidelity`, `depolarizing_param`, `pauli`), so no number
  is guessed. Per-qubit and per-gate records override device-wide defaults. Disabled gates and
  qubits, directed connectivity, provenance and license fields are part of the format. See
  [docs/profile-format.md](docs/profile-format.md).
- **Fingerprints.** Every profile has a SHA-256 fingerprint of its physics.
  `nv.load(ref, expect="nv:...")` fails if the profile changed. `nv cite` prints a citation
  with the full fingerprint, the NoiseVault version and the ref that loads the cited
  calibration. On a mismatch, the error names the calibration the ref loaded. If you have the
  pinned calibration, the hint is the `nv.load` call, with the same pin, that loads it. A dated
  ref with no calibration on that UTC day fails with "no ibm_fez profile calibrated on
  2025-03-01 UTC". For a device that `nv pull` serves, the hint is
  `nv pull ibm_fez --at 2025-03-01T23:59:59Z`. The hint also says to load the ref that
  `nv pull` prints.
- **25 bundled profiles that load offline.** The bundled profiles cover 18 IBM devices from
  qiskit-ibm-runtime and 5 Quantinuum devices from Quantinuum's published benchmark data. They
  also cover Google's Rainbow and Weber from cirq-google. All 25 profiles are Apache-2.0 data,
  and the files are byte-identical on Python 3.11 to 3.14.
- **Four exports.** Each export returns the framework's own object with a `.report`. The report
  states what the export reproduces exactly, approximates or omits. The four exports are:
  - `to_qiskit()` returns an `AerSimulator` with a device `Target`, so `transpile` compiles to
    the device's natives and avoids disabled gates.
  - `to_cirq()` returns a `cirq.NoiseModel`. `GridQubit(r, c)` addresses the qubit at those
    coordinates.
  - `to_pennylane()` returns a `qml.NoiseModel` for `qml.add_noise`, with gradients through the
    noise. A broadcast whose angles need different natives raises an error that names the fix,
    `qml.transforms.broadcast_expand`. Operator arithmetic such as `qml.prod` gets the noise of
    the gates it decomposes into.
  - `to_stim(circuit)` returns a noisy copy of a Stim circuit for QEC-size sampling and
    decoding. `CXSWAP`, `SWAPCX` and `CZSWAP` are the registry gates `cxswap`, `swapcx` and
    `czswap`. Each takes the profile's calibration of that gate. If the profile has no
    calibration of the gate, the export raises `MissingCalibrationError` and asks you to
    decompose the gate.
- **One rule for fixed-angle gates in every export.** A gate such as `s`, `sx` or `ms` takes
  the noise of its own native if the profile has that native. Otherwise, the gate takes the
  noise of the rotation that it equals.
- **No typical noise for gates that need a decomposition.** A gate that needs several native
  entanglers, such as `swap` or `ccx`, never takes the typical native gate's noise. A gate on
  more than two qubits also never takes the typical native gate's noise. If the profile does
  not calibrate such a gate, the export raises `MissingCalibrationError` with either
  `unknown_gates` value and asks you to decompose the gate.
- **Readable reports.** Approximation warnings point at your own line of code.
  `report.summary()` states the usage counts as sentences on one line that starts with `used:`,
  such as "cx took the typical native gate's noise 40 times". It names each `includes` item in
  plain words, such as "single-qubit gate error".
- **Live pulls without an account.** `nv pull` reads IBM's public calibration endpoint, with
  history through `--at`, and IonQ's published characterizations. Pulls through an IBM account
  also work. `nv pull` saves pulled profiles to `~/.noisevault/profiles`.
- **Importers.** Five importers read vendor data. The five importers are `from_qiskit_backend`,
  `from_ibm_csv`, `from_braket`, `from_cirq_google`, and the importer of Quantinuum's dated
  datasets. `from_braket` maps each native to the gate with the same matrix.
  `from_qiskit_backend` allows a gate only on the qubits that its `Target` lists, and takes the
  technology from the backend. `from_qiskit_backend` also labels a fake backend whose data is a
  model, such as `FakeNighthawk`, as `vendor_model`. An IBM gate error missing from a CSV or a
  pull takes the device median, and `provenance.notes` names those qubits and pairs.
  `from_ibm_csv` reads IBM's CSV formats from 2023 to 2026 and treats a cell that says
  `undefined` as blank. `from_ibm_csv` refuses a file in an older format and a file with two
  columns for the same value. It also refuses a copy in which a spreadsheet turned
  `partner:value` cells into times. Each refusal is a one-line error that says why. Importers
  raise `SourceDataError` for calibration data they cannot read. `SourceDataError` is both a
  `NoiseVaultError` and a `ValueError`. Its `hint` holds the next step, if there is one.
  `from_qiskit_backend` also raises `SourceDataError` for a backend with no fixed qubit count.
- **Hypothetical devices.** `Profile.uniform` makes a profile for a hypothetical device.
  `profile.suggest_layout(n)` picks a well-calibrated chain of qubits that each have every
  one-qubit native the device has.
- **Command line.** The command line is `nv`, also named `noisevault`. Its commands are `list`,
  `show`, `pull`, `diff`, `check`, `cite`, `validate`, `doctor` and `schema`.
  - A bare `nv` prints the help, which ends with three commands to start with. A usage
    mistake prints one line with the closest match.
  - A failure prints what went wrong on an `error:` line, and the next step, if there is one,
    on a `hint:` line. In Python, a `NoiseVaultError` keeps that step in `hint`. `str(error)`
    ends with that step.
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
  - NoiseVault gives a warning and skips a damaged or unreadable file in the vault. The other
    profiles still list and load.
- Documentation in [docs/](docs/) and runnable [examples](examples/).

### Changed

- To install from GitHub with the extra for your framework, run
  `pip install "noisevault[qiskit] @ git+https://github.com/Kyoshiki-Murasaki/noisevault"`.
  To try the command line without installing, run
  `uvx --from git+https://github.com/Kyoshiki-Murasaki/noisevault nv show ibm_fez`.
- The core install needs only numpy, pydantic, typer and rich. Qiskit, Cirq, PennyLane and
  Stim are extras, and importing `noisevault` imports none of them.
- Minimum versions are pennylane 0.43.3, stim 1.15, typer 0.27 and rich 13.8. CI installs
  every direct dependency at its declared minimum and runs the tests.
- NoiseVault supports Python 3.11 to 3.14.
- NoiseVault upgrades format 0.1 files in memory and gives a `MigrationWarning`. Save them to
  keep the 1.0 form.

### Removed

- The 0.1 Python API, command line, `snapshots/` folder and experiment scripts.

## 0.1.0 (2026-07-14)

Version 0.1.0 is the first release.

- Schema 0.1 for dated JSON files of IBM device calibrations, with provenance and raw-payload
  hashes.
- Importers for IBM Quantum accounts, Qiskit fake backends and IBM calibration CSV files.
- Exports to Qiskit Aer, Cirq and PennyLane noise models, compared on small benchmark circuits
  with exact density-matrix simulation.
