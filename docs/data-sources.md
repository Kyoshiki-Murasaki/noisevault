# Data sources

NoiseVault gets calibrations three ways:

- **Bundled.** 25 profiles ship inside the package and load offline with `nv.load`. All come
  from openly licensed sources.
- **Pulled.** `nv pull` (or `nv.pull`) fetches a live calibration from IBM or IonQ and saves it
  to your vault, `~/.noisevault/profiles`. Set `NOISEVAULT_HOME` to move the vault.
- **Imported.** A converter reads data you already have: a Qiskit backend, an IBM calibration
  CSV, saved Amazon Braket device properties, a calibration that cirq-google ships, or a local
  copy of a published archive of IBM calibrations.

`nv list` shows the vault first, then the bundled set. A vault profile with the same id,
calibration time and fingerprint as a bundled one hides it. When one device has several
calibrations on the same date, `nv list` adds the UTC time under the date, and a ref with the
full time, such as `ibm_fez@2025-02-26T20:16:25Z`, loads that one.

## Licensing policy

NoiseVault bundles only data whose license allows redistribution, and records that license in
each profile's `provenance`. Everything else stays on your machine. The rule is in the data:

- `provenance.redistributable` is `"yes"`, `"no"` or `"unknown"`.
- The build script refuses to bundle any profile that is not `"yes"`.
- Pulled and imported profiles from IBM's services, the IBM calibration archive, IonQ, Braket
  and your own files are `"no"` or `"unknown"`. They are written only to your vault or to the
  path you choose.

Check a profile's terms with `nv show REF`, which prints its source, license and
redistributable flag. Before you publish a pulled profile, for example next to a paper, read
the provider's terms. Publishing its fingerprint (`nv cite REF`) is always possible and lets
others check that they use the same calibration.

The bundled data keeps its upstream license and attribution. [NOTICE](../NOTICE) lists every
bundled file and its source.

## Bundled profiles

Every bundled profile loads offline with `nv.load(ID)`. None needs an account, and each one is
a single calibration with no history.

| Source | Profiles | License | Calibration dates | Stored |
| --- | --- | --- | --- | --- |
| qiskit-ibm-runtime 0.49.0 fake backends | 18 IBM devices: Eagle (`ibm_brisbane`, ...), Heron (`ibm_fez`, ...), Nighthawk (`ibm_berlin`, `ibm_miami`) and the 5-qubit Falcon `ibm_manila` | Apache-2.0 | 2024-05-27 to 2026-04-17 | Per-qubit T1, T2 and readout, per-gate errors and durations of every native. |
| Quantinuum hardware-specifications repository, pinned commit | `quantinuum_h1-1`, `h1-2`, `h2-1`, `h2-2`, `reimei` | Apache-2.0 | 2023-08-21 to 2025-08-28 | Device-wide average infidelity of each native gate, device-wide readout error, and leakage rates as `leakage` effects. No T1 or T2. |
| cirq-google 1.6.1 | `google_rainbow`, `google_weber` | Apache-2.0 | 2021-11-16 (rainbow) and 2021-11-03 (weber) | Per-qubit T1, T2 and readout, per-qubit and per-pair gate errors and durations, and fSim coherent errors as `coherent_overrotation` effects. |

The bundled files total 0.24 MB. Each bundled profile is a snapshot of one calibration. The
devices have been recalibrated many times since, so use a pull for current numbers.

## Pulled sources

`nv pull DEVICE [--at TIME] [--source NAME] [-o FILE]` picks the source from the device id.
Pulling a calibration already in the vault writes nothing.

### IBM public endpoint (`--source ibm`, the default for `ibm_...`)

| | |
| --- | --- |
| Access | `https://quantum.cloud.ibm.com/api/v1/public/backends/<name>/properties` |
| Account | None |
| History | Yes: `--at 2026-06-01` returns the newest calibration before that time. |
| License | Not an open license. Profiles are `redistributable: "unknown"` and stay local. |
| Stored | Per-qubit T1, T2 and readout, per-gate errors and durations of every native, disabled gates and qubits, the raw response's SHA-256. |

The endpoint is not documented by IBM and may change or disappear. If it fails, the error says
so and suggests the account source.

### IBM account (`--source ibm-account`)

| | |
| --- | --- |
| Access | `qiskit-ibm-runtime`, `backend.properties(datetime=...)`. Install `noisevault[ibm]`. |
| Account | Yes: the one saved with `QiskitRuntimeService.save_account`, or `IBM_QUANTUM_TOKEN` (with `IBM_QUANTUM_INSTANCE` when the key sees several instances). |
| History | Yes, through `--at`. |
| License | Your account's terms. `redistributable: "unknown"`. |
| Stored | The same fields as the public endpoint. |

### IonQ characterizations (`--source ionq`, the default for `ionq...`)

| | |
| --- | --- |
| Access | IonQ API v0.4, `https://api.ionq.co/v0.4/backends/<backend>/characterizations` |
| Account | None |
| History | Yes, through `--at`. Records without their own 1Q and 2Q fidelities are skipped for the next older one, and the reason goes in the provenance notes. |
| License | IonQ EULA, not an open license. `redistributable: "no"`. |
| Stored | Device-wide medians only: 1Q, 2Q and SPAM fidelity, gate and readout times, T1 and T2. Every qubit and pair gets the same values. |

Device ids are `ionq_forte-1`, `ionq_aria-1` and so on. With `--source ionq`, `forte-1` and
`qpu.forte-1` also work.
IonQ does not say which fidelity metric it reports. The importer reads it as average gate
fidelity and writes that into each gate's `assumption`, which `nv show` prints.

```bash
nv pull ibm_fez
nv pull ibm_fez --at 2026-06-01
nv pull ionq_forte-1
nv list
```

## Imported sources

| Converter | Reads | Account | History | License of the result | Stored |
| --- | --- | --- | --- | --- | --- |
| `nv.from_qiskit_backend(backend)` | Any Qiskit `BackendV2`, such as a fake backend or one from your account. | Only for account backends | No. It reads the backend's current `Target`. For an older IBM calibration, use `nv pull --at`. | Apache-2.0 for qiskit-ibm-runtime fake backends, else `unknown` | Per-qubit T1, T2 and readout, per-gate errors and durations of every native in the `Target`. |
| `nv.from_ibm_csv(path, device=..., calibrated_at=...)` | The calibration CSV you download from the IBM Quantum platform. Formats from 2023 to 2026. | To download it | One file is one calibration. You give its time as `calibrated_at`. | `unknown` | Per-qubit T1, T2, readout errors and readout length, per-gate errors and durations, and qubits marked not operational. |
| `nv.from_braket(path_or_dict, device=...)` | Braket device properties you saved with `AwsDevice(arn).properties.json()`, or their `standardized` part (v1, v2, v3). | AWS, to save it | No. One file is one snapshot. | AWS Customer Agreement, `no` | v1 and v2: per-qubit T1, T2 and fidelities, and per-pair gate fidelities. v3: device-level values. |
| `nv.from_calibration_archive(path, device, at=...)` | A local copy of the Hugging Face dataset `phanerozoic/qiskit-calibration-drift`, which records each new IBM calibration of `ibm_fez`, `ibm_kingston`, `ibm_marrakesh` and `ibm_torino`. Install `noisevault[hf]`. | None | Yes. Each property takes its newest calibration at or before `at`. | `unknown`. The dataset is CC-BY-4.0, and the numbers are IBM's. | The same fields as a pull. Early dates have fewer gates, and `ibm_torino` has no durations. |
| `nv.from_cirq_google(processor_id)` | The calibration cirq-google ships for `rainbow`, `weber` or `willow_pink`. Install `noisevault[google]` (cirq-google 1.6 or later). | None | No. cirq-google ships one calibration per processor. | Apache-2.0 | The same fields as the bundled Google profiles. |
| `noisevault.sources.quantinuum.from_repository(machine, date)` | Any of the 15 datasets in Quantinuum's repository at the pinned commit, such as `("H1-1", "2023_01_20")`. Downloads from GitHub. | None | Yes. Each dataset is one date, and `date=None` gives the machine's newest. | Apache-2.0 | The same fields as the bundled Quantinuum profiles. |
| `nv.Profile.uniform(...)`, or a file you write | A hypothetical device. | None | No | Yours | The values you give. |

```python
from qiskit_ibm_runtime.fake_provider import FakeTorino

import noisevault as nv

torino = nv.from_qiskit_backend(FakeTorino())
print(torino.id, torino.short_fingerprint, torino.provenance.license)
```

`willow_pink` is importable but not bundled. Its two-qubit values sit under cirq-google's
"per cycle" key, yet they reproduce Google's published per-gate CZ error for that chip. Until
that is resolved, the importer reads them as per-gate errors and says so in the gate's
`assumption`.

Some qiskit-ibm-runtime fake backends are models, not snapshots of a device: `FakeNighthawk`,
whose package says its error values are not typical of Nighthawk, and `FakeFractionalBackend`,
modeled on `FakeLima`. Their snapshots name themselves (`fake_nighthawk`) instead of a device,
so their profiles get `data_kind` `vendor_model` and a note in `provenance.notes` that quotes
the package. None of them is bundled.

Imported profiles are not saved anywhere until you call `profile.save(path)`.

### IBM calibration archive on Hugging Face

The dataset [phanerozoic/qiskit-calibration-drift](https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift)
records IBM calibrations. Its poller reads IBM's backend properties every 30 minutes and adds a
row for each new calibration of a property. The dataset covers `ibm_fez`, `ibm_marrakesh` and
`ibm_torino` from 31 January 2026, and `ibm_kingston` from 16 March 2026. NoiseVault reads a
local copy of its one parquet file and never downloads it for you.

To get the file, install the `hf` extra and the Hugging Face command line, then download the
data folder. The file is about 66 MB.

```bash
pip install "noisevault[hf] @ git+https://github.com/Kyoshiki-Murasaki/noisevault" huggingface_hub
hf download phanerozoic/qiskit-calibration-drift --repo-type dataset --include "data/*"
```

`hf download` prints the snapshot folder it wrote, which ends in `snapshots/<revision>`. The
file is `data/train-00000-of-00001.parquet` inside it. A `git clone` without Git LFS gives a
small pointer file in its place, and the importer rejects that file.

<!-- not-run: needs the 66 MB file that hf download writes -->
```python
import noisevault as nv

path = "<snapshot folder>/data/train-00000-of-00001.parquet"
for device, span in nv.calibration_archive_devices(path).items():
    print(device, span.first, span.last)
fez = nv.from_calibration_archive(path, "ibm_fez", at="2026-06-01")
```

`calibration_archive_devices` returns each device with two times. `first` is when the archive
first recorded the device, and `last` is its newest calibration. For each property, such as the
T1 of one qubit or the `cz` error of one pair, `from_calibration_archive` takes the newest
calibration at or before `at`. Without `at`, every property takes its newest calibration. An
`at` before `first` raises `SourceDataError`, because the archive holds only the calibrations
that were still current at its first poll. Each call reads the file again, which takes 2 to 3
seconds for one device.

The dataset's license is CC-BY-4.0, except for its sunspot column `SN`, which is CC-BY-NC-4.0.
The importer never reads `SN`. Each profile credits the dataset and IBM Quantum in `provenance`:

- `license` is `CC-BY-4.0`.
- `attribution` is `IBM Quantum, via phanerozoic/qiskit-calibration-drift`.
- `source` names the dataset and says that NoiseVault converted it. `source_url` links to the
  dataset.
- `source_hash` is the file's SHA-256, which equals the Hub's LFS object id, so it identifies the
  revision. When the path contains `snapshots/<revision>`, `extra.revision` records the revision
  and `source_url` links to that revision's file.
- `redistributable` is `unknown`, as for `nv pull`, because the numbers are IBM's.

The archive has these limits:

- Until 8 May 2026, the poller wrote rows with no unit, and those rows give T1 and T2 in
  seconds. The importer converts them to microseconds. They hold only T1, T2, the readout
  errors and the `sx` and `cz` errors. Rows from a backfill of IBM snapshots, calibrated from
  late January 2026, add the durations and most other gates.
- If the device's newest calibration has a gate that the time you ask for lacks, the profile
  leaves the gate out and `provenance.notes` names it. For example, `ibm_fez` at its `first`
  time has no `xslow`.
- IBM retired `ibm_torino` in April 2026, and the archive has only its rows with no unit.
  Its profiles have the `sx` and `cz` errors and no durations.
- The archive records a calibration when it first sees it. It never records that IBM stopped
  listing a property, so a profile keeps such a property at its last value. At 2026-09-01,
  `ibm_fez` has `xslow` errors from 29 May 2026 and a T2 on qubit 72 from 21 October 2025,
  which IBM's own snapshot for that time leaves out. `nv pull ibm_fez --at 2026-09-01` and the
  archive agree on all 1,132 gate records that both have. `provenance.notes` names every value
  calibrated more than 7 days before `at`, or before the newest calibration when you give no
  `at`, and the date of the oldest.
- A qubit whose `prob_meas1_prep0` or `prob_meas0_prep1` is 1 is disabled, and
  `provenance.notes` names it. A gate error of 1 disables that gate, as in a pull.

## How each source's numbers are read

Each importer turns its source's conventions into explicit fields, so no conversion guesses:

- **IBM.** Gate errors are average infidelities from randomized benchmarking. An error at the
  physical bound d/(d+1) or above (in practice `gate_error = 1`, IBM's marker for a broken gate)
  becomes `disabled: true`, and so does a qubit or gate marked not operational. `rz` is virtual.
  Device-wide defaults are medians over working qubits and gates, so a qubit marked not
  operational and every gate on it stay out of them. `from_qiskit_backend` disables a gate on every qubit or pair
  that the backend's `Target` does not list for it. In a CSV or a pull, a qubit or pair with no
  published error for a gate takes that gate's median, and `provenance.notes` names it.
  Two-qubit errors carry `"includes": ["1q_dressing"]`.
- **Quantinuum.** The importer repeats the analysis code in Quantinuum's repository
  (`qtm_spec`): randomized-benchmarking decays pooled over gate zones, converted to the average
  infidelity per native gate, with leakage added as `qtm_spec` reports it. The emulator
  parameters (`p1`, `p2`) are a different model and are not used. Leakage rates are kept as
  `leakage` effects.
- **Google.** Single-qubit RB Pauli errors are stored as `process_infidelity`. Two-qubit XEB
  errors per cycle have the single-qubit part removed, as Google does to infer per-gate errors.
  The coherent part of each fSim error is recorded as a `coherent_overrotation` effect.
- **IonQ.** Medians with `statistic: "median"` and the metric reading in `assumption`. SPAM
  fidelity becomes a symmetric readout error, and preparation stays unknown.

## Add a source

A source is one module in `src/noisevault/sources/`. To add one:

1. Write `src/noisevault/sources/<name>.py`. Convert the source's data into a profile dict and
   validate it with `Profile.from_dict`. Raise `SourceDataError` for data the module cannot
   read. Import optional packages inside functions, never at module level, so a core install
   keeps working.
2. Fill `provenance`: `data_kind`, `source_kind`, `source`, `source_url`, `license`,
   `attribution`, `redistributable`, `retrieved_at` and the `source_hash` of the raw bytes.
   Set `redistributable: "yes"` only when the license allows redistribution.
3. Make every interpretation explicit: the metric key that matches what the vendor measured,
   `method`, `statistic` and `includes`, and an `assumption` where the vendor leaves the meaning
   open. A vendor's sentinel values become `disabled`. Never store a per-cycle error as a
   per-gate error without converting it.
4. Expose `bundled_profiles() -> list[Profile]`. Return `[]` unless the data is openly
   licensed.
5. For a live source, expose `pull(device, *, at=None) -> Profile`, register it in
   `_PULL_SOURCES` in `src/noisevault/catalog.py`, and raise `SourceUnavailable` for every
   network or lookup failure.
6. Pass an error's next step as `hint=`, not in the message. `nv` prints the hint on its own
   `hint:` line.
7. For a bundled source, add the module to `SOURCES` in `scripts/build_catalog.py`, run
   `python scripts/build_catalog.py`, and commit the regenerated profiles and NOTICE.
8. Add tests in `tests/test_sources_<name>.py` that run on saved fixtures, not the network.

[CONTRIBUTING](../CONTRIBUTING.md) covers the test commands.
