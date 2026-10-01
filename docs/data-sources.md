# Data sources

NoiseVault gets calibrations three ways:

- **Bundled.** 25 profiles ship inside the package and load offline with `nv.load`. All come
  from openly licensed sources.
- **Pulled.** `nv pull` (or `nv.pull`) fetches a live calibration from IBM or IonQ and saves it
  to your vault, `~/.noisevault/profiles`. Set `NOISEVAULT_HOME` to move the vault.
- **Imported.** A converter reads data you already have: a Qiskit backend, an IBM calibration
  CSV, saved Amazon Braket device properties, or a calibration that cirq-google ships.

`nv list` shows the vault first, then the bundled set. A vault profile with the same id,
calibration time and fingerprint as a bundled one hides it. When one device has several
calibrations on the same date, `nv list` adds the UTC time under the date, and a ref with the
full time, such as `ibm_fez@2025-02-26T20:16:25Z`, loads that one.

## Licensing policy

NoiseVault bundles only data whose license allows redistribution, and records that license in
each profile's `provenance`. Everything else stays on your machine. The rule is in the data:

- `provenance.redistributable` is `"yes"`, `"no"` or `"unknown"`.
- The build script refuses to bundle any profile that is not `"yes"`.
- Pulled and imported profiles from IBM's services, IonQ, Braket and your own files are
  `"no"` or `"unknown"`. They are written only to your vault or to the path you choose.

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

## How each source's numbers are read

Each importer turns its source's conventions into explicit fields, so no conversion guesses:

- **IBM.** Gate errors are average infidelities from randomized benchmarking. An error at the
  physical bound d/(d+1) or above (in practice `gate_error = 1`, IBM's marker for a broken gate)
  becomes `disabled: true`, and so does a qubit or gate marked not operational. `rz` is virtual.
  Device-wide defaults are medians. Two-qubit errors carry `"includes": ["1q_dressing"]`.
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
   validate it with `Profile.from_dict`. Import optional packages inside functions, never at
   module level, so a core install keeps working.
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
   `_PULL_SOURCES` in `src/noisevault/catalog.py`, and raise `SourceUnavailable` with the next
   step for every network or lookup failure.
6. For a bundled source, add the module to `SOURCES` in `scripts/build_catalog.py`, run
   `python scripts/build_catalog.py`, and commit the regenerated profiles and NOTICE.
7. Add tests in `tests/test_sources_<name>.py` that run on saved fixtures, not the network.

[CONTRIBUTING](../CONTRIBUTING.md) covers the test commands.
