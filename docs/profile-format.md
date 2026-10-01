# Profile format 1.0

A profile is one JSON file that records the calibrated noise of one quantum device at one point
in time. It holds the numbers a simulator needs: gate errors and durations, coherence times,
readout and preparation errors, and which qubits and gate pairs exist. It also records where the
numbers came from and under what license.

NoiseVault reads a profile with `nv.load(path_or_ref)` and checks one with `nv validate FILE`.
Every export (`to_qiskit`, `to_cirq`, `to_pennylane`, `to_stim`) starts from the same rules,
described on this page. A file may be gzip-compressed (`.json.gz`).

The JSON Schema in [`schema/profile-1.0.json`](schema/profile-1.0.json) describes the structure of
a saved file. It does not express the cross-field rules below (one metric per spec, records
that match their definitions, qubit indices in range), and it lists only the canonical unit
spellings. `nv validate` checks everything. `nv schema` prints the schema.

## Two examples

A hand-written profile of a hypothetical 8-ion device:

```json
{
  "noisevault": "1.0",
  "device": {"name": "demo-ions", "vendor": "example", "technology": "trapped_ion", "num_qubits": 8},
  "connectivity": "all_to_all",
  "gates": {
    "rz": {"virtual": true},
    "r": {"avg_infidelity": 3e-5, "duration_us": 10},
    "zz": {"avg_infidelity": 1e-3, "duration_us": 200, "includes": ["1q_dressing"]}
  },
  "readout": {"p1_given_0": 0.001, "p0_given_1": 0.003},
  "idle": {"t2_s": 1}
}
```

Every gate of a kind has the same error, so the device-wide definitions in `gates` are the whole
story. `duration_us` and `t2_s` are hand-writing aliases. The loader stores them as
`duration_ns: 10000.0` and `t2_us: 1000000.0`. There is no T1, so gates get pure dephasing and
no amplitude damping. The profile id is `example_demo-ions`.

A hypothetical 5-qubit superconducting chip with per-qubit and per-gate records:

```json
{
  "noisevault": "1.0",
  "device": {"name": "demo5", "vendor": "example", "technology": "superconducting", "num_qubits": 5,
             "calibrated_at": "2026-09-01T08:00:00Z"},
  "connectivity": {"directed": true, "edges": [[1, 0], [1, 2], [3, 2], [3, 4]]},
  "gates": {
    "rz": {"virtual": true},
    "sx": {"avg_infidelity": 2.5e-4, "duration_ns": 32, "method": "rb", "statistic": "median"},
    "x": {"avg_infidelity": 2.5e-4, "duration_ns": 32, "method": "rb", "statistic": "median"},
    "ecr": {"avg_infidelity": 8e-3, "duration_ns": 500, "method": "rb", "statistic": "median"}
  },
  "readout": {"error": 0.02, "duration_ns": 1200},
  "prep": {"error": 0.001},
  "idle": {"t1_us": 150, "t2_us": 110, "t2_kind": "echo"},
  "qubits": [
    {"index": 3, "t1_us": 40, "t2_us": 35, "readout": {"p1_given_0": 0.03, "p0_given_1": 0.06}}
  ],
  "calibrations": [
    {"gate": "ecr", "qubits": [1, 0], "avg_infidelity": 7e-3, "statistic": "individual"},
    {"gate": "ecr", "qubits": [3, 2], "avg_infidelity": 1.2e-2, "duration_ns": 660},
    {"gate": "ecr", "qubits": [3, 4], "disabled": true},
    {"gate": "sx", "qubits": [4], "pauli": [1e-4, 1e-4, 3e-4]}
  ],
  "provenance": {"data_kind": "hypothetical", "source_kind": "hand_written"}
}
```

The definitions in `gates` hold device-wide defaults, here medians. A record in `calibrations`
overrides them on specific qubits. The table shows what each lookup resolves to under the rules
below.

| Lookup | Result | Why |
| --- | --- | --- |
| `ecr` on (1, 0) | 7e-3, 500 ns | The record overrides the metric and keeps the default duration. |
| `ecr` on (1, 2) | 8e-3, 500 ns | No record. The directed edge 1 to 2 allows the default. |
| `ecr` on (0, 1) | unavailable | `ecr` is not symmetric and the edge only goes 1 to 0. |
| `ecr` on (3, 2) | 1.2e-2, 660 ns | The record overrides both fields. |
| `ecr` on (3, 4) | disabled | Using it raises `DisabledGateError`. Qiskit's transpiler avoids it. |
| `sx` on 4 | Pauli channel, average infidelity 3.3e-4 | A `pauli` spec is the whole channel. |
| `sx` on 3 | 2.5e-4, 32 ns | The default. |
| qubit 3 | T1 40 us, T2 35 us, readout (0.03, 0.06) | The qubit record overrides the defaults. |
| qubit 0 | T1 150 us, T2 110 us, readout (0.02, 0.02) | The defaults. |

## Top-level fields

| Field | Required | Content |
| --- | --- | --- |
| `noisevault` | yes | The format version, `"1.0"`. |
| `device` | yes | Name, vendor, technology, qubit count, processor and calibration time. |
| `connectivity` | yes | `"all_to_all"`, or the coupling edges. |
| `gates` | yes | Gate definitions with device-wide defaults, keyed by gate name. |
| `readout` | no | Default readout error of every qubit. |
| `prep` | no | Default preparation (reset) error of every qubit. |
| `idle` | no | Default T1, T2 and extra dephasing of every qubit. |
| `qubits` | no | Per-qubit records that override `readout`, `prep` and `idle`. |
| `calibrations` | no | Per-gate records that override a definition on specific qubits. |
| `effects` | no | Physics the format records but no export models yet. |
| `benchmarks` | no | Free-form benchmark values, such as layer fidelity. |
| `provenance` | no | Source, license, attribution and retrieval details. |
| `extensions` | no | Free-form vendor data that is not part of the physics. |

Unknown keys are errors everywhere except inside `benchmarks`, `extensions` and
`provenance.extra`. Those three hold JSON data only: objects with string keys, arrays, strings,
finite numbers, `true`, `false` and `null`. From Python, a key that is not a string, a set, or
any other object is an error rather than a value converted on save. Values are strict: `"1"`, `3.0` for an integer field, or `1` for a boolean
are errors, not coercions.

## Device

| Field | Content |
| --- | --- |
| `name` | Device name, such as `ibm_fez` or `H1-1`. No `@`, `/` or `\`, and no padding. |
| `vendor` | Vendor, such as `ibm`, `quantinuum` or `google`. Optional. |
| `technology` | One of `superconducting`, `trapped_ion`, `neutral_atom`, `spin`, `photonic`, `other`. |
| `num_qubits` | Number of qubits, at least 1. Qubits are numbered `0` to `num_qubits - 1`. |
| `processor` | Processor family, such as `Heron r2`. Optional. |
| `calibrated_at` | When the calibration was taken, an ISO 8601 time with a timezone. Stored in UTC. Optional. |

### Profile ids and refs

The profile id is `name` when it already starts with `vendor_`, else `vendor_name`, lowercased,
with spaces as `-`: `ibm_fez`, `google_weber`, `quantinuum_h1-1`. With no vendor, the id is the
name. An id may use only `a-z`, `0-9`, `_`, `.` and `-`.

`nv.load` and every CLI command take a ref:

| Ref | Meaning |
| --- | --- |
| `path/to/file.json` | A file: anything with a path separator, a `.json` or `.gz` suffix, or an existing file name. |
| `ibm_fez` | The newest profile with that id. |
| `ibm_fez@2025-02-26` | The profile calibrated on that UTC date. |
| `ibm_fez@2025-02-26T20:16:25Z` | The profile calibrated at that time. |

A ref that matches two profiles raises `AmbiguousRef` and lists both. `expect=` pins a
fingerprint (see [Fingerprint and artifact hash](#fingerprint-and-artifact-hash)).

## Connectivity

`"all_to_all"` allows every pair. Otherwise give the edges:

```json
{"edges": [[0, 1], [1, 2]], "directed": false}
```

Undirected edges allow both operand orders and are stored as `[low, high]`, sorted. Directed
edges (`"directed": true`) allow only the listed order. Self-loops, repeated edges and indices
outside `0..num_qubits-1` are errors.

## Gates

`gates` maps each gate name the device offers to its definition. Use the canonical names of the
gate registry so every framework recognizes the gate:

`id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `rx`, `ry`, `rz`, `p`, `u1`, `u`,
`r`, `cx`, `cy`, `cz`, `ecr`, `swap`, `cxswap`, `swapcx`, `czswap`, `iswap`, `sqrt_iswap`, `rzz`,
`rxx`, `ryy`, `zz`, `ms`, `cswap`, `ccx`, `measure`, `reset`, `delay`.

`zz` is exp(-i pi/4 ZZ), the native entangler of Quantinuum machines. `ms` is IonQ's
Molmer-Sorensen gate with two phases. `r(theta, phi)` is a rotation about an axis in the XY
plane. `cxswap`, `swapcx` and `czswap` are Stim's `CXSWAP`, `SWAPCX` and `CZSWAP`. `cxswap`
applies `cx` and then `swap`, and `swapcx` applies them in the other order. Other names are
allowed (Google's `sycamore`, for example) but must state their arity with `qubits`. The Qiskit
export reports such a gate as omitted. The other exports apply it only when a circuit uses a gate
that they map to that name.

A definition and a calibration record share these fields:

| Field | Content |
| --- | --- |
| `avg_infidelity` | Average gate infidelity, 1 - F_avg. |
| `process_infidelity` | Process (entanglement) infidelity. |
| `depolarizing_param` | The lambda of Qiskit's `depolarizing_error`. |
| `pauli` | Pauli error probabilities: 3 numbers for 1 qubit, 15 for 2 qubits. |
| `duration_ns` | Gate duration in nanoseconds. |
| `virtual` | `true` for a gate done in software, such as a frame change. No error, no duration. |
| `disabled` | `true` when the gate must not be used. |
| `method` | How the error was measured: `rb`, `irb`, `srb`, `xeb`, `gst`, `layer`, `model`, `vendor`. |
| `measured` | `isolated` or `simultaneous`. |
| `statistic` | `individual`, `median` or `mean`. |
| `scope` | `gate` or `cycle`. A cycle error includes the surrounding layer. |
| `includes` | What the number already contains: `1q_dressing` (single-qubit gate error), `leakage`, `spam` (state preparation and measurement error). |
| `stderr` | Standard error of the metric. |
| `assumption` | How an importer read a vendor number, in words. `nv show` prints it. |

A definition also has `qubits` (the arity, required for names outside the registry) and
`symmetric` (whether a calibration on one operand order holds for the other).

### Metric keys name the metric

A spec holds at most one of `avg_infidelity`, `process_infidelity`, `depolarizing_param` and
`pauli`. The key says what the number means, so nothing is guessed. For an n-qubit gate with
d = 2^n, NoiseVault converts with:

- process infidelity = avg_infidelity * (d + 1) / d
- avg_infidelity = depolarizing_param * (d - 1) / d
- the sum of the `pauli` entries is the process infidelity

Bounds are checked: avg_infidelity is at most d / (d + 1), process infidelity at most 1,
depolarizing_param at most d^2 / (d^2 - 1), and `pauli` entries are non-negative with a sum of
at most 1. Two-qubit `pauli` vectors use Stim's order `IX, IY, IZ, XI, XX, ..., ZZ`, and the
first letter acts on `qubits[0]`. [Conventions](conventions.md) derives these relations.

A number is the total error of the operation. When T1, T2 and the duration are known, an export
adds relaxation for the duration and solves for the depolarizing part so the composed channel
has the stated error. A `pauli` spec is the whole channel, and no relaxation is added to it.

The qualifiers (`method` through `assumption`) do not change the conversion. They record what
the number is, so a reader can judge it. An export's report lists, under approximated, each gate
it used whose number is a per-cycle value, has `includes`, is a device median or mean, or
carries an assumption. `nv validate` warns about calibration records with `scope: cycle`.

### Gate states

Every lookup of a gate on specific qubits resolves to one of four states:

| State | When | What an export does |
| --- | --- | --- |
| `ideal` | `virtual: true` | No noise. Z-family gates (`z`, `s`, `t`, `p`, ...) are also ideal when `rz` is virtual, unless the Z-family gate has its own metric. |
| `calibrated` | A metric is present after merging. | Adds the gate's channels. |
| `uncalibrated` | A native with no metric anywhere. | Uses typical noise or raises (see [Conventions](conventions.md#gates-the-profile-does-not-calibrate)). |
| `disabled` | `disabled: true` after merging. | Raises `DisabledGateError`. |

`virtual: true` together with a metric or a positive duration is an error.

## Calibration records

A record in `calibrations` names a defined gate and the qubits it acts on, and carries any of
the shared fields:

```json
{"gate": "cz", "qubits": [2, 3], "avg_infidelity": 0.02}
```

Merge rules:

- Record fields override the definition field by field. A record with only `duration_ns` keeps
  the definition's metric.
- The metric is replaced as a whole. A record with `pauli` drops the definition's
  `avg_infidelity`, so a spec never ends up with two metrics.
- `disabled: true` ends the lookup. The gate is unusable on those qubits.

These are errors: two records for the same gate and qubits, a record for a gate missing from
`gates`, a record whose qubit count differs from the gate's arity, repeated qubits within one
record, and qubit indices outside the device.

### Direction and symmetry

`symmetric` defaults from the registry: `cz`, `rzz`, `rxx`, `ryy`, `zz`, `ms`, `iswap`, `swap`
and `sqrt_iswap` are symmetric, and `cx`, `cy`, `ecr` and unknown names are not. A definition
may override it.

The lookup of gate G on qubits (a, b) takes the first of:

1. A record for G on (a, b).
2. If G is symmetric, a record for G on (b, a). Its `pauli` labels are swapped to match.
3. G's definition, if connectivity allows (a, b). Undirected edges allow both orders, directed
   edges only the listed one, and `all_to_all` allows every pair.
4. Otherwise G is unavailable on (a, b).

## Qubits, readout, preparation and idle

Device-wide defaults:

```json
"readout": {"p1_given_0": 0.008, "p0_given_1": 0.015, "duration_ns": 1560},
"prep": {"error": 1e-4},
"idle": {"t1_us": 150, "t2_us": 110, "t2_kind": "echo"}
```

- `readout` holds either both `p1_given_0` (P(read 1 | prepared 0)) and `p0_given_1`, or one
  symmetric `error`. `duration_ns` is optional.
- `prep.error` is the probability that a reset leaves the qubit in |1>.
- `idle` holds `t1_us`, `t2_us`, `t2_kind` (`echo`, `ramsey` or `cpmg`) and
  `dephasing_rate_per_s`, an extra Z error with probability rate x time.

A record in `qubits` overrides these for one qubit:

| Field | Content |
| --- | --- |
| `index` | The qubit, `0..num_qubits-1`. Each index appears at most once. |
| `t1_us`, `t2_us`, `t2_kind`, `dephasing_rate_per_s` | Override `idle` field by field. |
| `readout` | Replaces the default readout as a whole. |
| `prep` | Replaces the default preparation error as a whole. |
| `label` | The vendor's name for the qubit, such as `q(0,5)`. |
| `coords` | Position on the chip. Cirq maps `GridQubit(r, c)` to the qubit at `[r, c]`. |
| `disabled` | `true` when the qubit must not be used. A layout onto it raises `LayoutError`. |

A missing readout or preparation error stays unknown. Exports apply no noise for it and list it
under "unknown" in the report. It is never silently zero. A T2 above 2 T1 is legal, but exports
clamp it to 2 T1 and report the clamp.

### Units and aliases

Saved files use `duration_ns`, `t1_us` and `t2_us`. A hand-written file may use any of these
spellings, and the loader converts them:

| Field | Accepted spellings |
| --- | --- |
| duration | `duration_ns`, `duration_us`, `duration_ms`, `duration_s` |
| T1 | `t1_ns`, `t1_us`, `t1_ms`, `t1_s` |
| T2 | `t2_ns`, `t2_us`, `t2_ms`, `t2_s` |

Two spellings of one field, a non-finite value, a negative duration and a T1 or T2 of zero or
less are errors.

## Effects

`effects` records physics that the gate-and-readout model does not cover. The list is closed in
format 1.0:

| `type` | Meaning |
| --- | --- |
| `leakage` | Population leaves the qubit subspace. |
| `atom_loss` | A neutral atom is lost from its trap. |
| `erasure` | An error at a known location, such as a lost photon in dual-rail encoding. |
| `crosstalk_measurement` | Measuring one qubit disturbs others. |
| `crosstalk_zz` | An always-on ZZ coupling between qubits. |
| `coherent_overrotation` | A systematic unitary error. |

Each effect names exactly one of `gate` (a defined gate) or `on` (`readout` or `idle`), and may
carry `qubits`, `prob`, `rate_per_s`, `strength_hz`, `angle_rad` and `heralded`.

```json
{"type": "atom_loss", "on": "readout", "prob": 0.005}
```

`allow` states the strictest treatment you accept: `omit` (the default), `approximate` or
`exact`. No export models effects in this release. Every export lists them under "omitted" in
its report, and an effect with `allow` set to `approximate` or `exact` makes the export raise
`UnsupportedEffect`. The format carries effects now so profiles do not need to change when
exports start modeling them.

## Benchmarks and extensions

`benchmarks` holds named benchmark values as free-form JSON, such as
`{"eplg_100": {"value": 3.2e-3, "convention": "avg_infidelity"}}`. Exports do not read it. It is
part of the fingerprint.

`extensions` holds vendor data that is not physics NoiseVault uses, such as Google's fSim error
angles. It is excluded from the fingerprint. Data for out-of-scope technologies goes here (see
[How technologies map](#how-technologies-map)).

## Provenance

| Field | Content |
| --- | --- |
| `data_kind` | `measured`, `vendor_model`, `spec_sheet`, `hypothetical` or `unknown`. |
| `source_kind` | `package_snapshot`, `public_api`, `account_api`, `user_file`, `published_data`, `vendor_sample`, `hand_written`, `derived` or `other`. |
| `source` | The source in words, such as `qiskit-ibm-runtime 0.49.0 FakeFez`. |
| `source_url` | Where the data came from. |
| `license` | The data's license, such as `Apache-2.0`. |
| `attribution` | Who to credit. `citation()` uses it. |
| `redistributable` | `yes`, `no` or `unknown`. Only `yes` can be bundled. |
| `retrieved_at` | When NoiseVault fetched the data. |
| `source_hash` | `sha256:<hex>` of the raw source bytes. |
| `tool` | The NoiseVault version that wrote the profile. |
| `derived_from` | The profile this one was derived from. |
| `notes` | Free-text notes, one per decision an importer made. |
| `extra` | Free-form source details, such as a commit hash. |

## Fingerprint and artifact hash

A profile has two hashes, both SHA-256 over canonical JSON (sorted keys, no whitespace, floats
as Python `repr`):

- The **fingerprint** covers the physics, which is everything except `provenance` and
  `extensions`. Two profiles with the same fingerprint produce the same noise models under one
  NoiseVault version. `nv:` plus the first 12 hex digits is the short form, such as
  `nv:06404cefa54f`.
- The **artifact hash** covers the whole profile, provenance included.

Canonical form makes the fingerprint independent of how a file is written. Records are sorted,
edges are sorted, and a `symmetric` or `qubits` value that only restates the registry default
is dropped. Re-ordering `calibrations` does not change the fingerprint. Editing
`provenance.notes` changes the artifact hash but not the fingerprint.

Pin a fingerprint when you load, and a different profile fails loudly:

```python
import noisevault as nv

fez = nv.load("ibm_fez", expect="nv:06404cefa54f")
print(fez.fingerprint)
```

`expect` takes the 12-digit short form or the full 64-digit hex. A mismatch raises
`FingerprintMismatch`. A paper should cite the full fingerprint; `nv cite` prints it.

`profile.save(path)` writes canonical JSON. A `.gz` suffix writes gzip with a zero timestamp,
so the same profile always produces the same bytes.

## Files in format 0.1

NoiseVault 0.1 files (`"schema_version": "0.1"`) still load. The loader upgrades them in memory
and emits a `MigrationWarning`. Save the profile to keep the 1.0 form.

## How technologies map

The format describes gates on qubits with Markovian errors. Each technology fits it this way:

- **Superconducting.** Natives such as `sx`, `x`, a virtual `rz`, and `cz`, `ecr` or `cx`.
  Coupling edges, directed for `ecr` and `cx` when the device only calibrates one direction.
  Per-qubit T1, T2 and asymmetric readout. Google's `sqrt_iswap` is a registry gate, and
  `sycamore` is defined with `"qubits": 2`.
- **Trapped ion.** `"connectivity": "all_to_all"`, natives such as `r`, a virtual `rz`, and `zz`
  or `ms`. Vendors often publish device-wide medians or means, recorded with `statistic`. T1 is
  usually absent and T2 long. Leakage goes in `effects`.
- **Neutral atom in gate mode.** Natives such as `rz`, `r` and `cz`, with `all_to_all` or the
  edges within the interaction radius. Atom loss goes in `effects` as `atom_loss`.
- **Spin qubits.** Like superconducting devices: coupling edges, per-qubit T1 and T2, and
  `dephasing_rate_per_s` for extra dephasing.
- **Photonic dual-rail.** Each qubit is a pair of modes. Gate errors map to the metrics above.
  Photon loss is an `erasure` effect with `"heralded": true`.

Out of scope, and why:

- **Analog neutral-atom computing** (programs as time-dependent Rydberg Hamiltonians). There is
  no gate set, so there are no per-gate errors to record. Keep the vendor's parameters in
  `extensions` if you want them pinned next to a gate-mode profile.
- **Continuous-variable photonics** (squeezed states, homodyne measurement). The modes are not
  qubits, so qubit channels do not describe them. The same `extensions` rule applies.
