# Counts format 1.0

A counts file records what a device measured when it ran a set of circuits. It binds the run to
the profile the circuits were planned from, to the physical qubits they ran on, and to the exact
ops the device executed. `nv compare` scores a profile on a counts file.

NoiseVault reads a counts file with `load_counts(path)` from `noisevault.counts`. A file may be
gzip-compressed (`.json.gz`). `plan(profile)` gives the circuits to run, and
`simulate(profile, circuits, shots=..., seed=...)` draws counts for them from a profile.

## An example

A hypothetical run of two circuits on three qubits of ibm_kingston. The counts are made up to
show the format.

```json
{
  "nv_counts": "1.0",
  "source": "hardware",
  "profile": {"id": "ibm_kingston",
              "fingerprint": "609c845ed93458cb1f3999275b0925f6c2731a5fadb581c6d2adc6f2c81f13b4"},
  "backend": "ibm_kingston",
  "run_at": "2026-04-16T09:30:02Z",
  "bit_order": "qiskit",
  "execution": {
    "client": "qiskit-ibm-runtime 0.49.0 SamplerV2, qiskit 2.5.2",
    "transpiled": false, "gate_twirling": false, "measure_twirling": false,
    "dynamical_decoupling": false, "init_qubits": true,
    "job_ids": ["example-job-1"],
    "options": {"twirling": {"enable_gates": false, "enable_measure": false},
                "dynamical_decoupling": {"enable": false},
                "execution": {"init_qubits": true, "meas_type": "classified"}}
  },
  "circuits": [
    {"name": "ghz_chain", "qubits": [148, 149, 150],
     "ops": [["sx", [0], []], ["sx", [1], []], ["sx", [2], []], ["cz", [0, 1], []],
             ["delay", [2], [68.0]], ["cz", [1, 2], []], ["sx", [0], []], ["sx", [1], []],
             ["sx", [2], []], ["delay", [0], [68.0]]],
     "shots": 4000,
     "counts": {"000": 978, "100": 19, "010": 1028, "110": 15, "001": 11, "101": 943,
                "011": 8, "111": 998}},
    {"name": "readout", "qubits": [148, 149, 150], "ops": [], "shots": 4000,
     "counts": {"000": 3923, "100": 34, "010": 39, "001": 4}}
  ]
}
```

Circuit qubit i of a circuit is `qubits[i]`, so `ghz_chain` runs on qubits 148, 149 and 150. Ops
name circuit qubits. The `cz` gate takes 68 ns on these qubits, so qubit 150 waits 68 ns for the
first `cz`. Qubit 148 idles 68 ns while the second `cz` runs. Each wait is a `delay` op. After the
last op, the device measures every circuit qubit, qubit i into classical bit i. With
`"bit_order": "qiskit"`, the rightmost character of a counts key is classical bit 0, so `"100"`
means that qubit 150 read 1.

## Top-level fields

Every field is required.

| Field | Content |
| --- | --- |
| `nv_counts` | The format version, `"1.0"`. |
| `source` | `hardware`, or `simulated` for counts drawn from a model. |
| `profile` | The `id` and the calibration `fingerprint` of the profile the circuits were planned from. |
| `backend` | The device that ran the circuits, as the profile's `device.name` spells it. |
| `run_at` | When the device started the first circuit, an ISO 8601 time with a timezone. Stored in UTC. |
| `bit_order` | How the counts keys order the classical bits, `qiskit` or `clbit0_left`. |
| `execution` | How the device ran the circuits. |
| `circuits` | The circuits, at least one, each with its ops and counts. |

Unknown keys are errors everywhere except inside `execution.options`. Values are strict. A count
of `"4000"` or `4000.0`, or a flag of `1`, is an error rather than a coercion.

`profile.fingerprint` is the fingerprint of the planning profile without `unmodeled_error`, as 64
hex digits. It equals `profile.uncorrected().fingerprint` in Python. `nv compare` scores the counts
only against a profile with this calibration fingerprint and this `backend`.

## Execution

Format 1.0 admits only a run of exactly the listed ops, with each shot starting from a fresh
ground state. Each flag must have the value that starts its row, and `load_counts` refuses a file
with the other value.

| Field | Content |
| --- | --- |
| `client` | The software that ran the circuits, such as `"qiskit-ibm-runtime 0.49.0 SamplerV2, qiskit 2.5.2"`. |
| `transpiled` | `false`. A transpiled run may have executed other ops than the listed ones. |
| `gate_twirling` | `false`. Gate twirling adds random Pauli gates that the ops do not list. |
| `measure_twirling` | `false`. Measurement twirling flips qubits at random before measuring them, which changes readout error. |
| `dynamical_decoupling` | `false`. Dynamical decoupling adds pulses on idle qubits that the ops do not list. |
| `init_qubits` | `true`. Without it, a shot may start where the previous one ended, not from 0. |
| `job_ids` | The ids of the jobs that ran the circuits. Optional, default `[]`. |
| `options` | The options the run was submitted with, as a JSON object nested at most 64 levels deep. `options` itself is level 1. Optional, default `{}`. |

When `options` holds one of these Qiskit Runtime `SamplerV2` options, the option must have the
value shown, or `load_counts` refuses the file:

| Option | Required value |
| --- | --- |
| `twirling.enable_gates` | `false` |
| `twirling.enable_measure` | `false` |
| `dynamical_decoupling.enable` | `false` |
| `execution.init_qubits` | `true` |
| `execution.meas_type` | `"classified"` |

These are the `SamplerV2` defaults in qiskit-ibm-runtime 0.40 and 0.49, so a run with the default
options passes.

A job id links to the IBM Quantum account that ran the job, so share a counts file only where you
would share its job ids. NoiseVault reads counts files from your disk and never uploads them. A
profile fitted with `nv compare` stores the SHA-256 of the counts file, not the counts or the job
ids.

## Circuits

| Field | Content |
| --- | --- |
| `name` | The circuit's name. Names are unique within a file. |
| `qubits` | The physical qubits, 1 to 10 and distinct. Circuit qubit i is `qubits[i]`. |
| `ops` | The ops the device ran, in order, on circuit qubits. Can be empty. |
| `shots` | The number of shots, at least 1. |
| `counts` | The number of shots that gave each outcome. The counts add up to `shots`. |

The device measures every circuit qubit once, after the last op, into the classical bit with the
same index. A file holds at most 4096 outcomes over all its circuits, where a circuit on n qubits
has 2^n.

### Ops

An op is `[name, [circuit qubits], [parameters]]`, as `nv check --json` writes it, such as
`["rz", [0], [1.5708]]`. The name is `delay`, or a gate in the
[gate registry](profile-format.md#gates) that has a unitary. A gate acts on as many distinct
qubits as the registry says. Its parameters are the registry's, as finite numbers in radians.

`["delay", [q], [ns]]` idles circuit qubit q for ns nanoseconds, 0 or more. A delay idles one
qubit. `measure` and `reset` are not ops.

### Counts keys

A key has one character, `0` or `1`, per circuit qubit. `bit_order` says which end holds
classical bit 0:

| `bit_order` | Classical bit 0 | Key for "circuit qubit 0 read 1, qubit 1 read 0" |
| --- | --- | --- |
| `qiskit` | The rightmost character, as Qiskit prints counts. | `"01"` |
| `clbit0_left` | The leftmost character. | `"10"` |

`load_counts` stores keys as `clbit0_left`, sorted, with zero counts dropped. `circuit.vector()`
gives the counts as an array in the order of the reference simulator's probabilities, with
circuit qubit 0 as the most significant bit.

## Planned circuits

`plan(profile)` returns the circuits `nv check` runs, on the qubits `nv check` picks.
`plan(profile, layout)` uses the qubits of `layout` instead, as `profile.check(layout=...)` does.
`plan` schedules each gate as soon as its qubits are free, using the gate durations the profile
states, and writes every wait as a `delay`. It also pads each qubit with a delay to the end of the
circuit. No qubit then has a gap that the ops do not fill, so any scheduling policy on the device
gives the same timeline. A gate with no stated duration, such as a virtual `rz`, takes 0 ns. A
profile with `unmodeled_error` plans the same circuits as its calibration.

```python
import noisevault as nv
from noisevault.counts import load_counts, plan, simulate

kingston = nv.load("ibm_kingston@2026-04-15")
circuits = plan(kingston)
counts = simulate(kingston, circuits, shots=4000, seed=1)
counts.save("kingston.counts.json")

print([c.name for c in circuits])
print(load_counts("kingston.counts.json") == counts)
# ['ghz_chain', 'mirror', 'single_qubit', 'readout']
# True
```

`simulate` draws the counts from the reference simulator, with the profile's unmodeled-error
factors applied. It binds the counts to the profile's calibration fingerprint and sets `source` to
`simulated`. `run_at` defaults to the current UTC time. The same `seed` and `run_at` give the same
file.

## SHA-256 and canonical form

`counts.sha256` is `sha256:` plus the SHA-256 of the canonical JSON (sorted keys, no whitespace)
of the loaded file. The loaded file has its keys in one order, with zero counts dropped, so the
same run has one SHA-256 whether its file says `qiskit` or `clbit0_left`. `counts.save(path)`
writes readable JSON of the loaded form, so the same run always gives the same bytes.

## Errors

`load_counts` raises `nv.CountsError`, a `ValueError`, for a file it refuses. The message is one
line that starts with the path and names the field, such as
`run.counts.json: circuits[0]: counts sum to 3999, but shots is 4000`. `hint` says how to fix it.
A file that cannot be read raises `OSError`.
