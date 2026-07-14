# Autonomous run log

## Run scope and contract

Primary autonomous ownership ran on 2026-07-14 in
`/Users/nidhigoyal/Documents/noisevault-pilot`. The user-supplied task and
`COWORK_TASK.md` were treated as the acceptance contract. The referenced
`CHATGPT_WORK_TASK.md` and `PLANS.md` did not exist in the supplied packet;
`PLANS.md` was created as the living record.

The untouched 77-file packet was preserved in Git commit `ce262b4` on branch
`pilot/noisevault` before any implementation changes.

## Environment and safety gates

- Operating system: macOS 26.4.1 (build 25E253), Apple silicon.
- System default Python: 3.14.6, outside the declared 3.11–3.13 range.
- Pilot runtime: Python 3.12.13, automatically selected by the repaired bootstrap.
- Core versions: NoiseVault 0.1.0, NumPy 2.5.1, Pydantic 2.13.4,
  Qiskit 2.5.0, Qiskit Aer 0.17.2, Qiskit IBM Runtime 0.47.0,
  Cirq Core 1.6.1, PennyLane 0.45.1, SciPy 1.18.0,
  Matplotlib 3.11.0, pytest 9.1.1, and Ruff 0.15.21.
- IBM token environment variables were checked by name only and were absent.
- The final automatic metadata attempt also found no saved usable IBM account.
- `NOISEVAULT_ALLOW_HARDWARE` was absent.
- Hardware/QPU jobs submitted: none.
- Remote at baseline and final validation: none configured.

Dependency installation filled the filesystem with package caches. Only pip and
uv caches were cleared; repository evidence and environments were preserved.

## Commands and outcomes

1. `bash scripts/bootstrap.sh` against the untouched packet — exit 1 at the
   Python-version gate because `python3` resolved to 3.14.6.
2. `PYTHON_BIN=python3.12 bash scripts/bootstrap.sh` — exit 0; created a fresh
   `.venv`, installed all pilot/dev dependencies, reported all three frameworks
   available, passed the baseline 15 tests and Ruff, and completed the baseline
   automatic pilot.
3. Repaired offline smoke:
   `python scripts/run_pilot.py --mode offline --profile smoke` — exit 0;
   exported-adapter execution succeeded for 2- and 3-qubit circuits.
4. Repeated focused and complete validation during repair — final suite grew to
   27 tests, all passing; Ruff remained clean.
5. Final exact publication workflow: `bash scripts/bootstrap.sh` with no override
   — exit 0. It automatically selected Python 3.12.13, reinstalled the editable
   package, passed 27 tests in 4.82 seconds, passed Ruff, attempted metadata-only
   IBM authentication, and completed the full automatic profile.
6. `python scripts/recompute_metrics.py` — exit 0 / `PASS`; independently
   recomputed Experiments A–D using only saved probability vectors and no
   NoiseVault metric functions. All 144 vectors were finite, nonnegative,
   correctly sized, and normalized with maximum mass error
   `3.3306690738754696e-16`.
7. `python -m build` — built `noisevault-0.1.0.tar.gz` and
   `noisevault-0.1.0-py3-none-any.whl` successfully in isolated environments.
8. `python -m twine check dist/*` — both artifacts passed.
9. Every final PNG was opened at rendered resolution and checked against the
   JSON and source snapshots.

## Material methodology and implementation changes

### Combined error parameterization

Defect: the baseline gave depolarization the full archived average gate error and
then composed thermal relaxation, systematically exceeding the source error.
For FakeManila CX `[0,1]`, the archived/native average infidelity was about
`0.0088277121`, while the baseline reconstruction was about `0.0121364239`.

Fix: residual depolarization is now solved after calculating the relaxation
fidelity, matching Qiskit Aer's documented backend-noise convention. If the
reported error is below the relaxation floor, no depolarization is added. The
final Experiment A maximum calibration-target delta is
`8.326672684688674e-17`. `docs/METHODOLOGY.md` records this material change.

### Public converter execution and framework semantics

Defect: baseline Experiment B used three independent engines but bypassed the
exported converter objects. This hid two adapter bugs: Aer's row-oriented
readout API received the internal column-oriented matrix, and PennyLane's
order-insensitive wire predicate inserted both CNOT directions.

Fix: Experiment B now executes the actual Qiskit Aer `NoiseModel`, Cirq
`NoiseModel`, and PennyLane `NoiseModel` via each public adapter. Aer readout is
transposed only at its API boundary; PennyLane uses order-sensitive predicates.
Canonical multi-qubit Kraus matrices are explicitly reordered at Qiskit's
little-endian local-operator boundary. Regression coverage includes asymmetric
readout and a non-symmetric `X⊗I` two-qubit Kraus test.

### Qarg alignment, topology, and calibration semantics

Defect: baseline selection returned a sorted connected set, which did not ensure
that consecutive logical CNOTs had ordered calibrated physical edges. Missing
two-qubit gates silently received a relaxation-only default, reverse-direction
calibrations could be borrowed, and lower-error aliases could override an exact
operation calibration.

Fix: all pilot subsets are directed calibrated paths; size comparisons use
nested prefixes. Experiment D selects a path present in both snapshots.
Unsupported directed entanglers are rejected. Exact operation names precede
documented fallbacks, and qarg reversal is never implicit. Experiment A uses
exact direction-aligned prefix calibrations.

### Partial calibration, ingestion, and validation

- Partial `T1`/`T2` records now retain the available amplitude-damping or
  dephasing component.
- Missing CSV values remain null instead of becoming fabricated zero error;
  undated CSV imports cannot qualify as temporal evidence.
- IBM faulty-gate metadata is imported when available.
- Snapshot validation rejects non-finite values, naive timestamps, and incomplete
  qubit tables. Invalid archive entries can no longer disappear silently.
- Probability normalization rejects non-finite and materially negative inputs.
- Current and legacy/V2 fake-provider classes are discovered, and identical fake
  imports no longer rewrite immutable snapshots.

### Result reproducibility and reporting

- Experiment D now saves both dated probability vectors for every framework,
  making its TVD independently recomputable.
- Verified status requires complete A–D evidence and all expected circuits;
  threshold misses and open gaps return nonzero from the CLI/script.
- Machine-readable results include explicit authentication, hardware, snapshot,
  provenance, expected-count, and threshold fields.
- Reports state exact nonzero scientific-notation TVDs and explicit pass/fail.
- The heatmap uses the fixed `0–0.02` acceptance scale. The GHZ figure identifies
  the packaged calibration and nested path. The drift figure states
  `DETERMINISTIC SYNTHETIC FIXTURES`, both providers/timestamps, and the TVD.

## Data acquisition and provenance

No live IBM snapshot was acquired. Final archive inventory:

| Provider | Backend | Source timestamp | Raw hash | Path |
|---|---|---|---|---|
| `demo` | `demo_linear_5` | `2024-01-15T09:00:00Z` | `sha256:98d477858773707ad4dd3b1f9014dd14f9f0ed0304f4f3b16f615e8d5c9335e3` | `snapshots/demo/demo_linear_5_2024-01-15.json` |
| `demo` | `demo_linear_5` | `2026-07-14T09:00:00Z` | `sha256:5cc220806770ce1fb642583e02a624947a06860d3928d1075e4be2228c6a9ad9` | `snapshots/demo/demo_linear_5_2026-07-14.json` |
| `qiskit_fake` | `fake_manila` | `2024-05-27T15:27:23-03:00` | `sha256:f79216df928d9dba24b98d2f65f3bbf4b918c46e61e7a6455f0c8218ae4179a0` | `snapshots/qiskit_fake/fake_manila/2024-05-27T15-27-23-03-00.json` |

The fake-provider entry is a packaged calibration artifact, not a current live
device observation. The two demo entries are deterministic synthetic fixtures.

## Final validated results

- Verdict: `OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS`; this is not a
  completed empirical live-device pilot.
- Experiment A: complete/pass; packaged FakeManila data; maximum native versus
  reconstructed TVD `2.9842295301563126e-10`, strict threshold `< 0.01`;
  calibration-target delta `8.326672684688674e-17`.
- Experiment B: complete/pass; 36 of 36 circuits; maximum pairwise TVD
  `2.4424906541753444e-15`, mean `6.386881641277444e-16`, threshold `<= 0.02`.
- Experiment C: complete; nested path `[0,1,2,3,4]`; reference GHZ fidelity
  `0.9393617392812814`, `0.8432575063633894`, `0.8274892699450827`, and
  `0.8107439266475058` for 2–5 qubits; maximum GHZ framework TVD
  `2.255140518769849e-16`.
- Experiment D: complete as a deterministic synthetic archive-semantics
  demonstration only; demo timestamps `2024-01-15T09:00:00Z` and
  `2026-07-14T09:00:00Z`; maximum framework drift TVD
  `0.01470699724357372`; not empirical calibration evidence.

## Visual inspection

- `experiments/figures/cross_framework_tvd.png`: all 36 labels and three pair
  columns are legible; color scale is fixed at the `0.02` threshold; title max
  agrees with JSON.
- `experiments/figures/ghz_fidelity.png`: axes, ideal baseline, four model
  markers, packaged provenance, nested path, and values are legible and agree.
- `experiments/figures/drift_comparison.png`: integer qubit axes, T1/T2 values,
  edge-error bars, providers, dates, synthetic warning, and TVD agree with the
  two fixture snapshots and JSON.

## Independent reviews

Independent subagents performed code, scientific-method, and baseline
result-consistency reviews. Their substantive findings are the defects and fixes
recorded above. A final saved-vector recomputation was then performed by the
standalone clean-room script.

## Remaining limitation/blocker

The only external evidence blocker is unavailable IBM Quantum authentication:
there are zero live IBM calibration snapshots and no hardware validation point.
The offline packaged-calibration result is technically verified, but it cannot
support claims about current live-device fidelity or empirical drift.

