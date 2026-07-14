# NoiseVault results for independent interpretation

## Interpretation brief

The exact repository verdict is
`OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS`. This means the package,
schema, public converters, three independent framework engines, threshold logic,
reports, and deterministic archive workflow were verified. It does **not** mean
that a current live device was measured or modeled empirically.

The primary A–C data are the calibration artifact packaged with Qiskit IBM
Runtime's `FakeManilaV2`. Experiment D uses two deliberately synthetic dated
fixtures. IBM authentication was unavailable, zero live IBM snapshots were
harvested, and no QPU job ran.

## Environment and reproducibility

- Run timestamp in the final JSON: `2026-07-14T09:28:19.523710Z`.
- Operating system: macOS 26.4.1, build 25E253, Apple silicon.
- Python: 3.12.13 in a fresh `.venv` created by the bootstrap.
- Package versions: NoiseVault 0.1.0; NumPy 2.5.1; Pydantic 2.13.4;
  Qiskit 2.5.0; Qiskit Aer 0.17.2; Qiskit IBM Runtime 0.47.0;
  Cirq Core 1.6.1; PennyLane 0.45.1; SciPy 1.18.0;
  Matplotlib 3.11.0; pytest 9.1.1; Ruff 0.15.21.
- Final `bash scripts/bootstrap.sh`: exit 0; 27 tests passed in 4.82 seconds;
  Ruff passed; full automatic profile passed.
- Independent `scripts/recompute_metrics.py`: `PASS` for all 144 saved vectors;
  maximum normalization error `3.3306690738754696e-16`.
- Build: sdist/wheel succeeded; both passed `twine check`.

## Snapshot inventory and evidence class

| Provider | Backend | Calibration/source timestamp | Raw hash | Path | Evidence class |
|---|---|---|---|---|---|
| `qiskit_fake` | `fake_manila` | `2024-05-27T15:27:23-03:00` | `sha256:f79216df928d9dba24b98d2f65f3bbf4b918c46e61e7a6455f0c8218ae4179a0` | `snapshots/qiskit_fake/fake_manila/2024-05-27T15-27-23-03-00.json` | Packaged calibration artifact, not live/current. |
| `demo` | `demo_linear_5` | `2024-01-15T09:00:00Z` | `sha256:98d477858773707ad4dd3b1f9014dd14f9f0ed0304f4f3b16f615e8d5c9335e3` | `snapshots/demo/demo_linear_5_2024-01-15.json` | Deterministic synthetic fixture. |
| `demo` | `demo_linear_5` | `2026-07-14T09:00:00Z` | `sha256:5cc220806770ce1fb642583e02a624947a06860d3928d1075e4be2228c6a9ad9` | `snapshots/demo/demo_linear_5_2026-07-14.json` | Deterministic synthetic fixture. |

All three schema files validate. Their raw hashes are provenance identifiers, not
provider signatures. No `ibm` live data or `ibm_csv` imported data are present.

## Material method correction made during validation

The packet baseline applied a depolarizing channel whose average infidelity was
the full archived gate error and then added thermal relaxation. That double
counted the source error. The repaired method first computes relaxation fidelity,
then solves for the residual depolarizing strength so the composition targets the
archived error, matching Qiskit Aer's backend-noise convention. If relaxation
alone exceeds the archived error, the relaxation floor is retained.

Additional corrections enforce directed qargs, a calibrated path for every
benchmark entangler, exact-name-before-alias selection, explicit Qiskit local
Kraus reordering, Aer readout-matrix transposition, PennyLane order-sensitive
CNOT predicates, partial T1/T2 semantics, and actual execution of all three
exported adapter objects. `docs/METHODOLOGY.md` and
`reports/AUTONOMOUS_RUN_LOG.md` contain the full rationale.

## Experiment A — native-model reconstruction

- Status: complete / **PASS**.
- Evidence: packaged `qiskit_fake` FakeManila calibration on prefix qargs `[0,1]`.
- Maximum native-versus-reconstructed TVD: `2.9842295301563126e-10`.
- Acceptance target: strictly `< 0.01`.
- Maximum reconstructed versus Aer-model calibration-target delta:
  `8.326672684688674e-17`; acceptance tolerance `1e-10`; **PASS**.
- Readout was disabled in both compared models. The output vectors for all four
  circuits and the per-gate calibration checks are stored under `experiment_a`
  in the JSON.

Interpretation constraint: this supports faithful reconstruction of Aer's
standard calibration-derived approximation on the tested qargs and circuits. It
is not process tomography and says nothing about current hardware accuracy.

## Experiment B — cross-framework agreement

- Status: complete / **PASS**.
- Complete circuits: `36 / 36`.
- Engines: Qiskit Aer, Cirq, and PennyLane, each reached through the exported
  NoiseVault adapter and run independently with exact density matrices.
- Maximum pairwise TVD: `2.4424906541753444e-15`.
- Saved mean pairwise TVD: `6.386881641277444e-16`.
- Target: `<= 0.02`.
- Independent recomputation maximum: `2.4424906541753444e-15`; **PASS**.

Interpretation constraint: machine-precision agreement demonstrates converter
consistency for this canonical local Markovian model. It does not validate that
the model predicts a physical processor.

## Experiment C — GHZ degradation

- Status: complete.
- Evidence: packaged FakeManila calibration.
- Nested directed path: `[0,1,2,3,4]`; every prefix uses calibrated entanglers.
- Reference GHZ fidelities:
  - 2 qubits: `0.9393617392812814`
  - 3 qubits: `0.8432575063633894`
  - 4 qubits: `0.8274892699450827`
  - 5 qubits: `0.8107439266475058`
- Maximum GHZ cross-framework TVD: `2.255140518769849e-16`.

The monotonic decline is a model sanity check under a fixed nested path. It is
not evidence that a live device follows the same curve.

## Experiment D — dated archive semantics

- Status: complete as a **deterministic synthetic archive-semantics
  demonstration**.
- Older/newer source dates: `2024-01-15T09:00:00Z` and
  `2026-07-14T09:00:00Z`.
- Providers/backend: `demo` + `demo`, `demo_linear_5`.
- Distinct raw payload hashes: yes.
- Shared directed path: `[0,1,2]`.
- Circuit: `mirror_n3_d8`.
- Maximum framework TVD between dated vectors: `0.01470699724357372`.
- The JSON stores older and newer distributions for reference, Qiskit, Cirq, and
  PennyLane; independent recomputation matches exactly.

This result proves that the archive and converters can compare distinct dated
payloads. It must not be cited as measured device drift, packaged calibration
drift, or a live-versus-stale comparison.

## Figure readings

1. `experiments/figures/cross_framework_tvd.png` uses the fixed acceptance scale
   `0–0.02`; all cells are visually near zero, and the title reports maximum
   `2.442e-15` / pass. Circuit labels and all three framework pairs are legible.
2. `experiments/figures/ghz_fidelity.png` shows the ideal baseline and four
   overlapping reference/framework traces on nested path `[0,1,2,3,4]`. The
   packaged provider/date are in the title; the curve drops from about `0.9394`
   to `0.8107`.
3. `experiments/figures/drift_comparison.png` compares T1, T2, and directed edge
   errors for the two fixture dates. Its title explicitly states
   `DETERMINISTIC SYNTHETIC FIXTURES` and reports TVD `0.014707`.

## Limitations and non-claims

The canonical model includes local depolarization, zero-temperature amplitude
relaxation, pure dephasing, and independent asymmetric readout confusion. It
omits coherent over-rotations, crosstalk, leakage, correlated readout,
non-Markovian effects, context/spectator dependence, pulse behavior, uncertainty,
and change between calibration and workload execution.

Portable high-level operations can use documented nearest calibrated primitives;
this is not a pulse decomposition. Experiment A is not full channel tomography.
Experiment B proves portability consistency, not hardware realism. Experiment C
is a modeled sanity curve. Experiment D is synthetic only.

## Authentication, hardware, blockers, and Git

- Live IBM authentication available: **no**.
- Live IBM calibration snapshots: **0**.
- Hardware job ran: **no**.
- Primary scientific blocker: no current live calibration or hardware point, so
  the empirical live-device pilot remains incomplete.
- Publication blocker: no authorized Git remote was configured; nothing was pushed.
- Branch: `pilot/noisevault`.
- Validated implementation/results commit:
  `a76c09e94a230d3d54055220343caf8101d2e3da`.
- Final worktree: clean after committing the handoff documents; final `HEAD` is
  reported in the originating Codex response.

## Files to use in the independent review

Attach this file together with:

- `experiments/results/pilot_results.json`
- `experiments/figures/cross_framework_tvd.png`
- `experiments/figures/ghz_fidelity.png`
- `experiments/figures/drift_comparison.png`

The generated narrative study report is `experiments/PILOT_RESULTS.md`; the full
execution and repair log is `reports/AUTONOMOUS_RUN_LOG.md`.
