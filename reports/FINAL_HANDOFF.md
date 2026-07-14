# NoiseVault final handoff

## Exact verdict

`OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS`

The repaired offline/package pilot satisfies the applicable engineering and
converter-validation acceptance criteria. It is **not** a completed empirical
live-device pilot: IBM Quantum authentication was unavailable, no live IBM
calibration was harvested, Experiment D used deterministic synthetic fixtures,
and no QPU job ran.

## Repository state

- Branch: `pilot/noisevault`.
- Validated implementation/results commit:
  `a76c09e94a230d3d54055220343caf8101d2e3da`.
- Untouched input packet commit:
  `ce262b45b5c9ff0ff39c271f4603d43e1e45805c`.
- Worktree: clean after the final documentation commit; final `HEAD` is reported
  in the Codex handoff response because a commit cannot contain its own hash.
- Remote/push status: no remote was configured, so nothing was pushed.
- Exact push command after the user configures an authorized remote named
  `origin`: `git push -u origin pilot/noisevault`.

## Verification

- Fresh environment/install: passed. The initial system default was unsupported
  Python 3.14.6; the repaired bootstrap automatically found Python 3.12.13,
  created `.venv`, and installed `.[pilot,dev]` successfully.
- Final one-command workflow: `bash scripts/bootstrap.sh` — exit 0.
- Tests: `pytest` — **27 passed in 4.82 seconds**.
- Lint: `ruff check .` — **All checks passed**.
- Offline smoke: passed for 2- and 3-qubit circuits in Qiskit Aer, Cirq, and
  PennyLane using the exported converter objects.
- Full automatic pilot: exit 0; 36/36 circuits completed in all three frameworks;
  144 reference/framework probability vectors saved.
- Snapshot schema: all 3 archive files valid; only expected zero-duration virtual
  `rz` warnings were recorded.
- Independent recomputation: `python scripts/recompute_metrics.py` — **PASS**;
  A–D recomputed from saved vectors with no project metric imports; maximum
  normalization error `3.3306690738754696e-16`.
- Package: sdist and wheel built; `twine check` passed both.
- Visual QA: every final figure opened and checked against JSON/snapshots; labels,
  axes, threshold scale, values, dates, and provenance agree.
- Secret scan: no credential/private-key value was found. The only matching
  source is the intentionally redacted token example in `README.md`. No `.env`
  file or credential was committed.

## Data provenance

| Provider | Backend | Source timestamp | Qubits | Raw hash | Path |
|---|---|---|---:|---|---|
| `qiskit_fake` | `fake_manila` | `2024-05-27T15:27:23-03:00` | 5 | `sha256:f79216df928d9dba24b98d2f65f3bbf4b918c46e61e7a6455f0c8218ae4179a0` | `snapshots/qiskit_fake/fake_manila/2024-05-27T15-27-23-03-00.json` |
| `demo` | `demo_linear_5` | `2024-01-15T09:00:00Z` | 5 | `sha256:98d477858773707ad4dd3b1f9014dd14f9f0ed0304f4f3b16f615e8d5c9335e3` | `snapshots/demo/demo_linear_5_2024-01-15.json` |
| `demo` | `demo_linear_5` | `2026-07-14T09:00:00Z` | 5 | `sha256:5cc220806770ce1fb642583e02a624947a06860d3928d1075e4be2228c6a9ad9` | `snapshots/demo/demo_linear_5_2026-07-14.json` |

The FakeManila file is a packaged Qiskit fake-provider calibration artifact. It
is neither live authentication evidence nor a current device measurement. The
demo pair is deterministic synthetic data used only for reproducible pipeline
and archive-semantics validation. There are no `ibm` or `ibm_csv` files in the
final archive.

- IBM authentication available: **no**.
- Live IBM snapshots harvested: **0**.
- Hardware/QPU job ran: **no**.

## Experiment A — native Aer reconstruction

- Status: complete / **PASS**.
- Provenance: packaged `qiskit_fake` FakeManila calibration, prefix qargs `[0,1]`.
- Method: native `NoiseModel.from_backend()` versus snapshot reconstruction;
  exact direction-aligned basis calibrations; readout disabled on both sides.
- Maximum TVD: `2.9842295301563126e-10`.
- Threshold: strictly `< 0.01`.
- Maximum reconstructed versus Aer-model calibration-target error:
  `8.326672684688674e-17` with tolerance `1e-10` — **PASS**.
- Scientific scope: strong agreement on the tested basis circuits and per-gate
  average infidelity, not full process tomography or hardware validation.

## Experiment B — cross-framework agreement

- Status: complete / **PASS**.
- Circuits: `36 / 36`; Qiskit Aer, Cirq, and PennyLane all completed.
- Maximum pairwise TVD: `2.4424906541753444e-15`.
- Mean pairwise TVD saved by the pilot: `6.386881641277444e-16`.
- Threshold: `<= 0.02` exact density-matrix target.
- Independent clean-room recomputation maximum:
  `2.4424906541753444e-15` — exact headline match.
- The run uses each exported adapter and independent framework engine. Shared
  code is limited to canonical channel construction and exact classical readout
  post-processing; the reference runner is separate from all three wrappers.

## Experiment C — GHZ degradation

- Status: complete.
- Provenance: packaged FakeManila calibration.
- Nested directed physical path: `[0,1,2,3,4]`.
- Reference GHZ fidelities for 2, 3, 4, and 5 qubits:
  `0.9393617392812814`, `0.8432575063633894`,
  `0.8274892699450827`, `0.8107439266475058`.
- Maximum GHZ cross-framework TVD: `2.255140518769849e-16`.
- Figure and JSON agree; fidelity decreases with size on one nested path, avoiding
  the baseline's changing-subset confound.

## Experiment D — versioned drift semantics

- Status: complete as a **deterministic synthetic archive-semantics
  demonstration**; it is not empirical drift evidence and has no pass threshold.
- Older/newer files: `snapshots/demo/demo_linear_5_2024-01-15.json` and
  `snapshots/demo/demo_linear_5_2026-07-14.json`.
- Source timestamps: `2024-01-15T09:00:00Z` and `2026-07-14T09:00:00Z`.
- Providers: `demo`, `demo`; raw hashes are distinct.
- Shared directed physical path: `[0,1,2]`.
- Maximum dated-distribution TVD: `0.01470699724357372`.
- Older/newer probability vectors are saved per framework; independent
  recomputation matches the headline exactly.

## Artifacts

- Generated study report: `experiments/PILOT_RESULTS.md`.
- Machine-readable results: `experiments/results/pilot_results.json`.
- Independent-interpretation dossier: `reports/RESULTS_FOR_INTERPRETATION.md`.
- Full autonomous evidence log: `reports/AUTONOMOUS_RUN_LOG.md`.
- Living execution record: `PLANS.md`.
- Figures:
  - `experiments/figures/cross_framework_tvd.png`
  - `experiments/figures/ghz_fidelity.png`
  - `experiments/figures/drift_comparison.png`
- Snapshot archive: `snapshots/`.

## Remaining blockers and next action

1. IBM Quantum authentication was unavailable. Consequently, the repository has
   no live calibration snapshot, no live-versus-packaged drift comparison, and no
   empirical current-device claim. Resolve by configuring an authorized IBM
   Quantum Platform account, then rerun the full automatic profile; credentials
   must remain outside Git.
2. No authorized Git remote was configured, so the branch was not pushed. After
   reviewing the artifacts and configuring an authorized remote named `origin`,
   run `git push -u origin pilot/noisevault`.

For independent scientific interpretation, attach
`reports/RESULTS_FOR_INTERPRETATION.md`,
`experiments/results/pilot_results.json`, and all files in
`experiments/figures/` to a separate ChatGPT conversation.
