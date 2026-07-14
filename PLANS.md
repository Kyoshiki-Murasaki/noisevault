# NoiseVault autonomous pilot execution plan

This is the living execution record for the autonomous validation run begun on
2026-07-14. The governing acceptance contract is the user-supplied task together
with `COWORK_TASK.md`; the referenced `CHATGPT_WORK_TASK.md` was absent from the
supplied repository at baseline.

## Status legend

- `[x]` complete and evidenced
- `[~]` in progress
- `[ ]` pending

## Execution record

- [x] Read `AGENTS.md`, `COWORK_TASK.md`, `README.md`, `docs/METHODOLOGY.md`,
  `docs/LIMITATIONS.md`, `docs/AUTONOMY.md`, and `docs/ORIGINAL_PILOT_PLAN.md`.
- [x] Confirmed `CHATGPT_WORK_TASK.md` and `PLANS.md` were absent before changes.
- [x] Inventoried all 77 supplied project files and captured initial result,
  snapshot, and figure hashes.
- [x] Confirmed no Git repository, configured remote, IBM token environment
  variable, or `NOISEVAULT_ALLOW_HARDWARE` opt-in existed at baseline.
- [x] Initialized Git on `pilot/noisevault` and preserved the untouched packet in
  commit `ce262b4` after a staged secret-pattern audit.
- [x] Ran the exact bootstrap once with the system default Python; it stopped at
  the declared version gate because the default is Python 3.14.6.
- [x] Reran bootstrap with its supported `PYTHON_BIN=python3.12` override. The
  environment installed, all three frameworks were available, 15 tests passed,
  lint passed, and the full automatic pilot completed using packaged and demo data.
- [x] Reproduce and repair independent review findings, especially Aer readout
  orientation and Experiment A native-model reconstruction semantics.
- [x] Add focused scientific-invariant and framework-independence tests covering
  2- and 3-qubit execution, endianness, qarg order, CNOT direction, Kraus order,
  and asymmetric readout.
- [x] Run offline smoke, complete tests and lint, full automatic pilot, independent
  recomputation from saved probability vectors, and schema validation for every snapshot.
- [x] Visually inspect every final generated figure and cross-check plotted values,
  axes, legends, and labels against the final JSON.
- [x] Reconcile all independent code, scientific, and result-consistency reviews.
- [~] Populate `reports/AUTONOMOUS_RUN_LOG.md`, `reports/FINAL_HANDOFF.md`, and
  `reports/RESULTS_FOR_INTERPRETATION.md` with final evidence and no placeholders.
- [ ] Build/check release artifacts, scan tracked content and task-created history
  for secrets, commit coherent changes, and leave a clean worktree.

## Immutable integrity constraints

- No QPU execution without a pre-existing `NOISEVAULT_ALLOW_HARDWARE=1`; none was present.
- `ibm`, `qiskit_fake`, `ibm_csv`, and `demo` provenance must remain distinct.
- Threshold misses remain results; tests and thresholds will not be weakened.
- A packaged-calibration or deterministic-fixture run is not a completed empirical
  live-device pilot.
