# Claude Cowork autonomous task: complete the NoiseVault pilot

## Mission

Take this repository from a prepared, executable pilot package to a fully verified final pilot repository. Work autonomously. Do not stop for routine clarification, dependency failures, public API drift, lint errors, test failures, plotting problems, or report formatting issues. Diagnose and repair them.

The original project rationale and intended experiments are in `docs/ORIGINAL_PILOT_PLAN.md`. The repository code is the current implementation, but neither the code nor the plan is assumed infallible. Preserve the scientific intent while correcting technical mistakes honestly.

## Non-negotiable integrity rules

1. Never invent a live calibration, hardware run, device name, date, metric, citation, or successful threshold.
2. Preserve the distinction among:
   - live IBM Quantum Platform calibration metadata (`provider=ibm`),
   - packaged Qiskit fake-provider calibration artifacts (`provider=qiskit_fake`),
   - imported CSV data (`provider=ibm_csv`), and
   - deterministic synthetic fixtures (`provider=demo`).
3. Do not submit a paid or free QPU job unless `NOISEVAULT_ALLOW_HARDWARE=1` already exists in the environment. The core task does not require hardware.
4. Do not expose or commit credentials. Check `.env`, shell variables, config files, logs, snapshots, and git diff before finalizing.
5. A threshold miss is a result. Investigate it, fix genuine converter defects, then report the final value even if it still misses.
6. Do not weaken tests or thresholds merely to obtain a pass. Any justified methodology change must be documented in `reports/AUTONOMOUS_RUN_LOG.md` and reflected in `docs/METHODOLOGY.md`.

## Required execution sequence

1. Read `README.md`, `docs/METHODOLOGY.md`, `docs/LIMITATIONS.md`, `docs/AUTONOMY.md`, and the original plan.
2. Inspect the complete repository before editing.
3. Run `bash scripts/bootstrap.sh`.
4. Fix every code, dependency, API, typing, test, and lint issue encountered. Prefer current public APIs and primary documentation.
5. Run the offline smoke profile until it succeeds:
   ```bash
   source .venv/bin/activate
   python scripts/run_pilot.py --mode offline --profile smoke
   ```
6. Run the complete test/lint suite:
   ```bash
   pytest
   ruff check .
   ```
7. Attempt the full automatic pilot:
   ```bash
   python scripts/run_pilot.py --mode auto --profile full
   ```
   Use already configured IBM credentials if available. Do not ask for a token. If authentication is absent, continue with packaged fake-backend and deterministic data and record one precise blocker.
8. Inspect all generated figures visually and verify that labels, axes, legends, and values agree with `pilot_results.json`.
9. Independently recompute the headline metrics from the machine-readable per-circuit probabilities. Add or improve tests if any discrepancy appears.
10. Review Experiment A critically. Confirm that native and reconstructed models use aligned qargs and compatible basis operations. If the current approach is invalid under installed Qiskit versions, implement a scientifically valid alternative rather than suppressing the experiment.
11. Review Experiment B for endianness, Kraus ordering, CNOT control/target orientation, readout-matrix orientation, and PennyLane/Cirq custom-channel semantics. Cross-framework agreement must not rely on all three wrappers calling the same internal simulator.
12. Review Experiment D. Prefer, in order:
    - two live dated snapshots of the same backend,
    - a live snapshot and matching packaged fake snapshot,
    - two packaged snapshots of the same backend,
    - the committed deterministic dated pair.
    State exactly which case was used.
13. Ensure the final report contains no placeholders such as `N`, `X`, `Y`, `TODO`, or an unqualified claim that synthetic data are live.
14. Create `reports/AUTONOMOUS_RUN_LOG.md` recording commands, environment, material fixes, data sources, skipped operations, and unresolved limitations.
15. Create `reports/FINAL_HANDOFF.md` containing:
    - overall verdict;
    - exact tests/lint results;
    - data sources and snapshot counts;
    - Experiment A–D headline metrics and pass/fail status;
    - whether live IBM authentication was available;
    - whether any hardware job was run;
    - exact paths to the final report, JSON, figures, and snapshots;
    - remaining blockers and recommended next action;
    - git branch, commit hash, and clean/dirty status.
16. Create `reports/RESULTS_FOR_INTERPRETATION.md` from `reports/RESULTS_FOR_INTERPRETATION_TEMPLATE.md`. It must be a self-contained, evidence-grounded report that the user can attach to a separate ChatGPT conversation together with `experiments/results/pilot_results.json`. Include exact metrics, provenance, interpretation constraints, figure paths, blockers, and git state; remove every placeholder.
17. Initialize git if needed, use branch `pilot/noisevault`, and create coherent commits. Do not configure or push a remote unless one is already authorized. End with a clean worktree except for explicitly documented generated artifacts that are intentionally untracked; preferably commit all intended final artifacts.

## Acceptance criteria

The task is complete only when all applicable items below are true:

- package installs in a fresh virtual environment;
- `pytest` passes;
- `ruff check .` passes;
- schema fixtures validate;
- Qiskit Aer, Cirq, and PennyLane each execute at least the 2-qubit and 3-qubit smoke circuits independently;
- generated probabilities are normalized and finite;
- Experiment B has a real three-framework comparison and the report states whether the `0.02` TVD target passed;
- Experiment C figure and metrics agree;
- Experiment D compares two genuinely dated snapshots and labels their provenance correctly;
- Experiment A is either complete with a valid native comparison or marked incomplete with a technically precise explanation after reasonable repair attempts;
- `experiments/PILOT_RESULTS.md` and `experiments/results/pilot_results.json` are mutually consistent;
- no secret is present in tracked files or git history created during this task;
- final handoff, interpretation report, and run log exist;
- repository is ready for the user to review and then publish to GitHub.

## Final response to the user inside Cowork

Do not give a vague summary. Return the contents or a faithful concise rendering of `reports/FINAL_HANDOFF.md`, including the exact verdict, metrics, file paths, commit hash, and blockers. Explicitly tell the user to attach `reports/RESULTS_FOR_INTERPRETATION.md` and `experiments/results/pilot_results.json` for independent interpretation. Do not say the pilot is complete if only the synthetic pipeline ran.
