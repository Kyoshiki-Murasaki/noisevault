# Local package validation before handoff

This file records only the validation performed while assembling the packet. It is not the final empirical pilot report and must not replace `reports/FINAL_HANDOFF.md`, which ChatGPT Work is instructed to create after rerunning the repository in its own environment.

## Result

- Full automatic profile executed successfully with Qiskit Aer, Cirq, and PennyLane.
- IBM Quantum authentication was unavailable in the assembly environment.
- No QPU job was submitted.
- Primary data: packaged `FakeManilaV2` calibration artifact plus deterministic dated fixtures.
- Overall verdict: `OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS`, not a completed live-calibration pilot.

## Local checks

- `pytest -q`: 15 tests passed after the final consistency test was added.
- `ruff check .`: passed.
- Python package sdist and wheel: built successfully.
- `twine check`: passed for both artifacts.
- Full pilot: 36 benchmark circuits, 144 saved framework/reference probability vectors.

## Preliminary metrics

- Experiment A: maximum TVD `0.0013190365764389755`; target `0.01`; passed on packaged `FakeManilaV2` data.
- Experiment B: maximum pairwise TVD `1.817990202823694e-15`; target `0.02`; passed across 36 circuits.
- Experiment C: GHZ-distribution fidelity declined from approximately `0.9621` at 2 qubits to `0.7889` at 5 qubits under the selected packaged calibration.
- Experiment D: simulation TVD `0.014153788899488833` using the two deterministic demo snapshots. This is an archive-semantics demonstration, not empirical hardware drift.

## Required next step

ChatGPT Work must rerun all checks, attempt live metadata harvesting using only credentials already present, inspect figures, make any repairs, produce the final run log and handoff, initialize git, and commit the final repository. The exact instructions are in `CHATGPT_WORK_TASK.md` and `AGENTS.md`.
