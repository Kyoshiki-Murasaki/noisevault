# NoiseVault pilot packet manifest

## Purpose

This repository is a self-contained, autonomous pilot-study package derived from the supplied NoiseVault project plan. It includes implementation code, dated fixtures, framework adapters, experiment orchestration, generated baseline results, tests, CI, release metadata, scientific limitations, and explicit Claude Cowork operating instructions.

## Start and handoff files

- `START_HERE.md` — user workflow.
- `CLAUDE_COWORK_PROMPT.txt` — prompt to paste into Claude Cowork.
- `COWORK_TASK.md` — exhaustive autonomous execution sequence and acceptance criteria.
- `.claude/CLAUDE.md` — persistent repository-level instructions.
- `reports/RESULTS_FOR_INTERPRETATION_TEMPLATE.md` — template Cowork must replace with the report to attach for independent analysis.

## Implemented pilot components

- schema `0.1`, validation, JSON serialization, hashes, and provenance;
- IBM live metadata harvester, Qiskit fake-provider importer, and CSV fallback;
- Qiskit Aer, Cirq, and PennyLane adapters and independent density-matrix runners;
- GHZ, mirror, and RB-style benchmark generation;
- exact TVD and Hellinger metrics plus readout-confusion handling;
- Experiments A–D, figures, Markdown report, and machine-readable per-circuit probabilities;
- duplicate-payload rejection for temporal drift comparisons;
- CI and scheduled-harvesting workflows;
- Apache-2.0 license and citation metadata.

## Assembly-environment validation

- Python `3.13.5`.
- `15` tests passed.
- `ruff check .` passed.
- sdist and wheel built; `twine check` passed.
- full automatic profile completed using packaged `FakeManilaV2` calibration data and deterministic fixtures.
- live IBM authentication was unavailable; no QPU job ran.
- baseline status: `OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS`.

The baseline is deliberately not described as a completed live-calibration or hardware-validation study. Claude Cowork must rerun the package, use existing credentials only when available, and produce the final provenance-qualified verdict.

## Deliberately excluded from the ZIP

- `.venv/`;
- Python bytecode and tool caches;
- build outputs (`dist/`, `build/`);
- credentials and local `.env` files;
- git history, so Cowork can initialize the requested branch and make coherent final commits.
