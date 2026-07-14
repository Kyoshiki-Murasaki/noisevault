# NoiseVault results for independent interpretation

Claude Cowork must replace this template with `reports/RESULTS_FOR_INTERPRETATION.md` after the final run.

Include, without placeholders:

- exact overall verdict and whether it is empirical, packaged-calibration, or synthetic-only;
- environment and package versions;
- snapshot inventory with provider, backend, calibration/source timestamp, raw hash, and file path;
- Experiment A–D methods, exact metrics, thresholds, pass/fail, and data provenance;
- full explanation of any method change or threshold miss;
- whether IBM authentication was available and whether any QPU job ran;
- figure paths and a one-paragraph reading of each figure;
- limitations and what the results do and do not establish;
- unresolved blockers;
- git branch, commit hash, worktree status;
- exact paths to `experiments/PILOT_RESULTS.md` and `experiments/results/pilot_results.json`.

The document should be self-contained enough for a separate analyst to interpret the pilot without reading terminal logs, but it must point to the machine-readable JSON for independent verification.
