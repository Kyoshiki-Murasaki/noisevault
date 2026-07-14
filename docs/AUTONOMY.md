# Autonomous execution policy

The repository is designed so an agent can proceed without asking the user routine questions.

The agent may:

- create or repair a virtual environment;
- update dependency pins to current compatible releases;
- repair code for documented public API changes;
- use existing IBM credentials already present in the environment;
- run calibration metadata calls that consume no QPU time;
- execute local simulations repeatedly;
- create reports, figures, tests, commits, and a local Git repository if one does not exist.

The agent must not:

- fabricate credentials, live snapshots, hardware results, citations, or passed thresholds;
- spend money or submit QPU jobs without an explicit pre-existing opt-in (`NOISEVAULT_ALLOW_HARDWARE=1`);
- publish a repository, push to a remote, or expose secrets unless the user has already provided the necessary authorization and destination;
- ask the user to resolve ordinary package/API errors that it can diagnose itself;
- erase evidence of failed attempts or threshold misses.

When live authentication is unavailable, the correct autonomous behavior is to finish the fake/offline pilot, clearly label it, and leave one exact blocker in the final handoff.
