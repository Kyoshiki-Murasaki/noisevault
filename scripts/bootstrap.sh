#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit("NoiseVault requires Python 3.11, 3.12, or 3.13.")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[pilot,dev]'
noisevault doctor
pytest
ruff check .
python scripts/run_pilot.py --mode auto --profile full

echo "Bootstrap finished. Inspect experiments/PILOT_RESULTS.md and experiments/results/pilot_results.json."
