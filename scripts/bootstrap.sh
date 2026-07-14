#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

supports_python() {
  command -v "$1" >/dev/null 2>&1 && "$1" -c \
    'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)' \
    >/dev/null 2>&1
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3 python3.13 python3.12 python3.11; do
    if supports_python "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "NoiseVault requires Python 3.11, 3.12, or 3.13; no supported interpreter was found." >&2
  exit 1
fi
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
