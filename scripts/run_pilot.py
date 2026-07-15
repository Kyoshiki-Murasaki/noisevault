#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisevault.pilot import run_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the NoiseVault pilot autonomously.")
    parser.add_argument("--mode", choices=["auto", "offline", "live"], default="auto")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    result = run_pilot(ROOT, mode=args.mode, profile=args.profile)
    print(json.dumps(result["summary"], indent=2))
    verified = {
        "PILOT_VERIFIED_WITH_LIVE_CALIBRATION_METADATA",
        "OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS",
        "OFFLINE_PIPELINE_VERIFIED",
    }
    return 0 if result["summary"]["status"] in verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
