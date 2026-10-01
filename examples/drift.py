"""How much did IBM Fez drift? Pull today's calibration and one from 90 days ago, then diff them.

Needs network access to IBM's public calibration endpoint (no account). Pulled profiles go to
your local vault (~/.noisevault/profiles), so the same refs load offline afterwards.
"""

import sys
from datetime import UTC, datetime, timedelta

import noisevault as nv

then = datetime.now(UTC) - timedelta(days=90)
try:
    newer = nv.pull("ibm_fez")
    older = nv.pull("ibm_fez", at=then)
except nv.SourceUnavailable as exc:
    print(f"skipped: {exc}")
    sys.exit(0)

for profile in (older, newer):
    print(
        f"pulled {profile.id}@{profile.device.calibrated_at:%Y-%m-%d %H:%M} UTC"
        f"  {profile.short_fingerprint}"
    )
print()
print(older.diff(newer).summary())
