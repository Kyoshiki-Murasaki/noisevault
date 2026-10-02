"""Show how much IBM Fez drifted in 90 days.

Pull today's calibration and the calibration from 90 days ago. Then diff the two profiles.

The script needs network access to IBM's public calibration endpoint, but no account. Each
nv.pull call saves the pulled profile in your vault (~/.noisevault/profiles), so the same refs
load offline after that.
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
