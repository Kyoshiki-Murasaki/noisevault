from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import DeviceNoiseSnapshot
from .schema import validate_snapshot


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def snapshot_content_hash(snapshot: DeviceNoiseSnapshot) -> str:
    payload = snapshot.model_dump(mode="json")
    payload["provenance"] = dict(payload["provenance"])
    payload["provenance"]["raw_hash"] = "sha256:SELF"
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def raw_payload_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_snapshot(path: str | Path, *, require_valid: bool = True) -> DeviceNoiseSnapshot:
    path = Path(path)
    snapshot = DeviceNoiseSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    report = validate_snapshot(snapshot)
    if require_valid and not report.valid:
        details = "; ".join(f"{i.path}: {i.message}" for i in report.errors)
        raise ValueError(f"Invalid snapshot {path}: {details}")
    return snapshot


def save_snapshot(snapshot: DeviceNoiseSnapshot, path: str | Path) -> Path:
    report = validate_snapshot(snapshot)
    if not report.valid:
        details = "; ".join(f"{i.path}: {i.message}" for i in report.errors)
        raise ValueError(f"Refusing to save invalid snapshot: {details}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
