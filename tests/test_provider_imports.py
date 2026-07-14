from __future__ import annotations

from noisevault.providers.qiskit_fake import import_fake_backend, list_fake_backends


def test_fake_provider_discovery_includes_current_and_v2_classes():
    names = list_fake_backends()
    assert "FakeManilaV2" in names
    assert any(not name.endswith("V2") for name in names)


def test_fake_import_is_idempotent(tmp_path):
    first = import_fake_backend("FakeManilaV2", tmp_path)
    first_content = first.read_bytes()
    first_mtime = first.stat().st_mtime_ns
    second = import_fake_backend("FakeManilaV2", tmp_path)
    assert second == first
    assert second.read_bytes() == first_content
    assert second.stat().st_mtime_ns == first_mtime

