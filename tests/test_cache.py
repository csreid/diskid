from pathlib import Path

from diskid import cache
from diskid.cli import rename_to_target


def test_save_and_load_mapping_roundtrip():
    mapping_id = cache.save_mapping("identify", "Test Show", [
        {"file": "/a/foo.mkv", "target": "/a/Test Show - S01E01.mkv", "confident": True},
    ])
    loaded = cache.load_mapping(mapping_id)
    assert loaded["id"] == mapping_id
    assert loaded["show"] == "Test Show"
    assert loaded["entries"][0]["target"] == "/a/Test Show - S01E01.mkv"


def test_load_mapping_missing_returns_none():
    assert cache.load_mapping("doesnotexist") is None


def test_rename_to_target_renames(tmp_path: Path):
    src = tmp_path / "foo.mkv"
    src.write_bytes(b"data")
    target = tmp_path / "bar.mkv"

    new_path, reason = rename_to_target(src, target)

    assert new_path == target
    assert reason == ""
    assert target.exists()
    assert not src.exists()


def test_rename_to_target_missing_source(tmp_path: Path):
    src = tmp_path / "missing.mkv"
    target = tmp_path / "bar.mkv"

    new_path, reason = rename_to_target(src, target)

    assert new_path is None
    assert reason == "source file missing"


def test_rename_to_target_existing_target(tmp_path: Path):
    src = tmp_path / "foo.mkv"
    src.write_bytes(b"data")
    target = tmp_path / "bar.mkv"
    target.write_bytes(b"already here")

    new_path, reason = rename_to_target(src, target)

    assert new_path is None
    assert reason == "target exists"
    assert src.exists()
