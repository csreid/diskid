import json
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


def test_reapplying_a_plan_keeps_the_record_of_files_already_moved(tmp_path):
    from diskid.eventlog import EventLog
    a, b = tmp_path / "a.mkv", tmp_path / "b.mkv"
    a.write_bytes(b"x")
    entries = [
        {"file": str(a), "target": str(tmp_path / "A.mkv"), "confident": True, "size": 1},
        {"file": str(b), "target": str(tmp_path / "B.mkv"), "confident": True, "size": 1},
    ]
    state = tmp_path / "state"
    pid = cache.save_mapping("test", "Show", entries, state_dir=state)
    events = EventLog(state / "events.jsonl")
    cache.apply_plan(cache.load_mapping(pid, state), events)   # b is missing: not moved
    b.write_bytes(b"x")
    results = cache.apply_plan(cache.load_mapping(pid, state), events)  # retry
    assert [r["renamed"] for r in results] == [True, True]
    assert (tmp_path / "A.mkv").exists() and (tmp_path / "B.mkv").exists()
    kinds = [json.loads(l)["event"] for l in (state / "events.jsonl").read_text().splitlines()]
    assert kinds.count("file_moved") == 2 and kinds.count("file_not_moved") == 1


def _disc_plan(tmp_path, with_leftover=False):
    """A disc folder with two files in extras/, both in a plan that moves them away."""
    disc = tmp_path / "staging" / "movies" / "Disc (2015)"
    (disc / "extras").mkdir(parents=True)
    entries = []
    for i, name in enumerate(["a.mkv", "b.mkv"], 1):
        f = disc / "extras" / name
        f.write_bytes(b"x")
        entries.append({"file": str(f), "target": str(tmp_path / "shows" / f"S01E0{i}.mkv"),
                        "confident": True, "size": 1})
    if with_leftover:
        (disc / "extras" / "bonus.mkv").write_bytes(b"x")
    state = tmp_path / "state"
    pid = cache.save_mapping("test", "Show", entries, state_dir=state, source_dir=str(disc))
    return disc, cache.load_mapping(pid, state), state


def test_apply_removes_the_emptied_disc_folder_and_logs_it(tmp_path):
    from diskid.eventlog import EventLog
    disc, plan, state = _disc_plan(tmp_path)
    cache.apply_plan(plan, EventLog(state / "events.jsonl"))
    assert not disc.exists()                        # extras/ and the disc folder are gone
    assert (tmp_path / "staging" / "movies").is_dir()   # but never the folder above it
    events = [json.loads(l) for l in (state / "events.jsonl").read_text().splitlines()]
    pruned = [e for e in events if e["event"] == "dirs_pruned"]
    assert pruned and str(disc) in pruned[0]["removed"]


def test_apply_keeps_a_folder_that_still_holds_files(tmp_path):
    disc, plan, _ = _disc_plan(tmp_path, with_leftover=True)
    cache.apply_plan(plan)
    assert (disc / "extras" / "bonus.mkv").exists()


def test_apply_leaves_the_folder_alone_while_arm_still_has_a_job_for_it(tmp_path):
    import sqlite3
    db = tmp_path / "arm.db"
    con = sqlite3.connect(db)
    con.execute("create table job (job_id integer, label text, status text, path text)")
    con.execute("insert into job values (1, 'X', 'transcoding', "
                "'/home/arm/media/staging/movies/Disc (2015)')")
    con.commit()
    con.close()
    disc, plan, _ = _disc_plan(tmp_path)
    cache.apply_plan(plan, arm_db=db)
    assert disc.is_dir() and (disc / "extras").is_dir()      # empty but possibly in use
    # once ARM says the job finished, a later apply cleans it
    con = sqlite3.connect(db)
    con.execute("update job set status = 'success'")
    con.commit()
    con.close()
    disc2, plan2, _ = _disc_plan(tmp_path / "again")
    (tmp_path / "again").mkdir(exist_ok=True)
    con = sqlite3.connect(db)
    con.close()
    cache.apply_plan(plan2, arm_db=db)
    assert not disc2.exists()
