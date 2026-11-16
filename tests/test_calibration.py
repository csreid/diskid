import json

from diskid import cache, calibration
from diskid.eventlog import EventLog
from diskid.match import BayesianIdentifier

from test_match import make_show, snippet_of


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_trace_has_one_entry_per_clip_with_running_posterior():
    docs = make_show()
    ident = BayesianIdentifier(docs)
    ident.observe(20.0, snippet_of(docs, (1, 3)))
    ident.observe(60.0, "   ")  # silent clip: recorded, not counted
    assert [c["t"] for c in ident.trace] == [20.0, 60.0]
    assert ident.trace[0]["top"][:2] == [1, 3] and 0 <= ident.trace[0]["null"] <= 1
    assert ident.trace[1]["top"] is None and ident.n_clips == 1


def test_model_constants_are_recorded_by_name():
    from diskid import match
    consts = calibration.model_constants()
    assert consts["TEXT_LLR_CAP"] == match.TEXT_LLR_CAP and "STT_ERROR_RATE" in consts


def test_applying_a_plan_labels_the_observation_but_skips_unmoved_files(tmp_path):
    a, b = tmp_path / "a.mkv", tmp_path / "b.mkv"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    entries = [
        {"file": str(a), "target": str(tmp_path / "A.mkv"), "confident": True, "size": 1,
         "calibration_id": "obs1", "imdb_episode": [2, 6]},
        {"file": str(b), "target": None, "confident": False, "size": 1,
         "calibration_id": "obs2", "imdb_episode": None},
    ]
    pid = cache.save_mapping("test", "Show", entries, state_dir=tmp_path / "state")
    plan = cache.load_mapping(pid, tmp_path / "state")
    log = tmp_path / "calibration.jsonl"
    cache.apply_plan(plan, EventLog(None), log)
    (label,) = _lines(log)
    assert label["event"] == "label" and label["id"] == "obs1"
    assert label["episode"] == [2, 6] and label["source"] == "plan_applied"


def test_no_log_path_means_no_label(tmp_path):
    calibration.write_label(None, "obs1", (1, 1), "manual")  # must not raise
    calibration.write_label(tmp_path / "c.jsonl", None, (1, 1), "manual")
    assert not (tmp_path / "c.jsonl").exists()


def test_label_command_requires_a_known_observation(tmp_path, monkeypatch):
    import pytest
    from diskid import cli
    monkeypatch.setenv("DISKID_CALIBRATION_LOG", str(tmp_path / "c.jsonl"))
    monkeypatch.setattr("sys.argv", ["diskid", "label", "nope", "S02E06"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert "no observation" in str(e.value)
    calibration.log(tmp_path / "c.jsonl").write("observation", id="abc")
    monkeypatch.setattr("sys.argv", ["diskid", "label", "abc", "S02E06"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0
    assert _lines(tmp_path / "c.jsonl")[-1]["source"] == "manual"
