"""The MCP server over real HTTP (Starlette's test client), with a fake identifier."""

import json
import time

import pytest

pytest.importorskip("mcp")
from starlette.testclient import TestClient  # noqa: E402

from diskid import cache, mcp_server  # noqa: E402
from diskid.identify import EpisodeRef, FileResult, Show  # noqa: E402
from diskid.jobs import RerunManager  # noqa: E402
from diskid.match import ConsensusScore  # noqa: E402

SHOW = Show(1, "Test Show")
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class FakeIdentifier:
    """identify_file 'finds' S01E03 at 0.999 after however many clips it was allowed."""
    def __init__(self):
        self.config = type("C", (), {})()
        self.progress = None
        self.seen_max_clips = None

    def identify_file(self, path, show=None, **_):
        import dataclasses as dc
        self.seen_max_clips = getattr(self.config, "max_clips", None)
        ref = EpisodeRef(1, 3, 1, 3, "imdb")
        return FileResult(path=path, status="identified", show=SHOW, episode=ref, clips_used=12,
                          candidates=[ConsensusScore(1, 3, 0.999, 0.0, 12, 3)],
                          target_path=path.with_name("Test Show - S01E03.mkv"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    import dataclasses
    state = tmp_path / "state"
    # the fake config must support dataclasses.replace in the rerun worker
    @dataclasses.dataclass(frozen=True)
    class Cfg:
        max_clips: int = 10
        min_clips: int = 3
    ident = FakeIdentifier()
    ident.config = Cfg()
    ctx = mcp_server.McpContext(
        state_dir=state, folders=[tmp_path / "staging"], reruns=RerunManager(ident, state),
        calibration_log=state / "calibration.jsonl")
    return tmp_path, state, ctx, ident


def make_plan(tmp_path, state, specs, show="Test Show"):
    """specs: (filename, confident, score). Returns (plan id, [source paths])."""
    d = tmp_path / "staging" / "Disc"
    d.mkdir(parents=True, exist_ok=True)
    entries, paths = [], []
    for i, (name, ok, score) in enumerate(specs, 1):
        f = d / name
        f.write_bytes(b"x" * 10)
        paths.append(f)
        entries.append(dict(
            file=str(f), target=str(tmp_path / "shows" / "Show (2020)" / "Season 01" /
                                    f"Show - S01E{i:02d}.mkv") if ok else None,
            confident=ok, status="identified" if ok else "uncertain",
            episode=f"S01E{i:02d}" if ok else None, score=score if ok else None,
            best_guess=f"S01E{i:02d}", best_score=score, size=10, calibration_id=f"obs{i}",
            imdb_episode=[1, i] if ok else None, warnings=[], error=None))
    return cache.save_mapping("test", show, entries, state_dir=state), paths


def call(client, tool, **args):
    r = client.post("/mcp", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": args}})
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    return "".join(c["text"] for c in result["content"])


@pytest.fixture
def client(env):
    _, _, ctx, _ = env
    app = mcp_server.make_app(ctx, host="127.0.0.1", port=8765, token=None)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


def test_refuses_a_network_bind_without_a_token(env):
    _, _, ctx, _ = env
    with pytest.raises(ValueError, match="without a token"):
        mcp_server.make_app(ctx, host="0.0.0.0", port=8765, token=None)


def test_bearer_token_is_required_when_set(env):
    _, _, ctx, _ = env
    app = mcp_server.make_app(ctx, host="0.0.0.0", port=8765, token="s3cret")
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    with TestClient(app) as c:
        assert c.post("/mcp", headers=HEADERS, json=body).status_code == 401
        bad = {**HEADERS, "Authorization": "Bearer nope"}
        assert c.post("/mcp", headers=bad, json=body).status_code == 401
        good = {**HEADERS, "Authorization": "Bearer s3cret"}
        r = c.post("/mcp", headers=good, json=body)
        assert r.status_code == 200
        assert {t["name"] for t in r.json()["result"]["tools"]} == {
            "queue", "unidentified", "inspect_file", "show_plan", "rerun", "job_status",
            "preview_apply", "apply", "label"}


def test_unidentified_lists_only_files_still_unsure_and_on_disk(env, client):
    tmp, state, *_ = env
    pid, (a, b) = make_plan(tmp, state, [("a.mkv", False, 0.1), ("b.mkv", False, 0.2)])
    text = call(client, "unidentified")
    assert "a.mkv" in text and "b.mkv" in text and f"{pid}.0" in text
    b.unlink()                                   # moved/deleted: drops off
    assert "b.mkv" not in call(client, "unidentified")
    # a newer plan where a.mkv is identified supersedes the old verdict
    cache.save_mapping("rerun", "Test Show", [dict(
        file=str(a), target=str(tmp / "t.mkv"), confident=True, status="identified",
        episode="S01E01", score=0.99, size=10, calibration_id="x", imdb_episode=[1, 1],
        warnings=[], error=None)], state_dir=state)
    assert call(client, "unidentified") == "No unidentified files waiting."


def test_show_plan_and_bad_ids_are_readable_errors(env, client):
    tmp, state, *_ = env
    pid, _ = make_plan(tmp, state, [("a.mkv", True, 0.999), ("b.mkv", False, 0.2)])
    text = call(client, "show_plan", plan_id=pid)
    assert "1/2 identified" in text and f"{pid}.1" in text and "UNSURE" in text
    assert call(client, "show_plan", plan_id="deadbeef").startswith("Error: no plan")
    assert call(client, "rerun", file_id="nonsense").startswith("Error:")


def test_apply_needs_preview_then_confirm_and_moves_files(env, client):
    tmp, state, *_ = env
    pid, (a, b) = make_plan(tmp, state, [("a.mkv", True, 0.999), ("b.mkv", True, 0.995)])
    assert "run preview_apply" in call(client, "apply", plan_id=pid, confirm=True)
    prev = call(client, "preview_apply", plan_id=pid)
    assert "would move 2" in prev and "Show (2020)/Season 01/Show - S01E01.mkv" in prev
    assert a.exists()                                       # a preview moves nothing
    assert "confirm must be true" in call(client, "apply", plan_id=pid, confirm=False)
    out = call(client, "apply", plan_id=pid, confirm=True)
    assert "moved 2" in out and not a.exists() and not b.exists()
    assert (tmp / "shows/Show (2020)/Season 01/Show - S01E01.mkv").exists()
    events = [json.loads(l) for l in (state / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "mcp_apply" and e["moved"] == 2 for e in events)
    assert "already applied" in call(client, "preview_apply", plan_id=pid)


def test_apply_refuses_if_files_changed_after_the_preview(env, client):
    tmp, state, *_ = env
    pid, (a, b) = make_plan(tmp, state, [("a.mkv", True, 0.999), ("b.mkv", True, 0.999)])
    call(client, "preview_apply", plan_id=pid)
    b.unlink()
    assert "changed since the preview" in call(client, "apply", plan_id=pid, confirm=True)
    assert a.exists()


def test_apply_refuses_a_plan_with_a_low_scoring_file(env, client):
    tmp, state, *_ = env
    pid, (a, _) = make_plan(tmp, state, [("a.mkv", True, 0.999), ("b.mkv", True, 0.95)])
    prev = call(client, "preview_apply", plan_id=pid)
    assert "CANNOT APPLY" in prev and "b.mkv" in prev
    assert "run preview_apply" in call(client, "apply", plan_id=pid, confirm=True)
    assert a.exists()


def test_apply_never_overwrites_an_existing_target(env, client):
    tmp, state, *_ = env
    pid, (a, _) = make_plan(tmp, state, [("a.mkv", True, 0.999), ("b.mkv", True, 0.999)])
    existing = tmp / "shows/Show (2020)/Season 01/Show - S01E01.mkv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"precious")
    prev = call(client, "preview_apply", plan_id=pid)
    assert "target already exists" in prev and "would move 1" in prev
    call(client, "apply", plan_id=pid, confirm=True)
    assert existing.read_bytes() == b"precious" and a.exists()


def test_rerun_runs_in_the_background_and_writes_a_new_plan(env, client):
    tmp, state, ctx, ident = env
    pid, (a,) = make_plan(tmp, state, [("a.mkv", False, 0.001)])
    started = call(client, "rerun", file_id=f"{pid}.0", clips=25)
    job = started.split()[1].rstrip(":")
    for _ in range(100):
        text = call(client, "job_status", job_id=job)
        if "done" in text or "failed" in text:
            break
        time.sleep(0.05)
    assert "done" in text and "identified as S01E03" in text and "New plan" in text
    assert ident.seen_max_clips == 25 and ident.config.max_clips == 10   # restored after
    assert a.exists()                                  # a rerun never moves anything
    assert "clips must be between" in call(client, "rerun", file_id=f"{pid}.0", clips=500)


def test_label_writes_a_manual_label_for_the_files_observation(env, client):
    tmp, state, ctx, _ = env
    pid, _ = make_plan(tmp, state, [("a.mkv", False, 0.1)])
    assert "Labelled a.mkv as S01E01" in call(client, "label", file_id=f"{pid}.0", episode="S01E01")
    rec = json.loads(ctx.calibration_log.read_text().splitlines()[-1])
    assert rec == {**rec, "event": "label", "id": "obs1", "episode": [1, 1], "source": "manual"}
    assert call(client, "label", file_id=f"{pid}.0", episode="nope").startswith("Error")


def test_a_rejected_request_logs_why_but_never_the_token(env, caplog):
    _, _, ctx, _ = env
    app = mcp_server.make_app(ctx, host="0.0.0.0", port=8765, token="s3cret-value")
    with TestClient(app) as c, caplog.at_level("WARNING", logger="diskid.mcp"):
        c.get("/")
        c.post("/mcp", headers={"Authorization": "Bearer nope"}, json={})
    text = caplog.text
    assert "no Authorization header" in text and "does not match" in text
    assert "s3cret-value" not in text and "nope" not in text


def test_a_rerun_plan_still_sees_its_disc_and_extras_is_not_shown_as_the_disc(env, client):
    """Reruns are one-file plans; their gap hint must come from the whole disc, and ARM's
    `extras/` subfolder must not be mistaken for a description of the file."""
    tmp, state, ctx, _ = env
    disc = tmp / "staging" / "Disc (2015)"
    (disc / "extras").mkdir(parents=True)
    def entry(name, ok, ep, guess):
        f = disc / "extras" / name
        f.write_bytes(b"x")
        return dict(file=str(f), target=str(tmp / "t" / name) if ok else None, confident=ok,
                    status="identified" if ok else "uncertain", episode=f"S01E{ep:02d}" if ok else None,
                    score=0.999 if ok else None, best_guess=guess, best_score=0.0, size=1,
                    calibration_id=None, imdb_episode=[1, ep] if ok else None, warnings=[], error=None)
    # the disc's own plan: E01, E02, E04 identified, one file unsure
    cache.save_mapping("watch", "Test Show", [entry("a.mkv", True, 1, "S01E01"),
                                              entry("b.mkv", True, 2, "S01E02"),
                                              entry("d.mkv", True, 4, "S01E04"),
                                              entry("c.mkv", False, 0, "S01E03")],
                       state_dir=state, source_dir=str(disc))
    # a later rerun of c.mkv: a ONE-file plan, still unsure
    cache.save_mapping("mcp-rerun", "Test Show", [entry("c.mkv", False, 0, "S01E03")],
                       state_dir=state, source_dir=str(disc))
    text = call(client, "unidentified")
    assert "1 file(s) not identified" in text                # the rerun supersedes the old entry
    assert "LIKELY AN EPISODE" in text
    assert "disc 'Disc (2015)'" in text and "extras" not in text.replace("extras/", "")


def test_edge_episodes_of_a_discs_run_count_as_likely_episodes(env, client):
    tmp, state, *_ = env
    pid, _ = make_plan(tmp, state, [("a.mkv", True, 0.999), ("b.mkv", True, 0.999),
                                    ("c.mkv", False, 0.0), ("d.mkv", False, 0.0)])
    plan = cache.load_mapping(pid, state)
    # identified E02 and E03 (positions 1-2); unsure files guess E01 (before) and E04 (after)
    for e, ep, guess in zip(plan["entries"], [2, 3, 0, 0], ["S01E02", "S01E03", "S01E01", "S01E04"]):
        e["imdb_episode"] = [1, ep] if e["confident"] else None
        e["best_guess"] = guess
    next((state / "plans").glob(f"*{pid}.json")).write_text(json.dumps(plan))
    text = call(client, "unidentified")
    assert text.count("LIKELY AN EPISODE") == 2 and "probably an extra" not in text


def staged(tmp, name, root="movies", n=1):
    d = tmp / "staging" / root / name
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"f{i}.mkv").write_bytes(b"x")
    return d


def test_queue_sorts_folders_by_what_to_do_about_them(env, client):
    tmp, state, ctx, _ = env
    ctx.folders = [tmp / "staging" / "tv", tmp / "staging" / "movies"]
    unprocessed = staged(tmp, "Show (2020)", "tv", 2)
    film = staged(tmp, "Zoolander (2001)")
    unclear = staged(tmp, "Community Season 1 disc 2 (2009)")
    ready_disc = staged(tmp, "Mad Men (2007)", "tv", 2)
    old = staged(tmp, "Barbie-in-Rock-N-Royals (2015)_1", n=2)
    entries = [dict(file=str(ready_disc / "f0.mkv"), target=str(tmp / "shows" / "S01E01.mkv"),
                    confident=True, status="identified", episode="S01E01", score=0.999, size=1,
                    calibration_id=None, imdb_episode=[1, 1], warnings=[], error=None)]
    pid = cache.save_mapping("watch", "Mad Men", entries, state_dir=state, source_dir=str(ready_disc))
    # a disc already filed: an applied plan moved one file out of `old`, two others remain
    done = [dict(file=str(old / "gone.mkv"), target=str(tmp / "x.mkv"), confident=True, size=1)]
    oid = cache.save_mapping("watch", "30 Rock", done, state_dir=state, source_dir=str(old))
    plan = cache.load_mapping(oid, state)
    plan["applied"] = {"at": "t", "results": [{"file": str(old / "gone.mkv"), "target": "x",
                                               "renamed": True, "reason": ""}]}
    next((state / "plans").glob(f"*{oid}.json")).write_text(json.dumps(plan))
    (state / "state.json").write_text(json.dumps({
        str(film): {"outcome": "triage_movie"}, str(unclear): {"outcome": "triage_review"}}))
    text = call(client, "queue")
    assert text.startswith("Summary: ")
    assert "1 plan(s) ready to apply (1 files)" in text and "2 folder(s) need a person" in text
    assert "1 movie(s) waiting for you" in text and "1 folder(s) of leftovers (ignorable)" in text
    needs = text.split("Needs a person:")[1].split("Movies waiting")[0]
    assert "Show (2020)" in needs and "not processed yet" in needs
    assert "Community Season 1 disc 2" in needs and "could not tell movie from TV" in needs
    assert "Zoolander" not in needs and "Barbie" not in needs and "Mad Men (2007)" not in needs
    assert f"plan {pid} (Mad Men): 1 file(s) ready, from 'Mad Men (2007)'" in text
    assert "Movies waiting for you to file (diskid doesn't file movies): Zoolander (2001)" in text
    assert "Leftovers from discs already filed (1 folders, 2 files" in text and "Barbie" in text


def test_queue_keeps_arm_job_stages_apart(env, client):
    import sqlite3
    tmp, state, ctx, _ = env
    db = tmp / "arm.db"
    con = sqlite3.connect(db)
    con.execute("create table job (job_id integer, title text, label text, status text, "
                "no_of_titles integer, errors text, path text)")
    con.executemany("insert into job values (?,?,?,?,?,?,?)", [
        (1, "T", "Old_Disc", "success", 3, None, "/x/old"),
        (2, "T", "Mad_Men_Disc_4", "transcoding", 4, None, "/x/mm"),
        (3, "T", "Stranger_Than_Fiction", "transcoding", 4, None, "/x/stf"),
        (4, "T", "OITNB_Disc_1", "ripping", 4, None, "/x/oitnb"),
    ])
    con.commit()
    con.close()
    ctx.arm_db = db
    text = call(client, "queue")
    assert "1 ripping | 2 transcoding" in text
    ripping = [l for l in text.splitlines() if l.strip().startswith("Ripping now")]
    trans = [l for l in text.splitlines() if l.strip().startswith("Transcoding")]
    assert len(ripping) == 1 and "OITNB_Disc_1" in ripping[0] and "Mad_Men" not in ripping[0]
    assert len(trans) == 1 and "Mad_Men_Disc_4" in trans[0] and "Stranger_Than_Fiction" in trans[0]
    assert "the disc is already out" in trans[0]
    assert "Last finished: job 1 Old_Disc - success" in text
    assert "Ready to apply: nothing" in call(client, "queue")
