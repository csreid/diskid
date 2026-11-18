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
            "queue", "unidentified", "show_plan", "rerun", "job_status",
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


def test_queue_shows_disc_state(env, client):
    tmp, state, *_ = env
    make_plan(tmp, state, [("a.mkv", True, 0.999)])
    text = call(client, "queue")
    assert "1 disc folder(s)" in text and "Disc" in text and "waiting" in text


def test_a_rejected_request_logs_why_but_never_the_token(env, caplog):
    _, _, ctx, _ = env
    app = mcp_server.make_app(ctx, host="0.0.0.0", port=8765, token="s3cret-value")
    with TestClient(app) as c, caplog.at_level("WARNING", logger="diskid.mcp"):
        c.get("/")
        c.post("/mcp", headers={"Authorization": "Bearer nope"}, json={})
    text = caplog.text
    assert "no Authorization header" in text and "does not match" in text
    assert "s3cret-value" not in text and "nope" not in text
