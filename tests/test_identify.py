"""Library API tests with whisper, ffprobe and the subtitle archive faked out."""

import json
import subprocess
from pathlib import Path

import pytest

from diskid import identify as ident_mod
from diskid import (
    Identifier, IdentifierConfig, Show, ShowNotFoundError, apply_rename,
)
from test_match import make_show, snippet_of

SHOW = Show(1234567, "Test Show")


@pytest.fixture
def docs():
    return make_show()


@pytest.fixture
def make_identifier(monkeypatch, docs):
    """Identifier whose 'transcripts' are looked up by filename -> episode key."""
    def build(transcripts: dict[str, tuple[int, int]], **cfg):
        cfg.setdefault("order", "imdb")
        ident = Identifier(IdentifierConfig(**cfg))
        ident._shows[SHOW.imdb_id] = ident_mod._ShowData(
            SHOW, docs, {k: 9000 + k[1] for k in docs})
        monkeypatch.setattr(ident_mod.audio, "probe_duration", lambda p: 1500.0)
        monkeypatch.setattr(
            ident, "_transcribe_clip",
            lambda video, start: snippet_of(docs, transcripts[video.name], 100 + int(start) % 200)
            if video.name in transcripts else "")
        return ident
    return build


def touch(tmp_path, *names):
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"x")
        paths.append(p)
    return paths


def test_identify_file(tmp_path, make_identifier):
    (video,) = touch(tmp_path, "title_t01.mkv")
    res = make_identifier({"title_t01.mkv": (1, 3)}).identify_file(video, SHOW)
    assert res.status == "identified" and res.identified
    assert (res.episode.season, res.episode.episode) == (1, 3)
    assert res.episode.order == "imdb" and res.episode.imdb_id == 9003
    assert res.target_name == "Test Show - S01E03.mkv"
    assert res.target_path == tmp_path / "Test Show - S01E03.mkv"
    assert res.candidates[0].key == (1, 3) and res.clips_used >= 1
    json.dumps(res.to_dict())  # fully serializable


def test_unknown_content_is_uncertain(tmp_path, make_identifier):
    (video,) = touch(tmp_path, "extra.mkv")
    res = make_identifier({}).identify_file(video, SHOW)  # empty transcripts
    assert res.status == "uncertain" and res.episode is None and res.target_name is None


def test_ffmpeg_failure_is_failed_result(tmp_path, make_identifier, monkeypatch):
    (video,) = touch(tmp_path, "bad.mkv")
    ident = make_identifier({})

    def boom(p):
        raise subprocess.CalledProcessError(1, "ffprobe")
    monkeypatch.setattr(ident_mod.audio, "probe_duration", boom)
    res = ident.identify_file(video, SHOW)
    assert res.status == "failed" and res.error


def test_missing_file_raises(tmp_path, make_identifier):
    with pytest.raises(FileNotFoundError):
        make_identifier({}).identify_file(tmp_path / "nope.mkv", SHOW)


def test_disc_assigns_distinct_episodes(tmp_path, make_identifier):
    touch(tmp_path, "a.mkv", "b.mkv", "dup.mkv", "junk.mkv")
    ident = make_identifier({
        "a.mkv": (1, 1), "b.mkv": (1, 2), "dup.mkv": (1, 2), 
    })
    disc = ident.identify_disc(tmp_path, SHOW)
    by = {f.path.name: f for f in disc.files}
    assert by["a.mkv"].episode.episode == 1
    assert sum(by[n].identified for n in ("b.mkv", "dup.mkv")) == 1
    loser = next(by[n] for n in ("b.mkv", "dup.mkv") if not by[n].identified)
    assert loser.status == "uncertain" and loser.warnings
    assert len(disc.identified) + len(disc.unidentified) == len(disc.files)
    json.dumps(disc.to_dict())


def test_disc_guesses_show_from_folder_name(tmp_path, make_identifier, monkeypatch):
    disc_dir = tmp_path / "Test Show (2020)"
    disc_dir.mkdir()
    touch(disc_dir, "a.mkv", "b.mkv")
    ident = make_identifier({"a.mkv": (1, 1), "b.mkv": (1, 2)})
    monkeypatch.setattr(ident_mod.imdb, "resolve_show_candidates",
                        lambda name, year=None: [(SHOW.imdb_id, SHOW.title)])
    monkeypatch.setattr(ident_mod.audio, "probe_title", lambda p: None)
    disc = ident.identify_disc(disc_dir)
    assert disc.show == SHOW and len(disc.identified) == 2


def test_disc_no_show_raises(tmp_path, make_identifier, monkeypatch):
    touch(tmp_path, "a.mkv")
    ident = make_identifier({})
    monkeypatch.setattr(ident_mod.imdb, "resolve_show_candidates", lambda n, y=None: [])
    monkeypatch.setattr(ident_mod.audio, "probe_title", lambda p: None)
    with pytest.raises(ShowNotFoundError):
        ident.identify_disc(tmp_path)


def test_apply_rename_never_overwrites(tmp_path, make_identifier):
    video, other = touch(tmp_path, "title_t01.mkv", "Test Show - S01E03.mkv")
    res = make_identifier({"title_t01.mkv": (1, 3)}).identify_file(video, SHOW)
    out = apply_rename(res)
    assert not out.renamed and out.reason == "target exists" and video.exists()
    other.unlink()
    out = apply_rename(res)
    assert out.renamed and not video.exists() and out.target.exists()


def test_apply_rename_requires_identified(tmp_path, make_identifier):
    (video,) = touch(tmp_path, "extra.mkv")
    out = apply_rename(make_identifier({}).identify_file(video, SHOW))
    assert not out.renamed and video.exists()


def test_config_validation():
    with pytest.raises(ValueError):
        IdentifierConfig(order="dvd")
    with pytest.raises(ValueError):
        IdentifierConfig(min_score=1.5)


def test_remote_stt_configured_never_loads_local_whisper(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("local whisper must not load")
    monkeypatch.setattr(ident_mod.stt, "load_model", boom)
    ident = Identifier(IdentifierConfig(stt_url="http://stt:8000/v1", model="medium"))
    assert ident.device == "remote"
    assert ident._remote.model == "Systran/faster-whisper-medium"
    assert ident._remote.url == "http://stt:8000/v1/audio/transcriptions"
