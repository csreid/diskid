from pathlib import Path
from types import SimpleNamespace

from diskid.audio import probe_title


def test_probe_title_returns_tag(monkeypatch):
    import diskid.audio as audio
    monkeypatch.setattr(
        audio.subprocess, "run",
        lambda *a, **k: SimpleNamespace(stdout="Breaking Bad\n"),
    )
    assert probe_title(Path("video.mkv")) == "Breaking Bad"


def test_probe_title_returns_none_when_untagged(monkeypatch):
    import diskid.audio as audio
    monkeypatch.setattr(
        audio.subprocess, "run",
        lambda *a, **k: SimpleNamespace(stdout=""),
    )
    assert probe_title(Path("video.mkv")) is None
