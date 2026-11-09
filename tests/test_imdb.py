import io
import json
from contextlib import contextmanager

from diskid.imdb import resolve_show, resolve_show_candidates

FUTURAMA_SUGGESTIONS = {
    "d": [
        {"id": "tt0149460", "l": "Futurama", "qid": "tvSeries", "y": 1999},
        {"id": "tt0471711", "l": "Futurama: Bender's Big Score", "qid": "video", "y": 2007},
        {"id": "tt8964678", "l": "Futurama", "qid": "tvSeries", "y": 1964},
    ]
}


def _fake_urlopen(payload):
    @contextmanager
    def opener(url, timeout=10):
        yield io.BytesIO(json.dumps(payload).encode())
    return opener


def test_resolve_show_candidates_filters_non_series(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(FUTURAMA_SUGGESTIONS))
    candidates = resolve_show_candidates("futurama")
    assert candidates == [(149460, "Futurama"), (8964678, "Futurama")]


def test_resolve_show_candidates_prefers_year_match(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(FUTURAMA_SUGGESTIONS))
    candidates = resolve_show_candidates("futurama", year=1964)
    assert candidates[0] == (8964678, "Futurama")


def test_resolve_show_returns_first_candidate(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(FUTURAMA_SUGGESTIONS))
    assert resolve_show("futurama") == (149460, "Futurama")


def test_resolve_show_raises_when_no_series(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen({"d": []}))
    try:
        resolve_show("nonexistent")
        assert False, "expected LookupError"
    except LookupError:
        pass
