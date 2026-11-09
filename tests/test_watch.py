import os
import time

from diskid.identify import (
    DiscResult, EpisodeRef, FileResult, Show, ShowNotFoundError,
)
from diskid.watch import DiscWatcher, WatchConfig, fingerprint

SHOW = Show(1, "Test Show")


class StubIdentifier:
    """identify_disc 'identifies' every file as consecutive episodes."""
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def identify_disc(self, directory, show=None):
        self.calls.append(directory)
        if self.fail:
            raise self.fail
        files = []
        for i, p in enumerate(sorted(q for q in directory.rglob("*") if q.is_file()), 1):
            ref = EpisodeRef(1, i, 1, i, "imdb")
            files.append(FileResult(
                path=p, status="identified", show=SHOW, episode=ref,
                target_name=f"Test Show - S01E{i:02d}{p.suffix}"))
        return DiscResult(directory, SHOW, files)


def make_disc(root, name="Test Show (2020)", files=("title_t00.mkv", "title_t01.mkv"), age=1000):
    d = root / name
    d.mkdir()
    for f in files:
        (d / f).write_bytes(b"x" * 10)
    old = time.time() - age
    for p in [*d.iterdir(), d]:
        os.utime(p, (old, old))
    return d


def watcher(tmp_path, ident, **cfg):
    watch = tmp_path / "completed"
    watch.mkdir(exist_ok=True)
    return watch, DiscWatcher(ident, WatchConfig(watch, tmp_path / "state", settle=60, **cfg))


def test_processes_settled_disc_once(tmp_path):
    ident = StubIdentifier()
    watch, w = watcher(tmp_path, ident)
    disc = make_disc(watch)
    assert w.scan_once() == 1
    assert sorted(p.name for p in disc.iterdir()) == [
        "Test Show - S01E01.mkv", "Test Show - S01E02.mkv"]
    assert w.scan_once() == 0 and len(ident.calls) == 1   # fingerprint unchanged
    assert list((tmp_path / "state" / "reports").glob("*.json"))


def test_state_survives_restart(tmp_path):
    ident = StubIdentifier()
    watch, w = watcher(tmp_path, ident)
    make_disc(watch)
    w.scan_once()
    _, w2 = watcher(tmp_path, ident)
    assert w2.scan_once() == 0 and len(ident.calls) == 1


def test_waits_for_disc_to_settle(tmp_path):
    ident = StubIdentifier()
    watch, w = watcher(tmp_path, ident)
    make_disc(watch, age=5)          # modified 5s ago, settle is 60
    assert w.scan_once() == 0 and not ident.calls


def test_reprocesses_when_contents_change(tmp_path):
    ident = StubIdentifier()
    watch, w = watcher(tmp_path, ident)
    disc = make_disc(watch)
    w.scan_once()
    new = disc / "title_t05.mkv"
    new.write_bytes(b"y")
    old = time.time() - 1000
    os.utime(new, (old, old)); os.utime(disc, (old, old))
    assert w.scan_once() == 1 and len(ident.calls) == 2


def test_dry_run_renames_nothing(tmp_path):
    watch, w = watcher(tmp_path, StubIdentifier(), dry_run=True)
    disc = make_disc(watch)
    before = fingerprint(disc)
    assert w.scan_once() == 1
    assert fingerprint(disc) == before


def test_no_show_is_recorded_not_retried(tmp_path):
    ident = StubIdentifier(fail=ShowNotFoundError("nope"))
    watch, w = watcher(tmp_path, ident)
    make_disc(watch, name="Some Movie (1999)")
    w.scan_once(); w.scan_once()
    assert len(ident.calls) == 1


def test_transient_error_retries_later_not_every_scan(tmp_path):
    ident = StubIdentifier(fail=OSError("network down"))
    watch, w = watcher(tmp_path, ident)
    make_disc(watch)
    w.scan_once(); w.scan_once()
    assert len(ident.calls) == 1            # backed off, not hot-looping
    assert not w._state                      # and not recorded as done


def test_ignores_hidden_dirs_and_loose_files(tmp_path):
    ident = StubIdentifier()
    watch, w = watcher(tmp_path, ident)
    make_disc(watch, name=".incomplete")
    (watch / "stray.mkv").write_bytes(b"x")
    assert w.scan_once() == 0 and not ident.calls
