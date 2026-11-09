"""Watch a directory of ripped discs (ARM's COMPLETED_PATH) and identify new ones.

ARM has no post-job hook, so this polls: each immediate subdirectory is one
disc (`Title (Year)/title_tNN.mkv`). A disc is processed once it has been quiet
(no file created/modified) for `settle` seconds, then renamed in place.

State is kept in `state.json` so a disc is not reprocessed after a restart. It
is reprocessed only if its contents change (new/removed/resized files), e.g.
ARM re-rips into the same folder. Files that could not be identified are left
untouched; a per-disc JSON report is written for review.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .identify import DiscResult, Identifier, ShowNotFoundError, apply_renames

log = logging.getLogger("diskid.watch")

#: Seconds to wait before retrying a disc that hit a transient error
#: (network, ffmpeg missing, ...).
RETRY_DELAY = 900.0


class StopRequested(Exception):
    """Raised from the Identifier's progress callback to abort promptly."""


def default_state_dir() -> Path:
    """$STATE_DIRECTORY (set by systemd's StateDirectory=), else XDG state."""
    env = os.environ.get("STATE_DIRECTORY")
    if env:
        return Path(env.split(":")[0])
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "diskid"


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file())


def fingerprint(directory: Path) -> str:
    """Hash of the disc's file names and sizes."""
    h = hashlib.sha1()
    for p in _files(directory):
        h.update(f"{p.relative_to(directory)}\0{p.stat().st_size}\n".encode())
    return h.hexdigest()


def last_activity(directory: Path) -> float:
    """Newest mtime among the directory tree's files and directories."""
    newest = directory.stat().st_mtime
    for p in directory.rglob("*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except FileNotFoundError:  # vanished mid-scan: it is certainly active
            return time.time()
    return newest


@dataclass
class WatchConfig:
    watch_dir: Path
    state_dir: Path
    interval: float = 30.0
    """Seconds between scans."""
    settle: float = 120.0
    """A disc must be unmodified this long before it is processed."""
    dry_run: bool = False
    """Identify and report, but never rename."""


class DiscWatcher:
    def __init__(self, identifier: Identifier, config: WatchConfig):
        self.identifier = identifier
        self.config = config
        self._stop = False
        self._retry_after: dict[str, float] = {}
        config.state_dir.mkdir(parents=True, exist_ok=True)
        (config.state_dir / "reports").mkdir(exist_ok=True)
        self._state_path = config.state_dir / "state.json"
        self._state: dict[str, dict] = self._load_state()

    # -- state -----------------------------------------------------------

    def _load_state(self) -> dict[str, dict]:
        try:
            return json.loads(self._state_path.read_text())
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as e:
            log.warning("ignoring unreadable state file %s: %s", self._state_path, e)
            return {}

    def _save_state(self) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        tmp.replace(self._state_path)

    def _record(self, disc_dir: Path, **info) -> None:
        self._state[str(disc_dir)] = {
            "fingerprint": fingerprint(disc_dir),  # post-rename contents
            "processed": datetime.now(timezone.utc).isoformat(),
            **info,
        }
        self._save_state()

    # -- control ---------------------------------------------------------

    def stop(self) -> None:
        """Request shutdown; aborts any in-flight disc at the next clip."""
        self._stop = True

    def check_stop(self) -> None:
        if self._stop:
            raise StopRequested

    # -- work ------------------------------------------------------------

    def pending(self, now: float | None = None) -> list[Path]:
        """Disc folders that are settled and new/changed since last processed."""
        now = time.time() if now is None else now
        out = []
        for d in sorted(p for p in self.config.watch_dir.iterdir()
                        if p.is_dir() and not p.name.startswith(".")):
            key = str(d)
            if self._retry_after.get(key, 0) > now:
                continue
            try:
                if not _files(d):
                    continue
                if now - last_activity(d) < self.config.settle:
                    continue
                if self._state.get(key, {}).get("fingerprint") == fingerprint(d):
                    continue
            except FileNotFoundError:
                continue  # moved away while scanning
            out.append(d)
        return out

    def process_disc(self, disc_dir: Path) -> DiscResult | None:
        """Identify (and unless dry-run, rename) one disc; records state."""
        log.info("identifying %s", disc_dir.name)
        try:
            disc = self.identifier.identify_disc(disc_dir)
        except ShowNotFoundError as e:
            log.warning("%s: %s (not a series, or not in the archive); leaving untouched",
                        disc_dir.name, e)
            self._record(disc_dir, outcome="no_show")
            return None
        except StopRequested:
            raise
        except Exception as e:  # transient (network, ffmpeg...): retry later
            log.error("%s: %s: %s; will retry in %.0f min", disc_dir.name,
                      type(e).__name__, e, RETRY_DELAY / 60)
            self._retry_after[str(disc_dir)] = time.time() + RETRY_DELAY
            return None

        renamed = 0
        if not self.config.dry_run:
            for out in apply_renames(disc.identified):
                if out.renamed:
                    renamed += 1
                    if out.source != out.target:
                        log.info("renamed %s -> %s", out.source.name, out.target.name)
                else:
                    log.warning("not renamed: %s (%s)", out.source.name, out.reason)
        else:
            for f in disc.identified:
                log.info("[dry run] %s -> %s", f.path.name, f.target_name)
        for f in disc.unidentified:
            why = f.error or (f.best and f"best S{f.best.season:02d}E{f.best.episode:02d} "
                                         f"p={f.best.score:.3f}") or "no match"
            log.info("left alone: %s (%s)", f.path.name, f.status + ": " + why)

        report = self.config.state_dir / "reports" / (
            f"{disc_dir.name}-{datetime.now():%Y%m%d-%H%M%S}.json")
        report.write_text(json.dumps(disc.to_dict(), indent=2))
        log.info("%s: %s, %d/%d identified, %d renamed (report: %s)", disc_dir.name,
                 disc.show.title if disc.show else "?", len(disc.identified),
                 len(disc.files), renamed, report)
        self._record(
            disc_dir, outcome="done", show=disc.show.title if disc.show else None,
            identified=len(disc.identified), total=len(disc.files), renamed=renamed,
            dry_run=self.config.dry_run,
        )
        return disc

    def scan_once(self) -> int:
        """Process every pending disc; returns how many were handled."""
        n = 0
        for d in self.pending():
            self.check_stop()
            if self.process_disc(d) is not None:
                n += 1
        return n

    def run_forever(self) -> None:
        log.info("watching %s (every %.0fs, settle %.0fs%s)", self.config.watch_dir,
                 self.config.interval, self.config.settle,
                 ", DRY RUN" if self.config.dry_run else "")
        try:
            while not self._stop:
                try:
                    self.scan_once()
                except FileNotFoundError:
                    log.warning("watch dir %s is missing; waiting", self.config.watch_dir)
                end = time.monotonic() + self.config.interval
                while not self._stop and time.monotonic() < end:
                    time.sleep(min(1.0, self.config.interval))
        except StopRequested:
            pass
        log.info("stopped")
