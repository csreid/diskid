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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import arm, audio, cache, tmdb, triage
from .cache import default_state_dir  # noqa: F401  (re-exported)
from .eventlog import EventLog
from .identify import DiscResult, Identifier, ShowNotFoundError

log = logging.getLogger("diskid.watch")

#: Seconds to wait before retrying a disc that hit a transient error
#: (network, ffmpeg missing, ...).
RETRY_DELAY = 900.0


class StopRequested(Exception):
    """Raised from the Identifier's progress callback to abort promptly."""


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


def prune_empty_dirs(root: Path) -> None:
    """Remove `root`'s empty subfolders, and `root` itself if that leaves it empty.

    Only ever calls `rmdir`, so a folder holding any file is left alone. Used
    after files are moved out to a different output directory.
    """
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


@dataclass
class WatchConfig:
    watch_dir: Path | list[Path]
    """Folder(s) whose immediate subfolders are discs. ARM writes
    `<COMPLETED_PATH>/<type>/<title>/`, so for ARM watch `<COMPLETED_PATH>/tv` and
    `<COMPLETED_PATH>/unidentified`. A folder that doesn't exist yet is skipped."""
    state_dir: Path
    interval: float = 30.0
    """Seconds between scans."""
    settle: float = 120.0
    """A disc must be unmodified this long before it is processed."""
    dry_run: bool = False
    """Identify and report, but never rename."""
    arm_db: Path | None = None
    """ARM's SQLite database; if set, a disc's show name hint comes from its label."""
    triage_dirs: list[Path] = field(default_factory=list)
    """Folders of discs ARM filed as *movies* (e.g. `<staging>/movies`). Each disc is
    checked first (see `diskid.triage`): only those that are really TV discs are
    identified; the rest are left untouched."""
    tmdb_api_key: str | None = None
    """For the claimed film's runtime; None falls back to $TMDB_APIKEY."""

    def __post_init__(self) -> None:
        dirs = self.watch_dir if isinstance(self.watch_dir, list) else [self.watch_dir]
        self.roots: list[Path] = [Path(d) for d in dirs]
        self.watch_dir = self.roots[0] if self.roots else Path(".")
        self.triage_dirs = [Path(d) for d in self.triage_dirs]


class DiscWatcher:
    def __init__(self, identifier: Identifier, config: WatchConfig):
        self.identifier = identifier
        self.config = config
        self._stop = False
        self._retry_after: dict[str, float] = {}
        config.state_dir.mkdir(parents=True, exist_ok=True)
        (config.state_dir / "reports").mkdir(exist_ok=True)
        self.events = EventLog(config.state_dir / "events.jsonl")
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

    def _disc_dirs(self) -> list[Path]:
        found: list[Path] = []
        for root in [*self.config.roots, *self.config.triage_dirs]:
            try:
                found += [p for p in root.iterdir()
                          if p.is_dir() and not p.name.startswith(".")]
            except FileNotFoundError:
                continue  # e.g. ARM hasn't created <staging>/tv yet
        return sorted(found)

    def pending(self, now: float | None = None) -> list[Path]:
        """Disc folders that are settled and new/changed since last processed."""
        now = time.time() if now is None else now
        out = []
        for d in self._disc_dirs():
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

    def _triage(self, disc_dir: Path) -> triage.Verdict:
        """Is a disc ARM filed as a movie really a TV disc? Probes durations only."""
        durations = []
        for f in _files(disc_dir):
            try:
                durations.append(audio.probe_duration(f))
            except Exception:  # not video / unreadable: ignore for classification
                continue
        claimed = None
        job = arm.job_for(self.config.arm_db, disc_dir) if self.config.arm_db else None
        key = self.config.tmdb_api_key or tmdb.api_key()
        if job and job.get("imdb_id") and key:
            claimed = tmdb.movie_runtime(key, str(job["imdb_id"]))
        return triage.classify(durations, claimed)

    def process_disc(self, disc_dir: Path) -> DiscResult | None:
        """Identify (and unless dry-run, move) one disc; records state.

        Discs under a triage folder are first classified; only TV ones go on.

        Every disc that gets a show is saved as a plan (`<state>/plans/`), dry
        run or not, so `diskid apply <id>` can carry it out later without
        re-running identification.
        """
        if disc_dir.parent in self.config.triage_dirs:
            verdict = self._triage(disc_dir)
            self.events.write("disc_triaged", disc=str(disc_dir), verdict=verdict.kind,
                              reason=verdict.reason)
            if verdict.kind != "tv":
                log.info("%s: %s (%s); leaving untouched", disc_dir.name, verdict.kind,
                         verdict.reason)
                if verdict.kind == "review":
                    log.warning("%s: needs a look: %s", disc_dir.name, verdict.reason)
                self._record(disc_dir, outcome=f"triage_{verdict.kind}")
                return None
            log.info("%s: looks like a TV disc, not a movie (%s)", disc_dir.name, verdict.reason)
        log.info("identifying %s", disc_dir.name)
        try:
            hint = arm.show_hint(self.config.arm_db, disc_dir) if self.config.arm_db else None
            if hint:
                log.info("%s: show hint %r from ARM label", disc_dir.name, hint)
            disc = (self.identifier.identify_disc(disc_dir, hint=hint) if hint
                    else self.identifier.identify_disc(disc_dir))
        except ShowNotFoundError as e:
            log.warning("%s: %s (not a series, or not in the archive); leaving untouched",
                        disc_dir.name, e)
            self.events.write("disc_no_show", disc=str(disc_dir), reason=str(e))
            self._record(disc_dir, outcome="no_show")
            return None
        except StopRequested:
            raise
        except Exception as e:  # transient (network, ffmpeg...): retry later
            log.error("%s: %s: %s; will retry in %.0f min", disc_dir.name,
                      type(e).__name__, e, RETRY_DELAY / 60)
            self.events.write("disc_error", disc=str(disc_dir),
                              error=f"{type(e).__name__}: {e}", retry_in=RETRY_DELAY)
            self._retry_after[str(disc_dir)] = time.time() + RETRY_DELAY
            return None

        plan_id = cache.save_mapping(
            "watch", disc.show.title if disc.show else "", [cache.entry_for(f) for f in disc.files],
            state_dir=self.config.state_dir, dry_run=self.config.dry_run,
            show_imdb=disc.show.imdb_id if disc.show else None, source_dir=str(disc_dir),
        )
        moved = 0
        if not self.config.dry_run:
            plan = cache.load_mapping(plan_id, self.config.state_dir)
            for r in cache.apply_plan(plan, self.events):
                if r["renamed"]:
                    moved += 1
                    if r["file"] != r["target"]:
                        log.info("moved %s -> %s", Path(r["file"]).name, r["target"])
                elif r["reason"] != "not confident":
                    log.warning("not moved: %s (%s)", Path(r["file"]).name, r["reason"])
            if moved:
                prune_empty_dirs(disc_dir)
        else:
            for f in disc.identified:
                log.info("[dry run] %s -> %s", f.path.name, f.target_rel)
        for f in disc.unidentified:
            why = f.error or (f.best and f"best S{f.best.season:02d}E{f.best.episode:02d} "
                                         f"p={f.best.score:.3f}") or "no match"
            log.info("left alone: %s (%s)", f.path.name, f.status + ": " + why)

        report = self.config.state_dir / "reports" / (
            f"{disc_dir.name}-{datetime.now():%Y%m%d-%H%M%S}.json")
        report.write_text(json.dumps(disc.to_dict(), indent=2))
        log.info("%s: %s, %d/%d identified, %d moved (plan: %s%s)", disc_dir.name,
                 disc.show.title if disc.show else "?", len(disc.identified),
                 len(disc.files), moved, plan_id,
                 f"; apply with: diskid apply {plan_id}" if self.config.dry_run else "")
        self.events.write(
            "disc_processed", disc=str(disc_dir), show=disc.show.title if disc.show else None,
            identified=len(disc.identified), total=len(disc.files), moved=moved,
            dry_run=self.config.dry_run, plan=plan_id, report=str(report),
        )
        self._record(
            disc_dir, outcome="done", show=disc.show.title if disc.show else None,
            identified=len(disc.identified), total=len(disc.files), renamed=moved,
            dry_run=self.config.dry_run, plan=plan_id,
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
        log.info("watching %s (every %.0fs, settle %.0fs%s)",
                 ", ".join(map(str, self.config.roots)),
                 self.config.interval, self.config.settle,
                 ", DRY RUN" if self.config.dry_run else "")
        try:
            while not self._stop:
                self.scan_once()
                end = time.monotonic() + self.config.interval
                while not self._stop and time.monotonic() < end:
                    time.sleep(min(1.0, self.config.interval))
        except StopRequested:
            pass
        log.info("stopped")
