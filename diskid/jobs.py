"""Background re-identification of single files ("rerun this one with more samples").

A rerun transcribes up to a few dozen clips, which takes minutes -- longer than
most tool-calling clients will wait on one call. So `submit` returns a job id at
once and a single worker thread does the work (one at a time: the speech-to-text
server and the CPU are shared). Jobs live in memory only; what matters is kept
in the *plan* each rerun writes, so a restart loses nothing but the progress.

A rerun never moves anything. It writes a new dry-run plan holding one entry,
and its observation goes to the calibration log like any other identification.
"""

from __future__ import annotations

import dataclasses
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import cache, queries

MIN_CLIPS_CAP = 10     # a rerun samples at least this many (or `clips`, if fewer)
CLIPS_RANGE = (5, 40)  # accepted values for `clips`
KEEP_JOBS = 50


class RerunManager:
    def __init__(self, identifier, state_dir: Path):
        self.identifier = identifier
        self.state_dir = state_dir
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diskid-rerun")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._counter = itertools.count(1)

    def submit(self, fid: str, clips: int = 20) -> dict[str, Any]:
        """Queue a rerun of the file `fid` names. Raises `queries.LookupFailed` if it can't."""
        lo, hi = CLIPS_RANGE
        if not lo <= clips <= hi:
            raise queries.LookupFailed(f"clips must be between {lo} and {hi}")
        plan, _, entry = queries.get_entry(self.state_dir, fid)
        path = Path(entry["file"])
        if not path.is_file():
            raise queries.LookupFailed(f"{path.name} is no longer where the plan found it")
        if not plan.get("show"):
            raise queries.LookupFailed("that plan has no show name to match against")
        with self._lock:
            job_id = f"j{next(self._counter)}"
            job = dict(id=job_id, file_id=fid, name=path.name, show=plan["show"],
                       clips=clips, status="queued", progress="", submitted=time.time(),
                       plan_id=None, new_file_id=None, summary="")
            self._jobs[job_id] = job
            for old in list(self._jobs)[:-KEEP_JOBS]:
                del self._jobs[old]
        self._pool.submit(self._run, job, plan, path)
        return dict(job)

    def _run(self, job: dict, plan: dict, path: Path) -> None:
        job["status"] = "running"
        ident = self.identifier
        original = ident.config
        try:
            ident.config = dataclasses.replace(
                original, max_clips=job["clips"], min_clips=min(MIN_CLIPS_CAP, job["clips"]))

            def progress(ev) -> None:
                if getattr(ev, "stage", "") == "transcribing":
                    job["progress"] = f"clip {ev.clip}/{ev.max_clips}"
            ident.progress = progress
            res = ident.identify_file(path, job["show"])
            entry = cache.entry_for(res)
            plan_id = cache.save_mapping(
                "mcp-rerun", job["show"], [entry], state_dir=self.state_dir, dry_run=True,
                show_imdb=plan.get("show_imdb"),
                source_dir=plan.get("source_dir") or str(path.parent))
            job.update(plan_id=plan_id, new_file_id=queries.file_id(plan_id, 0))
            if res.identified:
                job["summary"] = (f"identified as {res.episode.code} "
                                  f"({entry['score'] or 0:.3f}) after {res.clips_used} clips")
            else:
                best = res.best
                job["summary"] = ("still not identified after "
                                  f"{res.clips_used} clips; best guess "
                                  + (f"S{best.season:02d}E{best.episode:02d} ({best.score:.3f})"
                                     if best else "none"))
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 -- reported to the caller, not raised
            job.update(status="failed", summary=f"{type(e).__name__}: {e}")
        finally:
            ident.config = original
            ident.progress = None

    def status(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id.strip())
        if job is None:
            raise queries.LookupFailed(f"no job {job_id!r} (jobs are forgotten on restart)")
        return dict(job)

    def active(self) -> list[dict[str, Any]]:
        return [dict(j) for j in self._jobs.values() if j["status"] in ("queued", "running")]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
