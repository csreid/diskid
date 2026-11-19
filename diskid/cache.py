"""Durable identification plans: what diskid decided, applicable later.

A *plan* is a JSON file listing each file's proposed destination, produced by
dry runs (`identify`, `identify-disc`, `watch --dry-run`) and, for audit, by
live runs too. `diskid apply <id|path>` carries a plan out without re-running
whisper/matching. Plans live in `<state dir>/plans/` (`$STATE_DIRECTORY` or
`~/.local/state/diskid`), not in a temp directory.

Schema (version 1): id, created, command, show, show_imdb, source_dir,
dry_run, entries[], applied (null until applied). Each entry has `file`,
`target`, `confident`, `episode` ("S01E02"), `score`, `size` (bytes when
planned; an apply refuses a file that has changed since) and `status`.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import arm, calibration
from .eventlog import EventLog
from .fsutil import prune_empty_dirs
from .identify import rename_to_target

PLAN_VERSION = 1


def default_state_dir() -> Path:
    """$STATE_DIRECTORY (set by systemd's StateDirectory=), else XDG state."""
    env = os.environ.get("STATE_DIRECTORY")
    if env:
        return Path(env.split(":")[0])
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "diskid"


def plans_dir(state_dir: Path | None = None) -> Path:
    d = (state_dir or default_state_dir()) / "plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


def entry_for(res: Any) -> dict:
    """Plan entry for a `FileResult`."""
    try:
        size = res.path.stat().st_size
    except OSError:
        size = None
    score = None
    if res.episode:
        score = next((c.score for c in res.candidates
                      if c.key == (res.episode.imdb_season, res.episode.imdb_episode)), None)
    return {
        "file": str(res.path.resolve()),
        "target": str(res.target_path.resolve()) if res.target_path else None,
        "confident": res.identified,
        "status": res.status,
        "episode": res.episode.code if res.episode else None,
        "episode_title": res.episode.title if res.episode else None,
        "score": score,
        "best_guess": (f"S{res.best.season:02d}E{res.best.episode:02d}" if res.best else None),
        "best_score": res.best.score if res.best else None,
        "size": size,
        "calibration_id": getattr(res, "calibration_id", None),
        "imdb_episode": ([res.episode.imdb_season, res.episode.imdb_episode]
                         if res.episode else None),
        "warnings": list(res.warnings),
        "error": res.error,
    }


def save_mapping(
    command: str, show: str, entries: list[dict], *,
    state_dir: Path | None = None, dry_run: bool = True, **extra: Any,
) -> str:
    """Write a plan; returns its short id."""
    plan_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)
    plan = {
        "version": PLAN_VERSION, "id": plan_id, "created": now.isoformat(timespec="seconds"),
        "command": command, "show": show, "dry_run": dry_run, **extra,
        "entries": entries, "applied": None,
    }
    # microseconds keep plans made in the same second in creation order
    path = plans_dir(state_dir) / f"{now:%Y%m%d-%H%M%S}{now.microsecond:06d}-{plan_id}.json"
    plan["_path"] = str(path)
    _write(path, plan)
    return plan_id


def _write(path: Path, plan: dict) -> None:
    data = {k: v for k, v in plan.items() if k != "_path"}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_mapping(ref: str, state_dir: Path | None = None) -> dict | None:
    """Load a plan by short id or by path to a plan file; None if not found.

    The returned dict carries a private `_path` key (where it was loaded from).
    """
    path = Path(ref)
    if not (path.suffix == ".json" and path.is_file()):
        matches = sorted(plans_dir(state_dir).glob(f"*-{ref}.json")) if ref.isalnum() else []
        if not matches:
            return None
        path = matches[-1]
    plan = json.loads(path.read_text())
    plan["_path"] = str(path)
    return plan


def list_plans(state_dir: Path | None = None) -> list[dict]:
    """Every plan, oldest first."""
    out = []
    for p in sorted(plans_dir(state_dir).glob("*.json")):
        try:
            plan = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        plan["_path"] = str(p)
        out.append(plan)
    return out


def _prune_source(plan: dict, results: list[dict], arm_db: Path | None) -> list[Path]:
    """Remove the plan's disc folder's empty folders after files moved out of it."""
    src = plan.get("source_dir")
    if not src or not any(r["renamed"] for r in results) or not Path(src).is_dir():
        return []
    root = Path(src)
    if arm_db is not None and arm.job_active(arm_db, root):
        return []
    return prune_empty_dirs(root)


def apply_plan(plan: dict, events: EventLog | None = None,
               calibration_log: Path | None = None, arm_db: Path | None = None) -> list[dict]:
    """Carry out a plan's confident entries; record the outcome in the plan file.

    Never overwrites; refuses a source whose size changed since planning. With
    `calibration_log`, a moved file's observation is labelled "plan_applied".
    Entries an earlier apply already moved are skipped and keep their recorded result.
    Afterwards the plan's `source_dir` (the disc folder) is cleaned of empty folders
    (`rmdir` only; it is removed too if that leaves it empty) unless `arm_db` says
    ARM still has a running job for it -- ARM creates a job's folder up front and
    fills it when the transcode ends, so an empty folder can be one in use.
    Returns one result dict per entry: file, target, renamed, reason.
    """
    events = events or EventLog(None)
    results = []
    # A retry must not forget (or re-log as failures) files an earlier run moved.
    done = {r["file"]: r for r in (plan.get("applied") or {}).get("results", []) if r["renamed"]}
    for e in plan["entries"]:
        if e["file"] in done:
            results.append(done[e["file"]])
            continue
        file = Path(e["file"])
        target = Path(e["target"]) if e.get("target") else None
        if not e.get("confident") or target is None:
            renamed, reason = False, "not confident"
        elif e.get("size") is not None and file.is_file() and file.stat().st_size != e["size"]:
            renamed, reason = False, "source changed since the plan was made"
        else:
            new, reason = rename_to_target(file, target)
            renamed = new is not None
        if renamed and e.get("calibration_id") and e.get("imdb_episode"):
            calibration.write_label(calibration_log, e["calibration_id"],
                                    tuple(e["imdb_episode"]), "plan_applied", plan=plan["id"])
        if e.get("confident"):
            events.write("file_moved" if renamed else "file_not_moved",
                         plan=plan["id"], source=str(file),
                         target=str(target) if target else None, reason=reason or None)
        results.append({"file": str(file), "target": str(target) if target else None,
                        "renamed": renamed, "reason": reason})
    removed = _prune_source(plan, results, arm_db)
    if removed:
        events.write("dirs_pruned", plan=plan["id"], removed=[str(p) for p in removed])
    plan["applied"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "results": results}
    plan["dry_run"] = False
    path = plan.get("_path")
    if path:
        _write(Path(path), plan)
    events.write("plan_applied", plan=plan["id"],
                 moved=sum(r["renamed"] for r in results), total=len(results))
    return results
