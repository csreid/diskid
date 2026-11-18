"""Read-side views of diskid's durable state: plans, per-disc state, ARM jobs.

Everything here only *reads* (plus `plan_moves`, which inspects the filesystem).
It exists so the MCP server -- and anything else that wants to answer "what's in
the queue?" -- doesn't re-derive it from files. Output is compact text, meant to
be read by a small language model as well as a person.

A *file id* names one entry of one plan: `<plan id>.<index>`, e.g. `5ee1cb8c.3`.
Plans are immutable once written, so an id keeps pointing at the same entry; a
rerun of a file makes a new plan and so a new id.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from . import arm, cache

#: How many of the newest plans `uncertain_files` looks through.
RECENT_PLANS = 60

_FILE_ID = re.compile(r"^([0-9a-f]{8})\.(\d+)$")


class LookupFailed(ValueError):
    """An id that doesn't name anything (message is written for the caller to read)."""


def file_id(plan_id: str, index: int) -> str:
    return f"{plan_id}.{index}"


def parse_file_id(text: str) -> tuple[str, int]:
    m = _FILE_ID.match(text.strip().removeprefix("file:"))
    if not m:
        raise LookupFailed(f"{text!r} is not a file id (expected like 5ee1cb8c.3)")
    return m[1], int(m[2])


def get_plan(state_dir: Path, plan_id: str) -> dict:
    plan_id = plan_id.strip().removeprefix("plan:")
    plan = cache.load_mapping(plan_id, state_dir) if plan_id.isalnum() else None
    if plan is None:
        raise LookupFailed(f"no plan {plan_id!r} (see queue / unidentified)")
    return plan


def get_entry(state_dir: Path, fid: str) -> tuple[dict, int, dict]:
    plan_id, idx = parse_file_id(fid)
    plan = get_plan(state_dir, plan_id)
    if idx >= len(plan["entries"]):
        raise LookupFailed(f"plan {plan_id} has {len(plan['entries'])} files; no file {idx}")
    return plan, idx, plan["entries"][idx]


def _name(entry: dict) -> str:
    return Path(entry["file"]).name


def _score(entry: dict) -> str:
    s = entry.get("score") if entry.get("confident") else entry.get("best_score")
    return f"{s:.3f}" if s is not None else "-"


def describe_plan(plan: dict) -> str:
    """One line per file: id, name, verdict, target; plus a header and counts."""
    es = plan["entries"]
    ok = [e for e in es if e["confident"]]
    state = "applied" if plan.get("applied") else "NOT applied"
    lines = [f"plan {plan['id']}  {plan['show'] or '?'}  {len(ok)}/{len(es)} identified  "
             f"({state}, made {plan['created'][:16].replace('T', ' ')})"]
    for i, e in enumerate(es):
        fid = file_id(plan["id"], i)
        if e["confident"]:
            tgt = Path(e["target"]).name if e.get("target") else "?"
            lines.append(f"  {fid}  {_name(e)}  ->  {e['episode']} ({_score(e)})  {tgt}")
        else:
            guess = e.get("best_guess") or "?"
            lines.append(f"  {fid}  {_name(e)}  ->  UNSURE, best guess {guess} ({_score(e)})")
    return "\n".join(lines)


def recent_plans(state_dir: Path, limit: int = RECENT_PLANS) -> list[dict]:
    """Newest `limit` plans, oldest first, skipping ones marked discarded."""
    plans = [p for p in cache.list_plans(state_dir) if not p.get("discarded")]
    return plans[-limit:]


def uncertain_files(state_dir: Path, limit: int = 20) -> list[dict]:
    """Files whose *latest* result is unsure and that still exist on disk.

    Newer plans supersede older ones for the same path, so a file that was
    rerun, identified and moved, or deleted drops off the list.
    """
    latest: dict[str, tuple[dict, int, dict]] = {}
    for plan in recent_plans(state_dir):
        for i, e in enumerate(plan["entries"]):
            latest[e["file"]] = (plan, i, e)
    out = []
    for path, (plan, i, e) in latest.items():
        if e["confident"] or e.get("error") or not os.path.isfile(path):
            continue
        out.append(dict(id=file_id(plan["id"], i), name=_name(e), disc=Path(path).parent.name,
                        show=plan["show"], best_guess=e.get("best_guess"),
                        best_score=e.get("best_score"), made=plan["created"]))
    out.sort(key=lambda r: r["made"], reverse=True)
    return out[:limit]


def describe_uncertain(rows: list[dict]) -> str:
    if not rows:
        return "No unidentified files waiting."
    lines = [f"{len(rows)} file(s) not identified (newest first):"]
    for r in rows:
        score = f"{r['best_score']:.3f}" if r["best_score"] is not None else "-"
        lines.append(f"  {r['id']}  {r['name']}  in '{r['disc']}'  show={r['show']}  "
                     f"best guess {r['best_guess'] or '?'} ({score})")
    lines.append("Rerun one with more samples: rerun(file_id).")
    return "\n".join(lines)


#: What the watcher's recorded outcomes mean, in words that can't be mistaken for a workflow step.
_OUTCOME_WORDS = {
    "triage_movie": "looks like a movie, not TV: left alone, nothing to do",
    "triage_review": "could not tell movie from TV: left alone, needs a person to look",
    "no_show": "could not work out which show this is: needs a person to say",
}


def _disc_state(state: dict, folder: Path) -> dict:
    return state.get(str(folder), {})


def ready_to_apply(state_dir: Path, min_score: float, limit: int = 10) -> list[str]:
    """Lines about plans that have files which could be moved right now (newest first).

    Uses the same checks as `plan_moves`, so stale plans (sources gone, targets
    already there) never show up.
    """
    lines = []
    for plan in reversed(recent_plans(state_dir)):
        info = plan_moves(plan, min_score)
        n = len(info["moves"])
        if info["blocked"]:
            lines.append(f"  plan {plan['id']} ({plan['show']}): {len(info['blocked'])} file(s) "
                         "score too low to apply; rerun them first")
        elif n:
            lines.append(f"  plan {plan['id']} ({plan['show']}): {n} file(s) ready "
                         f"-> preview_apply('{plan['id']}')")
        if len(lines) >= limit:
            break
    return lines


def queue(state_dir: Path, folders: list[Path], arm_db: Path | None = None,
          min_score: float = 0.99) -> str:
    """Disc folders on disk, each with its state, plus the latest ARM jobs."""
    try:
        state = json.loads((state_dir / "state.json").read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    lines = []
    for root in folders:
        try:
            discs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            continue
        for d in discs:
            try:
                n = sum(1 for f in d.rglob("*") if f.is_file())
            except OSError:
                n = 0
            if n == 0:
                continue
            s = _disc_state(state, d)
            outcome = s.get("outcome")
            if outcome == "done":
                what = f"checked: {s.get('identified')} of {s.get('total')} files identified (plan {s.get('plan')})"
                if s.get("renamed"):
                    what += f", {s['renamed']} already moved"
            elif outcome in _OUTCOME_WORDS:
                what = _OUTCOME_WORDS[outcome]
            elif outcome:
                what = outcome.replace("_", " ")
            else:
                what = "not processed yet (still being ripped, or settling)"
            lines.append(f"  [{root.name}] {d.name}  {n} files  {what}")
    head = [f"{len(lines)} disc folder(s) in staging:"] if lines else ["No disc folders in staging."]
    out = head + lines
    ready = ready_to_apply(state_dir, min_score)
    out.append("Ready to apply (identified files that can be moved into the library now):"
               if ready else "Ready to apply: nothing. No plan has files waiting to be moved.")
    out += ready
    if arm_db:
        jobs = arm.recent_jobs(arm_db, 3)
        if jobs:
            out.append("Latest ARM jobs:")
            for j in jobs:
                extra = f", errors: {j['errors']}" if j.get("errors") else ""
                out.append(f"  job {j['job_id']}  {j.get('label') or j.get('title')}  "
                           f"{j['status']}  ({j.get('no_of_titles')} titles){extra}")
    return "\n".join(out)


def plan_moves(plan: dict, min_score: float) -> dict[str, Any]:
    """What applying `plan` would do, checked against the disk right now.

    Returns `moves` (source, target), `problems` (reasons a file won't move; apply
    skips these safely), `blocked` (confident files scoring below `min_score`: an
    apply must refuse the whole plan while any exist), and `signature` -- a fingerprint of the moves, so
    an apply can insist on the exact set that was previewed.
    """
    moves, problems, blocked = [], [], []
    done = {r["file"] for r in (plan.get("applied") or {}).get("results", []) if r["renamed"]}
    for e in plan["entries"]:
        if not e["confident"] or e["file"] in done:
            continue
        src, tgt = e["file"], e.get("target")
        name = _name(e)
        if not tgt:
            problems.append(f"{name}: no target path")
        elif not os.path.isfile(src):
            problems.append(f"{name}: source file is gone")
        elif os.path.exists(tgt):
            problems.append(f"{name}: target already exists ({Path(tgt).name})")
        elif e.get("size") is not None and os.path.getsize(src) != e["size"]:
            problems.append(f"{name}: file changed since the plan was made")
        else:
            moves.append((src, tgt))
        if (e.get("score") or 0) < min_score:
            blocked.append(f"{name}: score {e.get('score') or 0:.3f} is below {min_score}")
    return {"moves": moves, "problems": problems, "blocked": blocked,
            "signature": json.dumps([plan["id"], moves], sort_keys=True)}
