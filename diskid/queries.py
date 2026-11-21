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


def find_observation(calibration_log: Path | None, obs_id: str | None) -> dict | None:
    """The calibration `observation` record with this id (None if absent or unreadable)."""
    if not calibration_log or not obs_id:
        return None
    needle = f'"id": "{obs_id}"'
    found = None
    try:
        with calibration_log.open() as f:
            for line in f:
                if needle in line and '"observation"' in line:
                    found = json.loads(line)
    except (OSError, json.JSONDecodeError):
        return None
    return found


def _p(x: float | None) -> str:
    """A probability, readable at both ends: 0.998, 4.2e-05."""
    if x is None:
        return "-"
    return f"{x:.3f}" if x >= 0.001 else (f"{x:.1e}" if x > 0 else "0")


def _code(season: int, episode: int) -> str:
    return f"S{season:02d}E{episode:02d}"


def disc_dir_of(path: str | Path) -> Path:
    """The disc folder a ripped file belongs to. ARM puts every non-main title in an
    `extras/` subfolder whatever it contains, so that name says nothing about the file."""
    parent = Path(path).parent
    return parent.parent if parent.name.lower() == "extras" else parent


def disc_context(plans: list[dict]) -> dict[Path, dict[int, set[int]]]:
    """For each disc folder, the episodes (IMDb numbering) identified for it in ANY plan.

    Reruns write one-file plans with no neighbours, so a file's disc context has to
    come from every plan that touched the same disc, moved or not.
    """
    ctx: dict[Path, dict[int, set[int]]] = {}
    for plan in plans:
        for e in plan["entries"]:
            if e["confident"] and e.get("imdb_episode"):
                s_, ep = e["imdb_episode"]
                ctx.setdefault(disc_dir_of(e["file"]), {}).setdefault(s_, set()).add(ep)
    return ctx


def context_for(state_dir: Path) -> dict[Path, dict[int, set[int]]]:
    return disc_context([p for p in cache.list_plans(state_dir) if not p.get("discarded")])


def gap_hint(plan: dict, entry: dict, context: dict | None = None) -> str | None:
    """Does this unsure file's best guess fill a hole in the disc's identified episodes?

    Discs hold consecutive episodes, so a file that can't be matched but whose
    best guess is exactly a missing number is very likely that episode (the
    matcher's hallucination weakness makes such files fail unsure). Numbers are
    in IMDb numbering, as `best_guess` is.
    """
    guess = entry.get("best_guess")
    m = re.fullmatch(r"S(\d+)E(\d+)", guess or "")
    if not m or entry["confident"]:
        return None
    season, ep = int(m[1]), int(m[2])
    if context is None:
        context = disc_context([plan])
    have = sorted(context.get(disc_dir_of(entry["file"]), {}).get(season, ()))
    if not have:
        return None
    lo, hi = have[0], have[-1]
    missing = [n for n in range(lo, hi + 1) if n not in have]
    if ep in missing:
        return (f"fits a gap: this disc's other files cover {_code(season, lo)}-E{hi:02d} "
                f"except {', '.join(f'E{n:02d}' for n in missing)}, and the best guess is "
                f"E{ep:02d} - probably that episode")
    if ep in (lo - 1, hi + 1):
        side = "just before" if ep < lo else "just after"
        return (f"fits the disc's sequence: its other files cover {_code(season, lo)}-E{hi:02d} "
                f"and the best guess E{ep:02d} comes {side} that run (a disc's first or last "
                "episode) - probably that episode")
    return (f"does not fit a gap: the disc's other files cover {_code(season, lo)}-E{hi:02d}, "
            f"and {_code(season, ep)} is " + ("already taken by another file" if ep in have
                                           else "outside that range") + " - possibly an extra")


def describe_file(plan: dict, idx: int, entry: dict, obs: dict | None, detail: bool = False,
                  context: dict | None = None) -> str:
    """Everything known about one file's verdict: top matches, null probability, clips, gap hint."""
    name = _name(entry)
    head = f"{file_id(plan['id'], idx)}  {name}  show={plan['show']}"
    if obs and obs.get("duration"):
        mins = obs["duration"] / 60
        head += f"  length {int(mins)}:{int(obs['duration'] % 60):02d}"
    lines = [head]
    if entry["confident"]:
        lines.append(f"Verdict: identified as {entry['episode']} ({_p(entry.get('score'))}).")
    else:
        lines.append("Verdict: NOT identified (needs a score of 0.99 or more).")
    if not obs:
        lines.append("No recorded clip details for this file (identified before calibration "
                     "logging, or logging is off).")
        if entry.get("best_guess"):
            lines.append(f"Best guess {entry['best_guess']} ({_p(entry.get('best_score'))}).")
    else:
        lines.append(f"Used {obs['n_clips']} clips"
                     + (f" ({len(obs['clips']) - obs['n_clips']} silent)"
                        if len(obs["clips"]) > obs["n_clips"] else "") + ".")
        lines.append("Top matches (probability, aired-order numbers):")
        top = obs["ranked"]
        for i, (s_, e_, p_) in enumerate(top):
            rel = ""
            if i and top[0][2] > 0 and p_ > 0:
                rel = f"  ({top[0][2] / p_:.0f}x less likely than #1)"
            lines.append(f"  {i + 1}. {_code(s_, e_)}  {_p(p_)}{rel}")
        lines.append(f"Not any episode of this show: {_p(obs.get('null'))}")
    hint = gap_hint(plan, entry, context)
    if hint:
        lines.append("Disc context: " + hint + ".")
    if obs and detail:
        lines.append("Clip by clip (time in file, leader so far, not-an-episode probability, text):")
        for c in obs["clips"]:
            text = (c["text"].strip() or "(silent)")[:70]
            if c.get("top"):
                lines.append(f"  {int(c['t']):>5}s  {_code(*c['top'][:2])} {_p(c['top'][2])}  "
                             f"null {_p(c.get('null'))}  \"{text}\"")
            else:
                lines.append(f"  {int(c['t']):>5}s  {text}")
    elif obs:
        lines.append("For the clip-by-clip transcripts call inspect_file with detail=true.")
    return "\n".join(lines)


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
    context = context_for(state_dir)
    latest: dict[str, tuple[dict, int, dict]] = {}
    for plan in recent_plans(state_dir):
        for i, e in enumerate(plan["entries"]):
            latest[e["file"]] = (plan, i, e)
    out = []
    for path, (plan, i, e) in latest.items():
        if e["confident"] or e.get("error") or not os.path.isfile(path):
            continue
        hint = gap_hint(plan, e, context)
        out.append(dict(id=file_id(plan["id"], i), name=_name(e), disc=disc_dir_of(path).name,
                        show=plan["show"], best_guess=e.get("best_guess"),
                        best_score=e.get("best_score"), made=plan["created"],
                        gap=(hint or "").split(":")[0]))
    out.sort(key=lambda r: r["made"], reverse=True)
    return out[:limit]


def describe_uncertain(rows: list[dict]) -> str:
    if not rows:
        return "No unidentified files waiting."
    lines = [f"{len(rows)} file(s) not identified (newest first):"]
    for r in rows:
        score = f"{r['best_score']:.3f}" if r["best_score"] is not None else "-"
        tag = {"fits a gap": "  [LIKELY AN EPISODE: best guess fills a gap on its disc]",
               "fits the disc's sequence": "  [LIKELY AN EPISODE: best guess continues its disc's run]",
               "does not fit a gap": "  [probably an extra: no gap on its disc fits]"}.get(r["gap"], "")
        lines.append(f"  {r['id']}  {r['name']}  disc '{r['disc']}'  show={r['show']}  "
                     f"best guess {r['best_guess'] or '?'} ({score}){tag}")
    lines.append("Why a file is unsure: inspect_file(file_id). Retry one with more samples: "
                 "rerun(file_id) (worth it mainly for the likely episodes).")
    return "\n".join(lines)


#: What the watcher's recorded outcomes mean, in words that can't be mistaken for a workflow step.
_OUTCOME_WORDS = {
    "triage_movie": "looks like a movie, not TV: left alone, nothing to do",
    "triage_review": "could not tell movie from TV: left alone, needs a person to look",
    "no_show": "could not work out which show this is: needs a person to say",
}


def _disc_state(state: dict, folder: Path) -> dict:
    return state.get(str(folder), {})


def _ready_plans(state_dir: Path, min_score: float, limit: int = 10) -> list[dict]:
    """Plans that have files which could be moved right now (newest first).

    Uses the same checks as `plan_moves`, so stale plans (sources gone, targets
    already there) never show up.
    """
    out = []
    for plan in reversed(recent_plans(state_dir)):
        info = plan_moves(plan, min_score)
        if not (info["moves"] or info["blocked"]):
            continue
        out.append(dict(id=plan["id"], show=plan["show"], n=len(info["moves"]),
                        blocked=len(info["blocked"]),
                        folders={disc_dir_of(src) for src, _ in info["moves"]}))
        if len(out) >= limit:
            break
    return out


def _moved_from(state_dir: Path) -> dict[Path, int]:
    """Disc folder -> how many files any applied plan has already moved out of it."""
    moved: dict[Path, int] = {}
    for plan in cache.list_plans(state_dir):
        for r in (plan.get("applied") or {}).get("results", []):
            if r["renamed"]:
                d = disc_dir_of(r["file"])
                moved[d] = moved.get(d, 0) + 1
    return moved


#: ARM job statuses in words that can't be lumped together ("ripping" is only the first stage).
_ARM_ACTIVE = {
    "ripping": "Ripping now (the disc is in the drive)",
    "transcoding": "Transcoding (the disc is already out; ARM is converting the rips)",
    "info": "Reading the disc (just inserted)",
    "waiting": "Waiting to start",
}


def _names(items: list[str], cap: int = 8) -> str:
    return ", ".join(items[:cap]) + (f", +{len(items) - cap} more" if len(items) > cap else "")


def queue(state_dir: Path, folders: list[Path], arm_db: Path | None = None,
          min_score: float = 0.99) -> str:
    """What needs attention, sorted by what to do about it.

    Staging folders are split into: ready to apply (shown under their plans),
    still being written by ARM, left over from discs already filed (ignorable),
    movies waiting for a person (diskid doesn't file movies), and the few that
    need a person to look.
    """
    try:
        state = json.loads((state_dir / "state.json").read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    ready = _ready_plans(state_dir, min_score)
    ready_folders = {d for r in ready for d in r["folders"]}
    moved = _moved_from(state_dir)

    busy, leftovers, movies, look = [], [], [], []
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
            if d in ready_folders:
                continue                                    # listed under Ready to apply
            if arm_db and arm.job_active(arm_db, d):
                busy.append((d.name, n))
            elif moved.get(d):
                leftovers.append((d.name, n))
            elif outcome == "triage_movie":
                movies.append((d.name, n))
            else:
                if outcome == "done":
                    why = f"checked, {s.get('identified')} of {s.get('total')} files identified, none ready"
                elif outcome in _OUTCOME_WORDS:
                    why = _OUTCOME_WORDS[outcome]
                else:
                    why = "not processed yet (still settling, or still being written)"
                look.append(f"  [{root.name}] {d.name}: {n} file(s) - {why}")

    jobs = arm.recent_jobs(arm_db, 8) if arm_db else []
    active = {k: [j for j in jobs if j["status"] == k] for k in _ARM_ACTIVE}
    done = [j for j in jobs if j["status"] in arm.FINISHED]

    parts = [f"{len(active['ripping'])} ripping", f"{len(active['transcoding'])} transcoding"] if jobs else []
    parts.append(f"{len(ready)} plan(s) ready to apply ({sum(r['n'] for r in ready)} files)")
    parts.append(f"{len(look)} folder(s) need a person")
    if movies:
        parts.append(f"{len(movies)} movie(s) waiting for you")
    if leftovers:
        parts.append(f"{len(leftovers)} folder(s) of leftovers (ignorable)")
    out = ["Summary: " + " | ".join(parts)]

    if jobs:
        out.append("ARM (the ripper):")
        for status, words in _ARM_ACTIVE.items():
            if active[status]:
                out.append(f"  {words}: " + "; ".join(
                    f"job {j['job_id']} {j.get('label') or j.get('title')} ({j.get('no_of_titles')} titles)"
                    for j in active[status]))
        for j in done[:2]:
            extra = f", errors: {j['errors']}" if j.get("errors") else ""
            out.append(f"  Last finished: job {j['job_id']} {j.get('label') or j.get('title')} - "
                       f"{j['status']}{extra}")
    if ready:
        out.append("Ready to apply (identified files that can be moved into the library now):")
        for r in ready:
            folder = _names(sorted(d.name for d in r["folders"]), 2)
            out.append(f"  plan {r['id']} ({r['show']}): " + (
                f"{r['blocked']} file(s) score too low to apply; rerun them first" if r["blocked"] and not r["n"]
                else f"{r['n']} file(s) ready, from '{folder}' -> preview_apply('{r['id']}')"))
    else:
        out.append("Ready to apply: nothing. No plan has files waiting to be moved.")
    if busy:
        out.append("Still being written by ARM (not ready yet): "
                   + _names([f"{name} ({n})" for name, n in busy]))
    if look:
        out.append("Needs a person:")
        out += look
    if movies:
        out.append("Movies waiting for you to file (diskid doesn't file movies): "
                   + _names([name for name, _ in movies]))
    if leftovers:
        files = sum(n for _, n in leftovers)
        out.append(f"Leftovers from discs already filed ({len(leftovers)} folders, {files} files; "
                   "usually extras or titles that matched nothing; safe to ignore): "
                   + _names([f"{name} ({n})" for name, n in leftovers], 6)
                   + ". unidentified lists them with hints.")
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
