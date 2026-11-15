"""diskid command line interface.

    diskid identify FILE --show NAME [--apply]
    diskid identify-disc DIR [--show NAME] [--apply]
    diskid apply MAPPING_ID
    diskid watch DIR [--dry-run]        # long-running service mode

Dry runs (no --apply) cache their proposed renames and print a mapping id;
`diskid apply MAPPING_ID` replays them later without re-running whisper/matching.

This is a thin wrapper over the library API in `diskid.identify`.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from . import cache, layout, match, subdb
from .identify import (  # noqa: F401  (re-exported for existing importers)
    DiscResult,
    FileResult,
    Identifier,
    IdentifierConfig,
    ProgressEvent,
    ShowNotFoundError,
    apply_rename,
    episode_filename,
    rename_to_target,
    sample_clips,
    sanitize_filename,
)
from .match import ConsensusScore


def log(msg: str) -> None:
    tqdm.write(msg, file=sys.stderr)


def print_ranked(ranked: list[ConsensusScore], top: int = 10) -> None:
    print(f"{'season':>6} {'episode':>7} {'score':>8} {'offset':>7} {'files':>5}")
    for ep in ranked[:top]:
        print(f"{ep.season:>6} {ep.episode:>7} {ep.score:>8.5f} "
              f"{ep.offset:>+7.1f} {ep.n_files:>5}")


def ranked_json(ranked: list[ConsensusScore], top: int = 10) -> list[dict]:
    return [vars(ep) for ep in ranked[:top]]


def make_identifier(args, quiet: bool = False) -> Identifier:
    """Build an Identifier from parsed CLI args, with tqdm/stderr progress."""
    config = IdentifierConfig(
        model=args.model, language=args.language, clip_duration=args.clip_duration,
        max_clips=args.clips, min_clips=args.min_clips,
        output_dir=args.output_dir, template=args.template, start=getattr(args, "start", None), min_score=args.min_score, eps=args.eps,
        order=args.order, zip_path=args.zip,
        stt_url=args.stt_url, stt_model=args.stt_model,
        stt_api_key=os.environ.get("DISKID_STT_API_KEY"), stt_timeout=args.stt_timeout,
    )
    state: dict = {"bar": None}

    def progress(ev: ProgressEvent) -> None:
        if ev.stage in ("loading_show", "trying_show"):
            log(ev.message)
        elif ev.stage == "transcribing":
            bar = state["bar"]
            if bar is None or bar.total != ev.max_clips or ev.clip == 1:
                if bar is not None:
                    bar.close()
                name = ev.path.name if ev.path else "clips"
                bar = state["bar"] = tqdm(total=ev.max_clips, desc=f"{name} [transcribing]",
                                          unit="clip", leave=False)
            bar.set_postfix_str((ev.text or "")[:60].replace("\n", " "))
            bar.update(1)
        elif ev.stage == "file_done" and state["bar"] is not None:
            state["bar"].close()
            state["bar"] = None

    return Identifier(config, progress=None if quiet else progress)


def cmd_identify(args) -> int:
    ident = make_identifier(args)
    try:
        res = ident.identify_file(args.video, args.show)
    except ShowNotFoundError as e:
        sys.exit(f"error: {e} (pass --show)")
    if res.status == "failed":
        sys.exit(f"error: {res.error}")
    for w in res.warnings:
        log(f"warning: {w}")

    renamed, fail_reason = False, ""
    if args.apply and res.identified:
        out = apply_rename(res)
        renamed, fail_reason = out.renamed, out.reason

    mapping_id = None
    if not args.apply:
        mapping_id = cache.save_mapping(
            "identify", res.show.title, [cache.entry_for(res)], show_imdb=res.show.imdb_id)

    note = ""
    if res.episode and res.episode.order == "tmdb" and (
            (res.episode.season, res.episode.episode)
            != (res.episode.imdb_season, res.episode.imdb_episode)):
        note = f" [was imdb S{res.episode.imdb_season:02d}E{res.episode.imdb_episode:02d}]"

    if args.json:
        print(json.dumps({
            "file": str(args.video),
            "show": res.show.title,
            "show_id": res.show.imdb_id,
            "ranked": ranked_json(res.candidates),
            "confident": res.identified,
            "target": res.target_rel,
            "renamed": renamed,
            "mapping_id": mapping_id,
        }, indent=2))
    else:
        print_ranked(res.candidates)
        best = res.best
        verdict = "confident" if res.identified else "NOT confident"
        print(f"\nbest match: S{best.season:02d}E{best.episode:02d} "
              f"(score {best.score:.3f}, {verdict})")
        if renamed:
            print(f"renamed to: {res.target_rel}{note}")
        elif res.target_rel and args.apply:
            print(f"not renamed: {fail_reason}")
        elif res.target_rel:
            print(f"would rename to: {res.target_rel}{note} (mapping id: {mapping_id}, "
                  f"apply with: diskid apply {mapping_id})")
    return 0 if res.identified else 2


def cmd_identify_disc(args) -> int:
    ident = make_identifier(args)
    if not any(p.is_file() for p in args.dir.rglob("*")):
        sys.exit(f"error: no files in {args.dir}")
    try:
        disc = ident.identify_disc(args.dir, args.show)
    except ShowNotFoundError as e:
        sys.exit(f"error: {e} (pass --show)")

    results = []
    mapping_entries = []
    for res in disc.files:
        rel = str(res.path.relative_to(args.dir))
        note = ""
        renamed = False
        if res.error:
            note = f" ({res.error})"
        elif res.identified:
            if res.episode.order == "tmdb" and (
                    (res.episode.season, res.episode.episode)
                    != (res.episode.imdb_season, res.episode.imdb_episode)):
                note = f" [was imdb S{res.episode.imdb_season:02d}E{res.episode.imdb_episode:02d}]"
            if args.apply:
                out = apply_rename(res)
                renamed = out.renamed
                if not renamed:
                    note += f" ({out.reason})"
            else:
                note += " (dry run, pass --apply)"
        elif res.best:
            note = (f" (best: S{res.best.season:02d}E{res.best.episode:02d} "
                    f"{res.best.score:.3f}, low confidence)")
        results.append({
            "file": rel,
            "episode": res.episode.code if res.episode else None,
            "score": next((c.score for c in res.candidates
                           if res.episode and c.key == (res.episode.imdb_season,
                                                        res.episode.imdb_episode)), None),
            "best_score": res.best.score if res.best else None,
            "target": res.target_rel,
            "renamed": renamed,
            "_note": note,
        })
        mapping_entries.append(cache.entry_for(res))

    mapping_id = None
    if not args.apply:
        mapping_id = cache.save_mapping(
            "identify-disc", disc.show.title, mapping_entries,
            show_imdb=disc.show.imdb_id, source_dir=str(args.dir.resolve()))

    if args.json:
        for r in results:
            del r["_note"]
        print(json.dumps({
            "show": disc.show.title, "results": results, "mapping_id": mapping_id,
        }, indent=2))
    else:
        print(f"\n{'file':40s} {'episode':>8} {'score':>6}  target")
        for r in results:
            print(f"{r['file']:40s} {r['episode'] or '?':>8} "
                  f"{r['score'] if r['score'] is not None else 0:>6.3f}  "
                  f"{(r['target'] or '-') + r['_note']}")
        if mapping_id:
            print(f"\nmapping id: {mapping_id} (apply with: diskid apply {mapping_id})")
    return 0 if all(r["episode"] for r in results) else 2


def cmd_watch(args) -> int:
    import signal
    from .watch import DiscWatcher, WatchConfig, default_state_dir

    out = args.output_dir
    if out is not None and any(
            out.resolve() == d.resolve() or d.resolve() in out.resolve().parents
            for d in [*args.dir, *args.triage]):
        sys.exit("error: --output-dir must not be inside the watched directory "
                 "(its contents would be scanned as discs)")
    ident = make_identifier(args, quiet=True)
    watcher = DiscWatcher(ident, WatchConfig(
        watch_dir=args.dir, state_dir=args.state_dir or default_state_dir(),
        interval=args.interval, settle=args.settle, dry_run=args.dry_run,
        arm_db=args.arm_db, triage_dirs=args.triage,
    ))

    def on_signal(signum, _frame):
        logging.getLogger("diskid.watch").info("signal %d: shutting down", signum)
        watcher.stop()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def abort_if_stopping(ev: ProgressEvent) -> None:
        watcher.check_stop()

    ident.progress = abort_if_stopping
    watcher.run_forever()
    return 0


def cmd_apply(args) -> int:
    from .eventlog import EventLog

    events = EventLog(cache.default_state_dir() / "events.jsonl")
    ok = True
    for ref in args.plan:
        plan = cache.load_mapping(ref)
        if plan is None:
            sys.exit(f"error: no such plan: {ref}")
        if plan.get("applied"):
            log(f"note: plan {plan['id']} was already applied at {plan['applied']['at']}; "
                "retrying any entries still unmoved")
        results = cache.apply_plan(plan, events)
        ok = ok and all(r["renamed"] for r in results
                        if any(e["file"] == r["file"] and e.get("confident") for e in plan["entries"]))
        if args.json:
            print(json.dumps({"mapping_id": plan["id"], "show": plan["show"],
                              "results": results}, indent=2))
        else:
            for r in results:
                if r["renamed"]:
                    print(f"moved: {r['file']} -> {r['target']}")
                else:
                    print(f"skipped: {r['file']} ({r['reason']})")
    return 0 if ok else 2


def cmd_plans(args) -> int:
    plans = cache.list_plans()
    if args.json:
        print(json.dumps([{k: v for k, v in p.items() if k != "entries"}
                          | {"files": len(p["entries"])} for p in plans], indent=2))
        return 0
    if not plans:
        print(f"no plans in {cache.plans_dir()}")
    for p in plans:
        n_ok = sum(bool(e.get("confident")) for e in p["entries"])
        state = f"applied {p['applied']['at']}" if p.get("applied") else "PENDING"
        print(f"{p['id']}  {p['created']}  {p['command']:<13} {p['show'] or '?':<28} "
              f"{n_ok}/{len(p['entries'])} identified  {state}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="diskid", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, watch=False):
        if not watch:
            p.add_argument("--show", help="show name (if omitted, best-effort guessed from "
                                           "the folder name / container title tag)")
            p.add_argument("--apply", action="store_true",
                           help="rename confidently identified files in place")
            p.add_argument("--json", action="store_true", help="machine-readable output on stdout")
            p.add_argument("-s", "--start", type=float, default=None,
                           help="sample a single clip at this offset only (default: spread "
                                "--clips clips across the middle 70%% of the runtime)")
        p.add_argument("-d", "--clip-duration", type=float, default=8.0,
                       help="seconds per clip (default: 8)")
        p.add_argument("-c", "--clips", type=int, default=10,
                       help="maximum clips to sample before giving up -- sampling "
                            "stops early once --min-score is cleared (default: 10)")
        p.add_argument("-o", "--output-dir", type=Path, metavar="DIR",
                       default=Path(os.environ["DISKID_OUTPUT_DIR"])
                       if os.environ.get("DISKID_OUTPUT_DIR") else None,
                       help="root the --template is relative to; identified files are "
                            "moved under it (default: $DISKID_OUTPUT_DIR, else rename in "
                            "place, beside the source)")
        p.add_argument("--template", metavar="JINJA",
                       default=os.environ.get("DISKID_TEMPLATE") or layout.DEFAULT_TEMPLATE,
                       help=("destination path, relative to --output-dir. Variables: show, "
                             "year, season, episode, code, episode_title, ext, stem, order, "
                             "show_imdb, episode_imdb; filter |pad. Example: "
                             f"'{layout.LIBRARY_TEMPLATE}' "
                             f"(default: $DISKID_TEMPLATE, else '{layout.DEFAULT_TEMPLATE}')").replace("%", "%%"))
        p.add_argument("--min-clips", type=int, default=3,
                       help="clips to observe before sampling may stop early, even if "
                            "--min-score is already cleared (default: 3)")
        p.add_argument("-m", "--model", default="small",
                       help="local whisper model size (default: small); with --stt-url it "
                            "only picks the default server model name")
        p.add_argument("--stt-url", default=None, metavar="URL",
                       help="transcribe via an OpenAI-compatible server instead of local "
                            "faster-whisper, e.g. http://host:8000/v1 (API key, if any, from "
                            "$DISKID_STT_API_KEY)")
        p.add_argument("--stt-model", default=None,
                       help="model name to request from the server (default: "
                            "Systran/faster-whisper-<--model>)")
        p.add_argument("--stt-timeout", type=float, default=120.0,
                       help="seconds to wait per clip from the server (default: 120)")
        p.add_argument("-l", "--language", default=None,
                       help="language code, e.g. 'en' (default: autodetect)")
        p.add_argument("--order", choices=("tmdb", "imdb"), default="tmdb",
                       help="episode numbering for filenames: tmdb (Jellyfin's "
                            "default metadata source) or imdb (the archive's "
                            "aired order) (default: tmdb)")
        p.add_argument("--zip", type=Path, default=subdb.DEFAULT_ZIP,
                       help=f"OPUS raw subtitle archive (default: {subdb.DEFAULT_ZIP})")
        p.add_argument("--min-score", type=float, default=0.99,
                       help="confidence: minimum posterior probability for the top "
                            "episode, versus every other episode and \"not in "
                            "database\" (default: 0.99)")
        p.add_argument("--eps", type=float, default=match.STT_ERROR_RATE,
                       help="assumed STT word-error rate; not yet calibrated against "
                            "labeled data, so treat it as a hand tuning knob -- higher "
                            "eps discounts matched words and softens the penalty for "
                            "unmatched ones, giving a more skeptical/slower-to-commit "
                            f"classifier, lower eps the opposite (default: {match.STT_ERROR_RATE})")

    p_id = sub.add_parser("identify", help="identify a single video file")
    p_id.add_argument("video", type=Path)
    common(p_id)

    p_disc = sub.add_parser("identify-disc", help="identify every video in a ripped-disc folder")
    p_disc.add_argument("dir", type=Path)
    common(p_disc)

    p_watch = sub.add_parser(
        "watch", help="watch a directory of ripped discs (e.g. ARM's COMPLETED_PATH), "
                      "identify new ones, and rename/move them per --template")
    p_watch.add_argument("dir", type=Path, nargs="+",
                         help="folder(s) whose subfolders are discs; with ARM's staging layout "
                              "pass <staging>/tv and <staging>/unidentified")
    p_watch.add_argument("--dry-run", action="store_true",
                         help="identify and write reports, but never rename")
    p_watch.add_argument("--interval", type=float, default=30.0,
                         help="seconds between scans (default: 30)")
    p_watch.add_argument("--settle", type=float, default=120.0,
                         help="seconds a disc folder must be unmodified before it is "
                              "processed (default: 120)")
    p_watch.add_argument("--arm-db", type=Path, metavar="FILE",
                         default=Path(os.environ["DISKID_ARM_DB"])
                         if os.environ.get("DISKID_ARM_DB") else None,
                         help="ARM's arm.db: get each disc's show name from the volume label "
                              "ARM recorded (useful when ARM's own identification is off and "
                              "folders are named 'None (None)') (default: $DISKID_ARM_DB)")
    p_watch.add_argument("--triage", type=Path, action="append", metavar="DIR",
                         default=[Path(p) for p in os.environ.get("DISKID_TRIAGE_DIR", "").split()],
                         help="a folder of discs ARM filed as movies (e.g. <staging>/movies): "
                              "each is checked, and only TV-shaped ones (several similar-length "
                              "files, or a runtime that doesn't fit the claimed film) are "
                              "identified as a series; movies are left alone. Repeatable "
                              "(default: $DISKID_TRIAGE_DIR)")
    p_watch.add_argument("--state-dir", type=Path, default=None,
                         help="where state.json and reports/ go (default: $STATE_DIRECTORY "
                              "or ~/.local/state/diskid)")
    common(p_watch, watch=True)

    p_apply = sub.add_parser("apply", help="carry out saved plan(s) from an earlier dry run, "
                                           "without re-running identification")
    p_apply.add_argument("plan", nargs="+", metavar="PLAN",
                         help="plan id (see `diskid plans`) or path to a plan .json")
    p_apply.add_argument("--json", action="store_true", help="machine-readable output on stdout")

    p_plans = sub.add_parser("plans", help="list saved plans and whether they were applied")
    p_plans.add_argument("--json", action="store_true", help="machine-readable output on stdout")

    args = parser.parse_args()
    if args.command == "watch":
        # journald adds timestamps; keep lines plain
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                            stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, format="warning: %(message)s",
                            stream=sys.stderr)

    if args.command == "apply":
        sys.exit(cmd_apply(args))
    if args.command == "plans":
        sys.exit(cmd_plans(args))
    if args.command == "watch":
        for d in args.dir:
            if not d.is_dir():
                logging.getLogger("diskid.watch").warning(
                    "watch dir %s does not exist yet; will pick it up when it appears", d)
        if args.output_dir is None and all(not d.exists() for d in args.dir):
            log("note: no watch directory exists yet")
        if not args.zip.is_file():
            sys.exit(f"error: no such file: {args.zip}")
        sys.exit(cmd_watch(args))

    target = args.video if args.command == "identify" else args.dir
    if not (target.is_file() if args.command == "identify" else target.is_dir()):
        sys.exit(f"error: no such {'file' if args.command == 'identify' else 'directory'}: {target}")
    if not args.zip.is_file():
        sys.exit(f"error: no such file: {args.zip}")

    handler = cmd_identify if args.command == "identify" else cmd_identify_disc
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
