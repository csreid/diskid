#!/usr/bin/env python3
"""Calibration harness (plan.md's "Calibration" section).

Runs diskid's identification pipeline against an already correctly-named
library and checks two things:

1. Accuracy: does the top-ranked episode match the filename?
2. Calibration: does "posterior 0.99" actually correspond to ~99% empirical
   accuracy, or does the model saturate to floating-point 1.0 regardless of
   whether it's right?

Mismatches are reported for manual review rather than treated as automatic
failures -- some shows' filenames are known to have a few mislabeled
episodes, so a disagreement could be a real miss or a bad label.

Usage: uv run python scripts/calibrate.py [--root DIR] [--suspect NAME ...]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

from diskid import Identifier, IdentifierConfig, ShowNotFoundError, imdb, subdb
from diskid.match import STT_ERROR_RATE

EPISODE_RE = re.compile(r"[Ss](\d+)[Ee](\d+)")
VIDEO_GLOBS = ("*.mkv", "*.mp4")


def parse_expected(video: Path) -> tuple[int, int] | None:
    m = EPISODE_RE.search(video.stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def run(args) -> list[dict]:
    identifier = Identifier(IdentifierConfig(
        model=args.model, max_clips=args.clips, clip_duration=args.clip_duration,
        min_score=args.min_score, eps=args.eps, zip_path=args.zip,
    ))
    print(f"whisper model={args.model} device={identifier.device}", file=sys.stderr)

    suspect_names = {s.lower() for s in args.suspect}
    show_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    results: list[dict] = []

    with open(args.out, "w") as out_f:
        for show_dir in show_dirs:
            show_name, _year = imdb.parse_title_year(show_dir.name)
            try:
                show = identifier.resolve_show(show_name)
            except LookupError as e:
                print(f"skip {show_dir.name}: {e}", file=sys.stderr)
                continue
            suspect = show_dir.name.lower() in suspect_names

            videos = sorted(p for g in VIDEO_GLOBS for p in show_dir.rglob(g))
            for video in tqdm(videos, desc=show_dir.name, unit="file"):
                expected = parse_expected(video)
                if expected is None:
                    continue

                try:
                    res = identifier.identify_file(video, show)
                except ShowNotFoundError as e:
                    print(f"skip {show_dir.name}: {e}", file=sys.stderr)
                    break
                top = res.best
                predicted = None
                if top is not None:  # calibration scores the best guess, confident or not
                    ref = identifier.episode_ref(show, top)
                    predicted = (ref.season, ref.episode)

                row = {
                    "show": show.title,
                    "file": str(video.relative_to(show_dir)),
                    "expected": list(expected),
                    "predicted": list(predicted) if predicted else None,
                    "score": top.score if top else None,
                    "null_probability": res.null_probability,
                    "n_clips": res.clips_used,
                    "correct": predicted == expected,
                    "suspect_labels": suspect,
                }
                results.append(row)
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()

    return results


def print_summary(results: list[dict]) -> None:
    reliable = [r for r in results if not r["suspect_labels"]]
    n, correct = len(reliable), sum(r["correct"] for r in reliable)
    print(f"\n{n} files (excluding suspect-label shows): {correct} correct "
          f"({correct / n:.1%})" if n else "\nno results")

    mismatches = [r for r in results if not r["correct"]]
    if mismatches:
        print("\nmismatches (review manually -- could be a real miss OR a bad label):")
        for r in mismatches:
            tag = "SUSPECT-LABEL" if r["suspect_labels"] else "REVIEW"
            print(f"  [{tag}] {r['show']} {r['file']}: expected "
                  f"S{r['expected'][0]:02d}E{r['expected'][1]:02d}, predicted {r['predicted']} "
                  f"(score={r['score']}, null_prob={r['null_probability']:.3g}, "
                  f"n_clips={r['n_clips']})")

    print("\nreliability (claimed posterior vs empirical accuracy):")
    edges = [0.5, 0.9, 0.99, 0.999, 0.9999, 1.0 + 1e-9]
    prev = 0.0
    for edge in edges:
        bucket = [r for r in reliable if r["score"] is not None and prev <= r["score"] < edge]
        if bucket:
            acc = sum(r["correct"] for r in bucket) / len(bucket)
            print(f"  score in [{prev:.4f}, {edge:.4f}): n={len(bucket):3d}  "
                  f"empirical accuracy={acc:.1%}")
        prev = edge


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path.home() / "Videos" / "Media" / "shows")
    ap.add_argument("--model", default="small")
    ap.add_argument("-c", "--clips", type=int, default=10)
    ap.add_argument("-d", "--clip-duration", type=float, default=8.0)
    ap.add_argument("--zip", type=Path, default=subdb.DEFAULT_ZIP)
    ap.add_argument("--min-score", type=float, default=0.99)
    ap.add_argument("--eps", type=float, default=STT_ERROR_RATE,
                     help="assumed STT word-error rate, hand-tunable pending real calibration")
    ap.add_argument("--out", type=Path, default=Path("calibration_results.jsonl"))
    ap.add_argument("--suspect", nargs="*", default=["Community"],
                     help="show folder names with known-unreliable labels (default: Community)")
    args = ap.parse_args()

    results = run(args)
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
